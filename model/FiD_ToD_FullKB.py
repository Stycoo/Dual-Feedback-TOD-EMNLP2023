# generator and ranker encoder sharing.
# ranking method: attribute value set(top-k entity) score --> entity score

from dataclasses import dataclass
import torch
import transformers
from transformers.modeling_outputs import (
    ModelOutput,
    BaseModelOutput,
)
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from typing import Optional, Tuple, Union
import types
from loss import kldivloss

@dataclass
class MySeq2SeqLMOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    retriever_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    generator_cross_attention: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class FiDT5(transformers.T5ForConditionalGeneration):
    def __init__(self, config, model_args):
        super().__init__(config)
        self.model_args = model_args
        self.wrap_encoder()

        if self.model_args.joint_generator_retriever_cotrain and self.model_args.joint_generator_retriever_cotrain_kd_type == 'cross_attn':
            self.overwrite_forward_crossattention()
            self.reset_score_storage()

        if self.model_args.joint_generator_ranker_consist_constrain or \
                (self.model_args.joint_generator_ranker_entRelevant_cal_type == 'cosin'
                 and self.model_args.joint_generator_retriever_cotrain_V2):
            self.linear_layer = nn.Linear(self.config.d_model, self.config.d_model)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        if input_ids != None:
            # inputs might have already be resized in the generate method
            if input_ids.dim() == 3:
                self.encoder.n_passages = input_ids.size(1)

        return self._forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    def get_generator_score(self, seq_logits, target, target_mask=None, n_entity=1):
        # seq_logits: bsz*n_entity, seq_len, vocab_size
        seq_logits = F.log_softmax(seq_logits, -1)
        pad_token_id = self.config.pad_token_id
        target.masked_fill_(target == -100, pad_token_id)
        target_logits = seq_logits.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1) # bsz*n_entity, seq_len
        if target_mask is None:
            target_mask = (~target.eq(0)).int()
        scores = (target_logits * target_mask).sum(-1) / target_mask.sum(-1)
        scores = scores.view(scores.size(0)//n_entity, n_entity)
        return scores

    def affine_transformation(self, input_features, mask):
        length = torch.sum(mask, dim=1)
        trans_tmp = F.relu(self.linear_layer(input_features))  # batch
        trans_tmp = trans_tmp * mask.unsqueeze(-1).float()
        trans_emb = torch.sum(trans_tmp, dim=1)
        return trans_emb * (1 / length.unsqueeze(-1))

    def get_entityRelevance_context_response(self, context_feature, decoder_feature, context_mask, label_mask, context_grain):
        '''
        :param context_feature: bsz*n_entity, text_len, hidden_size
        :param decoder_feature: bsz, beam_size, cand_len, hidden_size
        :param context_mask: bsz, n_entity, text_len
        :param label_mask: bsz*beam_size, cand_len
        :param context_grain: 'coarse' or 'fine'
        :return:
        '''
        batch_size, beam_size, cand_len, _ = decoder_feature.size()
        context_len = context_feature.size(1)
        decoder_feature = decoder_feature.view(batch_size*beam_size, cand_len, -1)

        if context_grain == 'coarse':
            context_feature = context_feature.view(batch_size, self.encoder.n_passages*context_len, self.config.d_model)
            context_mask = context_mask.view(batch_size, -1)  # bsz, n_entity*context_len
        else:
            context_mask = context_mask.view(-1, context_len)  # bsz*n_entity, context_len

        context_feature = self.affine_transformation(context_feature, context_mask) # coarse: bsz, hidden_size; fine: bsz*n_enity, hidden_size
        if context_grain == 'coarse':
            context_feature = context_feature.unsqueeze(1)
        else:
            context_feature = context_feature.view(batch_size, self.encoder.n_passages, -1)
        context_feature = context_feature.unsqueeze(1).expand(-1, beam_size, -1, -1)

        decoder_feature = self.affine_transformation(decoder_feature, label_mask)  # bsz*beam_size, hidden_size
        decoder_feature = decoder_feature.view(batch_size, beam_size, -1)
        decoder_feature = decoder_feature.unsqueeze(2).expand(-1, -1, context_feature.size(2), -1)

        cosin_sim = torch.cosine_similarity(context_feature, decoder_feature, dim=-1)  # bsz, beam_size, n_entity
        return cosin_sim

    def _shift_right(self, input_ids):
        pad_token_id = self.config.pad_token_id
        decoder_start_token_id = self.config.decoder_start_token_id

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
        return shifted_input_ids

    @torch.no_grad()
    def generator_lm_logits(self, decoder_input_ids, hidden_states, attention_mask, labels_mask=None, **kwargs):
        if self.model_args.joint_generator_retriever_cotrain_kd_type == 'cross_attn':
            self.reset_score_storage()

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            attention_mask=labels_mask,
            **kwargs,
        )
        decoder_cross_attention_scores = None
        if self.model_args.joint_generator_retriever_cotrain_kd_type == 'cross_attn':
            decoder_cross_attention_scores = self.get_crossattention_scores(context_mask=attention_mask,
                                                                            target_mask=labels_mask)

        sequence_output = decoder_outputs[0]
        singleContextENTpair_lm_logits = self.lm_head(sequence_output)
        return singleContextENTpair_lm_logits, decoder_cross_attention_scores, sequence_output

    @torch.no_grad()
    def sample_from_model(self, input_ids, attention_mask, encoder_outputs=None):
        batch_size = input_ids.size(0)
        candidate_id = self.generate_wrap(input_ids=input_ids, attention_mask=attention_mask,
                                          encoder_outputs=encoder_outputs,
                                          num_return_sequences=self.model_args.beam_size,
                                          num_beam_groups=self.model_args.beam_size,
                                          diversity_penalty=self.model_args.diversity_pen,
                                          num_beams=self.model_args.beam_size,
                                          max_length=self.model_args.response_maxlength,
                                          )
        return candidate_id.view(batch_size, self.model_args.beam_size, -1)

    def pad2max_len(self, input_tensor, max_len, pad_id):
        pad_size = max_len - input_tensor.shape[-1]
        if input_tensor.dim() == 2:
            pad_tensor = torch.full([input_tensor.shape[0], pad_size], pad_id,
                                    device=input_tensor.device).long()
        elif input_tensor.dim() == 3:
            pad_tensor = torch.full([input_tensor.shape[0], input_tensor.shape[1], pad_size], pad_id,
                                    device=input_tensor.device).long()
        else:
            raise ValueError
        return torch.cat([input_tensor, pad_tensor], dim=-1)

    def form_ngram(self, input_tensor, n=2):
        """
        input_tensor: batch x sample_num x seq_len
        return: batch x seq_len-3 x 4
        """
        bsz, cand_num, seq_len = input_tensor.size()
        seq_len_clip = seq_len - n + 1
        input_tensor_repeated = input_tensor[:, :, None, :].repeat(1, 1, seq_len_clip, 1)
        help_matrix_1 = torch.triu(torch.ones(seq_len, seq_len))
        help_matrix_2 = torch.triu(torch.ones(seq_len, seq_len), diagonal=n)
        help_matrix = (help_matrix_1 - help_matrix_2)[:seq_len_clip].bool()[None, None, :, :]
        ret_tensor = torch.masked_select(input_tensor_repeated, help_matrix.to(input_tensor.device))
        return ret_tensor.view(bsz, cand_num, seq_len_clip, n)

    @torch.no_grad()
    def torch_bleu(self, ref_tensor, hyp_tensor, pad_id, n_gram=2):
        """
        Calculates n-gram precision with brevity penalty. contributed by jinulee-v

        ref_tensor: batch x seq_len1
        hyp_tensor: batch x sample_num x seq_len2
        """
        # Determine batch size, sample count(=beam size), n-gram
        bsz, sample_num, _ = hyp_tensor.size()
        n = min(min(n_gram, ref_tensor.size(-1)), hyp_tensor.size(-1))

        # Generate masks
        ref_padding = (~(ref_tensor == pad_id)).float()
        ref_ngram_mask = torch.arange(0, ref_padding.size(1), device=ref_padding.device) * torch.ones_like(ref_padding)
        ref_ngram_mask = torch.where(
            ref_ngram_mask < (torch.sum(ref_padding, dim=-1, keepdim=True) - n + 1),
            ref_padding, torch.zeros_like(ref_padding)
        )[:, :ref_ngram_mask.size(-1) - n + 1]
        hyp_padding = (~(hyp_tensor == pad_id)).float()
        hyp_ngram_mask = torch.arange(0, hyp_padding.size(-1), device=hyp_padding.device) * torch.ones_like(hyp_padding)
        hyp_ngram_mask = torch.where(
            hyp_ngram_mask < (torch.sum(hyp_padding, dim=-1, keepdim=True) - n + 1),
            hyp_padding, torch.zeros_like(hyp_padding)
        )[:, :, :hyp_ngram_mask.size(-1) - n + 1]

        # Get n-grams
        ref_tensor = ref_tensor * ref_padding  # mask out paddings
        hyp_tensor = hyp_tensor * hyp_padding
        ref_tensor = ref_tensor[:, None, :].repeat(1, sample_num, 1)  # readjust ref size to match sys
        input_tensor1_ngram = self.form_ngram(ref_tensor, n).float()
        input_tensor2_ngram = self.form_ngram(hyp_tensor, n).float()  # batch x sample_num x seq_len-(n-1) x n

        # Calculate similarity matrix
        sim_matrix = (torch.norm(  # Calculate L2 norm to find if N-gram in `sys`` is present in `ref``
            input_tensor2_ngram.unsqueeze(3) - input_tensor1_ngram.unsqueeze(2),
            p=2, dim=-1
        ) == 0.0).to(torch.float)
        sim_matrix *= hyp_ngram_mask.unsqueeze(3) * ref_ngram_mask.unsqueeze(1).unsqueeze(2)
        sim_matrix = torch.sum(torch.max(sim_matrix, dim=-1).values, dim=-1)

        # Brevity penalty
        ref_len = torch.sum(ref_padding, dim=-1, keepdim=True)
        hyp_len = torch.sum(hyp_padding, dim=-1)
        bp = torch.exp(1 - (ref_len / hyp_len))
        bp = torch.where(ref_len >= hyp_len, bp, torch.ones_like(bp))

        return sim_matrix / torch.sum(hyp_ngram_mask, dim=-1) * bp  # batch x sample_num

    @torch.no_grad()
    def torch_f1(self, hyp, gold_entity_ids, entity_set):
        '''
        :param hyp: bsz, beam_size, cand_len
        :param gold_entity_ids: bsz, n_gold_entity
        :param entity_set: entity_num, value_len
        :return:
        '''

        gold_entity_mask = (~gold_entity_ids.eq(-100)).int()
        gold_entity_num = gold_entity_mask.sum(-1)  # bsz
        gold_entity_ids = gold_entity_ids * gold_entity_mask

        entity_set_mask = (~entity_set.eq(-100)).int()
        entity_textlen = entity_set_mask.sum(-1)

        ent_textlen_min = min(min(entity_textlen), hyp.size(-1))
        ent_textlen_max = min(max(entity_textlen), hyp.size(-1))
        ent_num, ent_textlen = entity_set.size()

        # find entity info in hyp responses
        hyp_entities = []
        for n in range(ent_textlen_min, ent_textlen_max+1):
            hyp_ngram = self.form_ngram(hyp, n)  # bsz, beam_size, cand_len-n+1, n
            bsz, beam_size, ngram_num, _ = hyp_ngram.size()
            padding_tensor = torch.zeros(size=(bsz, beam_size, ngram_num, ent_textlen - n),
                                         device=hyp_ngram.device) - 100
            hyp_ngram = torch.cat([hyp_ngram, padding_tensor], dim=-1)  # bsz, beam_size, ngram_num, ent_textlen
            hyp_ngram_used = hyp_ngram.unsqueeze(-2).expand(-1, -1, -1, ent_num,
                                                            -1)  # bsz, beam_size, ngram_num, ent_num, ent_textlen
            entity_set_used = entity_set.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(bsz, beam_size, ngram_num, -1,
                                                                                       -1)
            hyp_entity = (~(
                ((~(hyp_ngram_used - entity_set_used).eq(0))).int().sum(-1).eq(0).int().sum(2).eq(0))).int().unsqueeze(
                -1)  # bsz, beam_size, ent_num, 1
            hyp_entities.append(hyp_entity)

        hyp_entities_flag = torch.cat(hyp_entities, dim=-1).sum(-1)  # bsz, beam_size, ent_num
        hyp_entities_flag = (~hyp_entities_flag.eq(0)).int()

        # cal F1
        one_hot_label = torch.zeros_like(hyp_entities_flag)
        gold_entity_ids = gold_entity_ids.unsqueeze(1).expand(-1, hyp.size(1), -1)
        one_hot_label = one_hot_label.scatter_(-1, gold_entity_ids, torch.ones_like(hyp_entities_flag))

        positive_hyp_ent = ((hyp_entities_flag - one_hot_label).eq(0) * hyp_entities_flag).sum(-1)  # bsz, beam_size
        gold_entity_num = gold_entity_num.unsqueeze(-1).expand(-1, hyp.size(1))
        gold_entity_num = torch.where(~gold_entity_num.eq(0), gold_entity_num,
                                      (torch.ones_like(gold_entity_num) * torch.iinfo(
                                          gold_entity_num.dtype).max))
        precision = positive_hyp_ent / gold_entity_num

        hyp_entity_num = hyp_entities_flag.sum(-1)
        hyp_entity_num = torch.where(~hyp_entity_num.eq(0), hyp_entity_num,
                                     (torch.ones_like(hyp_entity_num) * torch.iinfo(
                                         hyp_entity_num.dtype).max))
        recall = positive_hyp_ent / hyp_entity_num
        f1 = (2 * precision * recall) / (precision + recall + 1e-6)
        return f1

    def ranking_loss(self, cos_distance, bleu_distance):
        # equivalent to initializing TotalLoss to 0
        # here is to avoid that some special samples will not go into the following for loop
        margin = 0.01
        ones = torch.ones(cos_distance.size(), device=cos_distance.device)
        loss_func = torch.nn.MarginRankingLoss(0.0)
        total_loss = loss_func(cos_distance, cos_distance, ones)

        # candidate loss
        n = cos_distance.size(1)
        for i in range(1, n):
            pos_score = cos_distance[:, :-i]
            neg_score = cos_distance[:, i:]
            same_mask = (torch.abs(bleu_distance[:, :-i] - bleu_distance[:, i:]) > margin).float()
            ones = torch.ones(pos_score.size(), device=cos_distance.device)
            loss_func = torch.nn.MarginRankingLoss(margin * i, reduction='none')  # batch x i
            marginal_loss = loss_func(pos_score, neg_score, ones)
            if same_mask.sum() > 0:
                total_loss += (marginal_loss * same_mask).sum() / same_mask.sum()

        return total_loss

    def mml_loss(self, lm_logits, topk_log_probs, labels, loss_mask):
        # Converting the tensors datatype to float
        lm_logits = lm_logits.float()
        topk_log_probs = topk_log_probs.float()

        topk = lm_logits.shape[1]
        # [B, K, L, V]
        lm_log_probs = F.log_softmax(lm_logits, dim=-1)

        # Converting the loss mask to bool tensor and inverting it.
        # and replacing the -1 in labels with 0
        labels = labels.masked_fill(~loss_mask.to(torch.bool), 0)

        # labels: [B, L] -> tiled_labels: [B, K, L]
        tiled_labels = torch.repeat_interleave(labels.unsqueeze(1), topk, dim=1)

        # [B, K, L] -> [B, K, L, 1]
        tiled_labels = tiled_labels.unsqueeze(-1)

        # [B, K, L, 1]
        gold_log_probs = torch.gather(lm_log_probs, dim=-1, index=tiled_labels)

        # [B, K, L, 1] -> [B, K, L]
        gold_log_probs = gold_log_probs.squeeze(-1)

        # [B, K, L]
        joint_gold_log_probs = topk_log_probs.unsqueeze(-1) + gold_log_probs

        # [B, L] -> [B, K, L]
        marginal_gold_log_probs = torch.logsumexp(joint_gold_log_probs, dim=1)

        # Applying mask to marginal loss
        lm_loss = -1 * torch.sum(marginal_gold_log_probs * loss_mask) / torch.sum(loss_mask)
        return lm_loss

    def _forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            decoder_input_ids: Optional[torch.LongTensor] = None,
            decoder_attention_mask: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            decoder_head_mask: Optional[torch.FloatTensor] = None,
            cross_attn_head_mask: Optional[torch.Tensor] = None,
            encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,

            # additional
            entity_score=None,
            res_entity_mask=None,
            gold_entities=None,
            attribute_values=None,
            step=None,
            **kwargs
    ) -> Union[Tuple[torch.FloatTensor], MySeq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        # get decoder inputs from shifting lm labels to the right
        if labels is not None:
            decoder_input_ids = self._shift_right(labels)

        assert attention_mask.dim() == 3
        batch_size, n_entity, text_length = attention_mask.size()

        encoder_attention_mask = attention_mask.view(batch_size, n_entity * text_length)

        decoder_cross_attention_scores = None
        if self.model_args.joint_generator_retriever_cotrain and \
                self.model_args.joint_generator_retriever_cotrain_kd_type == 'cross_attn':
            self.reset_score_storage()

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = decoder_outputs[0]
        decoder_feature = sequence_output
        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output) # if lm_marginalize lm_logits size: bsz * n_entity, target_len, vocab_size

        loss = None; retriever_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)).contiguous(), labels.view(-1))

            # distill knowledge from generator to retriever
            if self.model_args.joint_generator_retriever_cotrain and step > self.model_args.joint_generator_retriever_start_step \
                    and step < self.model_args.joint_generator_retriever_end_step:
                assert decoder_input_ids.size(0) == hidden_states.size(0)
                if self.model_args.joint_generator_ranker_target_grain == 'seq':  # whole target sequence
                    labels_mask = ~labels.eq(-100)
                elif self.model_args.joint_generator_ranker_target_grain == 'entity_token':  # focus on entity tokens
                    labels_mask = res_entity_mask
                else:
                    raise ValueError

                batch_size, text_length, hidden_size = hidden_states.size()
                hidden_states_used = hidden_states.view(batch_size, self.encoder.n_passages, text_length // self.encoder.n_passages,
                                                        hidden_size)
                encoder_attention_mask_used = encoder_attention_mask.view(batch_size, self.encoder.n_passages,
                                                                          text_length // self.encoder.n_passages)

                if self.model_args.joint_generator_retriever_cotrain_kd_type == 'cross_attn':
                    decoder_cross_attention_scores = self.get_crossattention_scores(context_mask=encoder_attention_mask,
                                                                                    target_mask=labels_mask)
                    generator_scores = decoder_cross_attention_scores

                elif self.model_args.joint_generator_retriever_cotrain_kd_type == 'ent_margin':
                    batch_size, text_length, hidden_size = hidden_states.size()
                    hidden_states_used = hidden_states.view(batch_size * self.encoder.n_passages,
                                                            text_length // self.encoder.n_passages,
                                                            hidden_size)
                    singleContextENTpair_lm_logits, _, _ = self.generator_lm_logits(
                        decoder_input_ids.repeat_interleave(self.encoder.n_passages, dim=0), hidden_states_used,
                        encoder_attention_mask.view(batch_size * self.encoder.n_passages, -1))
                    generator_scores = self.get_generator_score(singleContextENTpair_lm_logits,
                                                               labels.repeat_interleave(self.encoder.n_passages, dim=0),
                                                               labels_mask.repeat_interleave(self.encoder.n_passages, dim=0),
                                                               self.encoder.n_passages)  # bsz, n_entity


                elif self.model_args.joint_generator_retriever_cotrain_kd_type == 'mml':
                    singleContextENTpair_lm_logits = []
                    for i in range(self.encoder.n_passages):
                        singleContextENTpair_lm_logit, _, _ = self.generator_lm_logits(
                            decoder_input_ids, hidden_states_used[:, i, :, :], encoder_attention_mask_used[:, i, :],
                            **kwargs)
                        singleContextENTpair_lm_logits.append(singleContextENTpair_lm_logit)
                    singleContextENTpair_lm_logits = torch.cat(singleContextENTpair_lm_logits, dim=0)

                else:
                    raise ValueError

                # hard negative feedback & hard positive feedback
                if self.model_args.joint_generator_retriever_cotrain_PNFB:
                    cand_ids = self.sample_from_model(input_ids, attention_mask, encoder_outputs)
                    batch_size, beam_size, cand_len = cand_ids.size()
                    cand_ids_mask = ~cand_ids.eq(0)
                    labels = (~(labels.eq(-100))).int() * labels

                    # scoring by evaluation function: BLEU or F1
                    init_rank_ind = (torch.arange(0, beam_size) * torch.ones(size=(batch_size, beam_size))).to(labels.device)
                    if self.model_args.joint_generator_retriever_cotrain_cand_eval_metric == 'bleu':
                        eval_score = self.torch_bleu(labels, cand_ids, 0,
                                                     self.model_args.joint_generator_retriever_eval_ngram)
                        negative_mask = (eval_score < self.model_args.joint_generator_retriever_eval_neg_threshold).int()
                        # bsz, beam_size
                        positive_mask = (eval_score > self.model_args.joint_generator_retriever_eval_pos_threshold).int()

                        eval_indice = eval_score.sort(dim=-1)[1]
                        eval_rank = torch.zeros_like(eval_score).to(eval_indice.device)
                        eval_rank = torch.scatter(eval_rank, 1, eval_indice, init_rank_ind)

                    elif self.model_args.joint_generator_retriever_cotrain_cand_eval_metric == 'f1':
                        eval_score = self.torch_f1(cand_ids, gold_entities, attribute_values)
                        negative_mask = (eval_score < self.model_args.joint_generator_retriever_eval_neg_threshold).int()
                        positive_mask = (eval_score > self.model_args.joint_generator_retriever_eval_pos_threshold).int()

                        eval_indice = eval_score.sort(dim=-1)[1]
                        eval_rank = torch.zeros_like(eval_score).to(eval_indice.device)
                        eval_rank = torch.scatter(eval_rank, 1, eval_indice, init_rank_ind)

                    elif self.model_args.joint_generator_retriever_cotrain_cand_eval_metric == 'bleu_f1':
                        eval_bleu_score = self.torch_bleu(labels, cand_ids, 0, self.model_args.joint_generator_retriever_eval_ngram)
                        negative_bleu_mask = (eval_bleu_score < self.model_args.joint_generator_retriever_eval_neg_threshold).int()
                        positive_bleu_mask = (eval_bleu_score > self.model_args.joint_generator_retriever_eval_pos_threshold).int()

                        eval_f1_score = self.torch_f1(cand_ids, gold_entities, attribute_values)
                        negative_f1_mask = (eval_f1_score < self.model_args.joint_generator_retriever_eval_neg_threshold).int()
                        positive_f1_mask = (eval_f1_score > self.model_args.joint_generator_retriever_eval_pos_threshold).int()

                        negative_mask = negative_bleu_mask * negative_f1_mask
                        positive_mask = positive_bleu_mask * positive_f1_mask

                        eval_bleu_indice = eval_bleu_score.sort(dim=-1)[1]
                        eval_bleu_rank = torch.zeros_like(eval_bleu_score).to(eval_bleu_indice.device)
                        eval_bleu_rank = torch.scatter(eval_bleu_rank, 1, eval_bleu_indice, init_rank_ind)

                        eval_f1_indice = eval_f1_score.sort(dim=-1)[1]
                        eval_f1_rank = torch.zeros_like(eval_f1_score).to(eval_f1_indice.device)
                        eval_f1_rank = torch.scatter(eval_f1_rank, 1, eval_f1_indice, init_rank_ind)

                    else:
                        raise ValueError

                    cand_lm_logits, decoder_cross_attention_cand_scores, _ = self.generator_lm_logits(
                        cand_ids.view(-1, cand_len),  # bsz*beam_size, cand_len
                        hidden_states.repeat_interleave(beam_size, dim=0),
                        encoder_attention_mask.repeat_interleave(beam_size, dim=0),
                        labels_mask=cand_ids_mask.view(-1, cand_len), **kwargs)
                    # decoder_cross_attention_cand_scores: bsz*beam_size, n_entity
                    generator_cand_scores = self.get_generator_score(cand_lm_logits, cand_ids.view(-1, cand_len),
                                                                   n_entity=beam_size)  # bsz, beam_size
                    generator_cand_indice = generator_cand_scores.sort(dim=-1)[1]
                    generator_cand_rank = torch.zeros_like(generator_cand_scores).to(generator_cand_scores.device)
                    generator_cand_rank = torch.scatter(generator_cand_rank, 1, generator_cand_indice, init_rank_ind)

                    # hard negative candidate response mining
                    if self.model_args.hard_negative_mining_method == 'rankVariance_GP_FE':
                        _, hard_negative_indices = (eval_rank - generator_cand_rank).min(dim=-1)
                        _, hard_positive_indices = (eval_rank - generator_cand_rank).max(dim=-1)
                    elif self.model_args.hard_negative_mining_method == 'argmin_factEval':
                        _, hard_negative_indices = eval_score.min(dim=-1)
                        _, hard_positive_indices = eval_score.max(dim=-1)
                    elif self.model_args.hard_negative_mining_method == 'rankVariance_FEs':
                        _, hard_negative_indices = (eval_f1_rank - eval_bleu_rank).min(dim=-1)
                        _, hard_positive_indices = (eval_f1_rank - eval_bleu_rank).max(dim=-1)
                    else:
                        raise ValueError
                    negative_mask_used = torch.gather(negative_mask, 1, hard_negative_indices.unsqueeze(-1))  # bsz, 1
                    positive_mask_used = torch.gather(positive_mask, 1, hard_positive_indices.unsqueeze(-1))  # bsz, 1

                    # rectifying feedback
                    if self.model_args.joint_generator_retriever_cotrain_kd_type == 'ent_margin':
                        # hard negative entity margin prob
                        hard_negative_indices = hard_negative_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, cand_len)  # bsz, 1, cand_len
                        hard_negative_cand = torch.gather(cand_ids, 1, hard_negative_indices).squeeze(1)  # bsz, cand_len
                        batch_size, text_length, hidden_size = hidden_states.size()
                        hidden_states_used = hidden_states.view(batch_size * self.encoder.n_passages,
                                                                text_length // self.encoder.n_passages,
                                                                hidden_size)
                        hn_singleContextENTpair_lm_logits, _, _ = self.generator_lm_logits(
                            hard_negative_cand.repeat_interleave(self.encoder.n_passages, dim=0), hidden_states_used,
                            encoder_attention_mask.view(batch_size * self.encoder.n_passages, -1))
                        hard_negative_generator_scores = self.get_generator_score(hn_singleContextENTpair_lm_logits,
                                                                    hard_negative_cand.repeat_interleave(self.encoder.n_passages,dim=0),
                                                                    n_entity=self.encoder.n_passages)  # bsz, n_entity

                        # hard positive entity margin prob
                        hard_positive_indices = hard_positive_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, cand_len)
                        hard_positive_cand = torch.gather(cand_ids, 1, hard_positive_indices).squeeze(1)
                        hp_singleContextENTpair_lm_logits, _, _ = self.generator_lm_logits(
                            hard_positive_cand.repeat_interleave(self.encoder.n_passages, dim=0), hidden_states_used,
                            encoder_attention_mask.view(batch_size * self.encoder.n_passages, -1))
                        hard_positive_generator_scores = self.get_generator_score(hp_singleContextENTpair_lm_logits,
                                                                                  hard_positive_cand.repeat_interleave(self.encoder.n_passages,dim=0),
                                                                                  n_entity=self.encoder.n_passages)  # bsz, n_entity


                    # cross attention score
                    elif self.model_args.joint_generator_retriever_cotrain_kd_type == 'cross_attn':
                        decoder_cross_attention_cand_scores = torch.cat(decoder_cross_attention_cand_scores,
                                                                        dim=0)  # bsz*beam_size, n_entity
                        hard_negative_indices = hard_negative_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.encoder.n_passages)  # bsz, 1, cand_len
                        hard_negative_generator_scores = torch.gather(
                            decoder_cross_attention_cand_scores.view(batch_size, beam_size, self.encoder.n_passages),
                                     1, hard_negative_indices).squeeze(1)  # bsz, n_entity

                        hard_positive_indices = hard_positive_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.encoder.n_passages)
                        hard_positive_generator_scores = torch.gather(
                            decoder_cross_attention_cand_scores.view(batch_size, beam_size, self.encoder.n_passages),
                            1, hard_positive_indices).squeeze(1)

                    else:
                        raise ValueError

                    # retriever training loss
                    if self.model_args.joint_generator_retriever_cotrain_PNFB_loss_type == 'ranking_loss':
                        # (retriever_entity_score, hard_positive_score, hard_negtive_score)
                        gt_res_distance = kldivloss(entity_score, generator_scores.detach(), reduce=False)
                        gt_res_distance = gt_res_distance.sum(dim=-1)

                        hard_positive_distance = kldivloss(entity_score, hard_positive_generator_scores.detach(), reduce=False)
                        hard_positive_res_distance = hard_positive_distance * positive_mask_used  # bsz, n_entity
                        hard_positive_res_distance = hard_positive_res_distance.sum(dim=-1)

                        hard_negative_res_distance = kldivloss(entity_score, hard_negative_generator_scores.detach(), reduce=False)
                        hard_negative_res_distance = hard_negative_res_distance * negative_mask_used
                        hard_negative_res_distance = hard_negative_res_distance.sum(dim=-1)

                        loss_func = torch.nn.MarginRankingLoss(0.0)
                        ones = torch.ones(gt_res_distance.size(), device=gt_res_distance.device)
                        retriever_loss_1 = loss_func(hard_negative_res_distance, gt_res_distance, ones)
                        retriever_loss_2 = loss_func(hard_negative_res_distance, hard_positive_res_distance, ones)
                        retriever_constrain_loss = retriever_loss_1 + retriever_loss_2

                    elif self.model_args.joint_generator_retriever_cotrain_PNFB_loss_type == 'triplet_loss':
                        postive_res_distance = kldivloss(entity_score, generator_scores.detach(), reduce=False)
                        postive_res_distance = postive_res_distance.sum(dim=-1)
                        negative_res_distance = kldivloss(entity_score, hard_negative_generator_scores.detach(), reduce=False)  # bsz, n_entity
                        negative_res_distance = negative_res_distance * negative_mask_used
                        negative_res_distance = negative_res_distance.sum(dim=-1)
                        loss_func = torch.nn.MarginRankingLoss(0.0)
                        ones = torch.ones(postive_res_distance.size(), device=postive_res_distance.device)
                        retriever_constrain_loss = loss_func(negative_res_distance, postive_res_distance, ones)

                    else:
                        raise ValueError
                    retriever_loss = retriever_constrain_loss + kldivloss(entity_score, generator_scores.detach())
                    loss += retriever_loss


                else:
                    if self.model_args.joint_generator_retriever_cotrain_kd_type == 'mml':
                        assert singleContextENTpair_lm_logits is not None
                        singleContextENTpair_lm_logits = singleContextENTpair_lm_logits.view(batch_size,
                                                                                             self.encoder.n_passages,
                                                                                             decoder_input_ids.size(-1),
                                                                                             -1)
                        entity_log_prob = F.log_softmax(entity_score, dim=-1)
                        retriever_loss = self.mml_loss(singleContextENTpair_lm_logits, entity_log_prob, labels, labels_mask)
                    else:
                        retriever_loss = kldivloss(entity_score, generator_scores.detach())
                        loss += retriever_loss

        return MySeq2SeqLMOutput(
            loss=loss,
            retriever_loss=retriever_loss,
            logits=lm_logits,
            generator_cross_attention=decoder_cross_attention_scores,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,

        entity_score=None,
        step=None,
        **kwargs
    ):
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "decoder_input_ids": decoder_input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,

            "entity_score": entity_score,
            "step": step,
        }

    @torch.no_grad()
    def generate_wrap(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_start_token_id: Optional[int] = None,
        **model_kwargs
    ) -> torch.LongTensor:

        assert attention_mask.dim() == 3

        if model_kwargs['encoder_outputs'] is None:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **model_kwargs)
            model_kwargs['encoder_outputs'] = outputs
        batch_size, n_entity, text_len = attention_mask.size()

        model_kwargs['attention_mask'] = attention_mask

        decoder_start_token_id = (decoder_start_token_id if decoder_start_token_id is not None
                                  else self.config.decoder_start_token_id)

        decoder_input_ids = torch.full(
            (batch_size, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )

        res = self.generate(decoder_input_ids, **model_kwargs)
        return res


    def inference(self, input_ids, attention_mask, entity_score, response_max_length, **kwargs,):

        ans_outputs = None
        res_outputs = self.generate_wrap(input_ids=input_ids,
                                        max_length=response_max_length,
                                        attention_mask=attention_mask,
                                        entity_score=entity_score,
                                        encoder_outputs=None,
                                        **kwargs)

        return ans_outputs, res_outputs, None

    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap T5 encoder to obtain a Fusion-in-Decoder model.
        """
        self.encoder = EncoderWrapper(self.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap Fusion-in-Decoder encoder, useful to load T5 weights.
        """
        self.encoder = self.encoder.encoder
        block = []
        for mod in self.encoder.block:
            block.append(mod.module)
        block = nn.ModuleList(block)
        self.encoder.block = block

    def load_t5(self, t5):
        self.unwrap_encoder()
        self.encoder.load_state_dict(t5.encoder.state_dict())
        self.wrap_encoder()

        self.decoder.load_state_dict(t5.decoder.state_dict())
        self.lm_head.load_state_dict(t5.lm_head.state_dict())
        self.shared.load_state_dict(t5.shared.state_dict())

    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.encoder.encoder.block:
            mod.use_checkpoint = use_checkpoint

    def reset_score_storage(self):
        """
        Reset score storage, only used when cross-attention scores are saved
        to train a retriever.
        """
        for mod in self.decoder.block:
            mod.layer[1].EncDecAttention.score_storage = None

    def get_crossattention_scores(self, context_mask=None, target_mask=None):
        """
        Cross-attention scores are aggregated to obtain a single scalar per
        passage. This scalar can be seen as a similarity score between the
        question and the input passage. It is obtained by averaging the
        cross-attention scores obtained on the first decoded token over heads,
        layers, and tokens of the input passage.

        More details in Distilling Knowledge from Reader to Retriever:
        https://arxiv.org/abs/2012.04584.
        """
        scores = []
        n_passages = self.encoder.n_passages

        for mod in self.decoder.block:
            scores.append(mod.layer[1].EncDecAttention.score_storage.unsqueeze(2))
        scores = torch.cat(scores, dim=2)
        bsz, n_heads, n_layers, target_max_len, _ = scores.size()
        # batch_size, n_head, n_layers, target_max_len, n_passages, text_maxlength
        scores = scores.view(bsz, n_heads, n_layers, target_max_len, n_passages, -1)
        scores = scores.mean(dim=[1, 2])  # batch_size, target_max_len, n_passages, text_maxlength

        if target_mask is not None:
            target_mask = target_mask[:, :, None, None]
            scores = scores.masked_fill(~target_mask, 0.).sum(dim=1) / target_mask.sum(dim=1)
            # (bs, n_passages, context_maxlength+db_maxlength)
        else:
            scores = scores[:, 0, :, :]  # (bs, n_passages, context_maxlength+db_maxlength)

        avg_attention_mask = context_mask.view(bsz, n_passages, -1)
        if self.model_args.generator_distill_retriever_pooling == "avg_wo_context":
            decoder_cross_attention_scores = scores.masked_fill(~avg_attention_mask, 0.)[:, :,
                                             self.model_args.generator_text_maxlength:].sum(dim=2) / \
                                             avg_attention_mask[:, :, self.model_args.generator_text_maxlength:].sum(dim=2)
            # (bs, db_num)
        else:
            decoder_cross_attention_scores = scores.masked_fill(~avg_attention_mask, 0.).sum(
                dim=2) / avg_attention_mask.sum(dim=2)  # (bs, db_num)

        return decoder_cross_attention_scores

    def overwrite_forward_crossattention(self):
        """
        Replace cross-attention forward function, only used to save
        cross-attention scores.
        """
        for mod in self.decoder.block:
            attn = mod.layer[1].EncDecAttention
            attn.forward = types.MethodType(cross_attention_forward, attn)


class EncoderWrapper(torch.nn.Module):
    """
    Encoder Wrapper for T5 Wrapper to obtain a Fusion-in-Decoder model.
    """
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()
        self.encoder = encoder
        apply_checkpoint_wrapper(self.encoder, use_checkpoint)
        self.main_input_name = None

    def forward(self, input_ids=None, attention_mask=None, return_dict=True, **kwargs,):
        '''
        node here refer to selected entity attribute value set
        total_length = n_passages * passage_length
        full kb: all input vars related to KB are under the definition of full kb
        kb_adj_matrix: bsz, n_node, n_node (n_node: attribute value set of all entities in dataset)
        nodes_span: bsz, n_entity, n_attr, 3 (3: node id, st_ind, ed_ind)
        '''
        assert input_ids.dim() == 3
        batch_size, self.n_passages, text_length = input_ids.size()
        input_ids = input_ids.view(batch_size * self.n_passages, text_length)
        attention_mask = attention_mask.view(batch_size * self.n_passages, text_length)

        outputs = self.encoder(input_ids, attention_mask, return_dict=True)
        outputs.last_hidden_state = outputs.last_hidden_state.view(batch_size, self.n_passages * text_length, -1)

        return outputs


class CheckpointWrapper(torch.nn.Module):
    """
    Wrapper replacing None outputs by empty tensors, which allows the use of
    checkpointing.
    """
    def __init__(self, module, use_checkpoint=False):
        super().__init__()
        self.module = module
        self.use_checkpoint = use_checkpoint

    def forward(self, hidden_states, attention_mask, position_bias, **kwargs):
        if self.use_checkpoint and self.training:
            kwargs = {k: v for k, v in kwargs.items() if v is not None}
            def custom_forward(*inputs):
                output = self.module(*inputs, **kwargs)
                empty = torch.tensor(
                    [],
                    dtype=torch.float,
                    device=output[0].device,
                    requires_grad=True)
                output = tuple(x if x is not None else empty for x in output)
                return output

            output = torch.utils.checkpoint.checkpoint(
                custom_forward,
                hidden_states,
                attention_mask,
                position_bias
            )
            output = tuple(x if x.size() != 0 else None for x in output)
        else:
            output = self.module(hidden_states, attention_mask, position_bias, **kwargs)
        return output


def apply_checkpoint_wrapper(t5stack, use_checkpoint):
    """
    Wrap each block of the encoder to enable checkpointing.
    """
    block = []
    for mod in t5stack.block:
        wrapped_mod = CheckpointWrapper(mod, use_checkpoint)
        block.append(wrapped_mod)
    block = nn.ModuleList(block)
    t5stack.block = block


def cross_attention_forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
    """
    Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
    """
    # Input is (batch_size, seq_length, dim)
    # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
    # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
    batch_size, seq_length = hidden_states.shape[:2]

    real_seq_length = seq_length

    if past_key_value is not None:
        assert (
                len(past_key_value) == 2
        ), f"past_key_value should have 2 past states: keys and values. Got {len(past_key_value)} past states"
        real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

    key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

    def shape(states):
        """projection"""
        return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

    def unshape(states):
        """reshape"""
        return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

    def project(hidden_states, proj_layer, key_value_states, past_key_value):
        """projects hidden states correctly to key/query states"""
        if key_value_states is None:
            # self-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(hidden_states))
        elif past_key_value is None:
            # cross-attn
            # (batch_size, n_heads, seq_length, dim_per_head)
            hidden_states = shape(proj_layer(key_value_states))

        if past_key_value is not None:
            if key_value_states is None:
                # self-attn
                # (batch_size, n_heads, key_length, dim_per_head)
                hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
            else:
                # cross-attn
                hidden_states = past_key_value
        return hidden_states

    # get query states
    query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

    # get key/value states
    key_states = project(
        hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
    )
    value_states = project(
        hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
    )

    # compute scores
    scores = torch.matmul(
        query_states, key_states.transpose(3, 2)
    )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

    if position_bias is None:
        if not self.has_relative_attention_bias:
            position_bias = torch.zeros(
                (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
            )
            if self.gradient_checkpointing and self.training:
                position_bias.requires_grad = True
        else:
            position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

        # if key and values are already calculated
        # we want only the last query position bias
        if past_key_value is not None:
            position_bias = position_bias[:, :, -hidden_states.size(1):, :]

        if mask is not None:
            position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

    if self.pruned_heads:
        mask = torch.ones(position_bias.shape[1])
        mask[list(self.pruned_heads)] = 0
        position_bias_masked = position_bias[:, mask.bool()]
    else:
        position_bias_masked = position_bias

    scores += position_bias_masked

    if self.score_storage is None:
        self.score_storage = scores

    attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
        scores
    )  # (batch_size, n_heads, seq_length, key_length)

    attn_weights = nn.functional.dropout(
        attn_weights, p=self.dropout, training=self.training
    )  # (batch_size, n_heads, seq_length, key_length)

    # Mask heads if we want to
    if layer_head_mask is not None:
        attn_weights = attn_weights * layer_head_mask

    attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
    attn_output = self.o(attn_output)

    present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
    outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

    if output_attentions:
        outputs = outputs + (attn_weights,)

    return outputs