import torch
import transformers
from pathlib import Path
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
import torch.distributed as dist
import numpy as np
import json
import copy
import os

from options import Options
import util
import simcse_model
import FiD_ToD_FullKB as model
import data_loader

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def retriever_embedding_db(model, dataloader):
    """embedding all db text"""
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for k, batch in enumerate(dataloader):
            (index, text_ids, text_mask, text_token_type, attr_mask) = batch
            embeddings = model(input_ids=text_ids.long().cuda(), attention_mask=text_mask.long().cuda(),
                               token_type_ids=text_token_type.long().cuda(), output_hidden_states=True,
                               return_dict=True,
                               sent_emb=True).pooler_output
            all_embeddings.append(embeddings.cpu())
        all_embeddings = torch.cat(all_embeddings, dim=0)  # (all_db_num, hidden_size)
    model.train()
    return all_embeddings


def get_all_dbs_inputs(db_dataloader):
    all_db_ids = []
    all_db_mask = []
    all_db_token_type = []
    all_db_attr_mask = []
    for k, batch in enumerate(db_dataloader):
        (index, text_ids, text_mask, text_token_type, attr_mask) = batch
        all_db_ids.append(text_ids)
        all_db_mask.append(text_mask)
        if text_token_type is not None:
            all_db_token_type.append(text_token_type)
        if attr_mask is not None:
            all_db_attr_mask.append(attr_mask)

    all_db_ids = torch.cat(all_db_ids, 0)
    all_db_mask = torch.cat(all_db_mask, 0)
    all_db_token_type = torch.cat(all_db_token_type, 0) if len(all_db_token_type) > 0 else None
    all_db_attr_mask = torch.cat(all_db_attr_mask, 0) if len(all_db_attr_mask) > 0 else None

    return all_db_ids, all_db_mask, all_db_token_type, all_db_attr_mask


def concat_context_and_dbs_input(context_input, dbs_input):
    context_input = context_input.unsqueeze(1).repeat(1, dbs_input.size(1), 1)
    return torch.cat([context_input, dbs_input], dim=2)


def train(generator_model, generator_tokenizer, generator_optimizer, generator_scheduler, generator_db_collator,
          retriever_model, retriever_tokenizer, retriever_optimizer, retriever_scheduler, retriever_db_collator,
          step, train_dial_dataset, eval_dial_dataset, test_dial_dataset, dial_collator, db_dataset,
          opt, best_dev_score, checkpoint_path, attribute_values):
    if opt.is_main:
        try:
            tb_logger = torch.utils.tensorboard.SummaryWriter(Path(opt.checkpoint_dir) / opt.name)
        except:
            tb_logger = None
            logger.warning('Tensorboard is not available.')

    torch.manual_seed(opt.local_rank + opt.seed)  # different seed for different sampling depending on local_rank
    train_dial_sampler = RandomSampler(train_dial_dataset)  # if load_data use global rank and world size to distribute load，we dont need DistributedSampler
    train_dial_dataloader = DataLoader(
        train_dial_dataset,
        sampler=train_dial_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=10,
        collate_fn=dial_collator
    )
    db_sampler = SequentialSampler(db_dataset)
    generator_db_dataloader = DataLoader(db_dataset,
                                         sampler=db_sampler,
                                         batch_size=222,
                                         drop_last=False,
                                         num_workers=10,
                                         collate_fn=generator_db_collator)
    retriever_db_dataloader = DataLoader(db_dataset,
                                         sampler=db_sampler,
                                         batch_size=222,
                                         drop_last=False,
                                         num_workers=10,
                                         collate_fn=retriever_db_collator)
    generator_all_dbs_ids, generator_all_dbs_mask, _, _ = get_all_dbs_inputs(generator_db_dataloader)
    retriever_all_dbs_ids, retriever_all_dbs_mask, retriever_all_dbs_token_type, _ = get_all_dbs_inputs(retriever_db_dataloader)

    curr_loss = 0.0
    curr_retrieval_loss = 0.0
    epoch = 0
    generator_model.train()
    retriever_model.train()
    training_steps = min(opt.total_steps, opt.end_eval_step)
    retriever_all_dbs_embeddings = retriever_embedding_db(retriever_model,
                                                          retriever_db_dataloader)
    while step < training_steps:
        epoch += 1
        for i, batch in enumerate(tqdm(train_dial_dataloader)):
            step += 1

            if opt.use_gt_dbs is False and (step - 1) % opt.db_emb_update_steps == 0:
                retriever_all_dbs_embeddings = retriever_embedding_db(retriever_model, retriever_db_dataloader)
            elif opt.use_gt_dbs is True:
                if opt.use_retriever_for_gt is True and (step - 1) % opt.db_emb_update_steps == 0:
                    retriever_all_dbs_embeddings = retriever_embedding_db(retriever_model,
                                                                          retriever_db_dataloader)  # no grad
                elif opt.use_retriever_for_gt is False:
                    retriever_all_dbs_embeddings = None

            (index, resp_ori_input_ids, resp_ori_mask, \
            generator_context_input_ids, generator_context_mask, \
            retriever_context_input_ids, retriever_context_mask, retriever_context_token_type, \
            resp_delex_mask, gt_db_idx, gold_entities) = batch

            if opt.use_gt_dbs is False:
                retriever_context_embeddings = retriever_model(input_ids=retriever_context_input_ids.long().cuda(),
                                                               attention_mask=retriever_context_mask.long().cuda(),
                                                               token_type_ids=retriever_context_token_type.long().cuda(),
                                                               output_hidden_states=True,
                                                               return_dict=True,
                                                               sent_emb=True).pooler_output  # have grad
                retriever_all_dbs_scores = torch.einsum("bd,nd->bn", retriever_context_embeddings.detach().cpu(),
                                                        retriever_all_dbs_embeddings)  # (bs, all_db_num)
                retriever_top_k_dbs_index = retriever_all_dbs_scores.sort(-1, True)[1][:, :opt.top_k_dbs].unsqueeze(2)
            else:
                if opt.use_retriever_for_gt:
                    retriever_context_embeddings = retriever_model(input_ids=retriever_context_input_ids.long().cuda(),
                                                                   attention_mask=retriever_context_mask.long().cuda(),
                                                                   token_type_ids=retriever_context_token_type.long().cuda(),
                                                                   output_hidden_states=True,
                                                                   return_dict=True,
                                                                   sent_emb=True).pooler_output  # have grad
                    retriever_all_dbs_scores = torch.einsum("bd,nd->bn", retriever_context_embeddings.detach().cpu(),
                                                            retriever_all_dbs_embeddings)  # (bs, all_db_num)
                    retriever_gt_dbs_scores = torch.gather(retriever_all_dbs_scores, 1, gt_db_idx.long())  # (bs, gt_db_num)
                    top_k_dbs = opt.top_k_dbs
                    retriever_top_k_dbs_index = retriever_gt_dbs_scores.sort(-1, True)[1][:, :top_k_dbs]  # (bs, top_k)
                    retriever_top_k_dbs_index = torch.gather(gt_db_idx, 1, retriever_top_k_dbs_index.long()).unsqueeze(
                        2)  # (bs, top_k, 1)
                else:
                    retriever_top_k_dbs_index = gt_db_idx.unsqueeze(2)  # (bs, n_entity, 1)

            # get top-k db generator inputs and concat with context inputs and forward into generator model
            bsz = retriever_top_k_dbs_index.size(0)
            generator_db_len = generator_all_dbs_ids.size(-1)
            generator_top_k_dbs_ids = torch.gather(generator_all_dbs_ids.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                   retriever_top_k_dbs_index.long().repeat(1, 1, generator_db_len))
            generator_top_k_dbs_mask = torch.gather(generator_all_dbs_mask.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                    retriever_top_k_dbs_index.long().repeat(1, 1, generator_db_len))

            generator_context_top_k_dbs_input_ids = concat_context_and_dbs_input(generator_context_input_ids,
                                                                                 generator_top_k_dbs_ids)
            generator_context_top_k_dbs_mask = concat_context_and_dbs_input(generator_context_mask,
                                                                            generator_top_k_dbs_mask)

            if opt.use_gt_dbs is False:
                retriever_db_len = retriever_all_dbs_ids.size(-1)
                retriever_top_k_dbs_ids = torch.gather(retriever_all_dbs_ids.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                       retriever_top_k_dbs_index.long().repeat(1, 1, retriever_db_len))
                retriever_top_k_dbs_mask = torch.gather(retriever_all_dbs_mask.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                        retriever_top_k_dbs_index.long().repeat(1, 1, retriever_db_len))

                retriever_top_k_dbs_token_type = None
                if retriever_all_dbs_token_type is not None:
                    retriever_top_k_dbs_token_type = torch.gather(
                        retriever_all_dbs_token_type.unsqueeze(0).repeat(bsz, 1, 1), 1,
                        retriever_top_k_dbs_index.long().repeat(1, 1, retriever_db_len))

                retriever_top_k_dbs_embeddings = retriever_model(
                    input_ids=retriever_top_k_dbs_ids.view(-1, retriever_db_len).long().cuda(),
                    attention_mask=retriever_top_k_dbs_mask.view(-1, retriever_db_len).long().cuda(),
                    token_type_ids=retriever_top_k_dbs_token_type.view(-1, retriever_db_len).long().cuda()
                                                        if retriever_top_k_dbs_token_type is not None else None,
                    output_hidden_states=True,
                    return_dict=True,
                    sent_emb=True).pooler_output.view(bsz, opt.top_k_dbs, -1)  # have grad
                retriever_top_k_dbs_scores = torch.einsum("bad,bkd->bak", retriever_context_embeddings.unsqueeze(1),
                                                          retriever_top_k_dbs_embeddings).squeeze(1)  # (bs, top_k)
            else:
                if opt.use_retriever_for_gt:
                    retriever_db_len = retriever_all_dbs_ids.size(-1)
                    retriever_top_k_dbs_ids = torch.gather(retriever_all_dbs_ids.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                           retriever_top_k_dbs_index.long().repeat(1, 1,
                                                                                                   retriever_db_len))
                    retriever_top_k_dbs_mask = torch.gather(retriever_all_dbs_mask.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                            retriever_top_k_dbs_index.long().repeat(1, 1,
                                                                                                    retriever_db_len))
                    retriever_top_k_dbs_token_type = None
                    if retriever_all_dbs_token_type is not None:
                        retriever_top_k_dbs_token_type = torch.gather(
                            retriever_all_dbs_token_type.unsqueeze(0).repeat(bsz, 1, 1), 1,
                            retriever_top_k_dbs_index.long().repeat(1, 1, retriever_db_len))

                    top_k_dbs = opt.top_k_dbs
                    retriever_top_k_dbs_embeddings = retriever_model(
                        input_ids=retriever_top_k_dbs_ids.view(-1, retriever_db_len).long().cuda(),
                        attention_mask=retriever_top_k_dbs_mask.view(-1, retriever_db_len).long().cuda(),
                        token_type_ids=retriever_top_k_dbs_token_type.view(-1, retriever_db_len).long().cuda()
                        if retriever_top_k_dbs_token_type is not None else None,
                        output_hidden_states=True,
                        return_dict=True,
                        sent_emb=True).pooler_output.view(bsz, top_k_dbs, -1)  # have grad
                    retriever_top_k_dbs_scores = torch.einsum("bad,bkd->bak", retriever_context_embeddings.unsqueeze(1),
                                                              retriever_top_k_dbs_embeddings).squeeze(1)  # (bs, top_k)
                else:
                    retriever_top_k_dbs_scores = None

            generator_response = generator_model(
                input_ids=generator_context_top_k_dbs_input_ids.long().cuda(),
                attention_mask=generator_context_top_k_dbs_mask.cuda(),
                labels=resp_ori_input_ids.long().cuda(),
                return_dict=True,

                entity_score=retriever_top_k_dbs_scores if opt.joint_generator_retriever_cotrain or
                                                           opt.joint_generator_retriever_cotrain_V2 else None,
                res_entity_mask=resp_delex_mask.cuda(),
                gold_entities=gold_entities.long().cuda(),
                attribute_values=attribute_values.long().cuda(),
                step=step,
            )
            generator_loss = generator_response.loss
            retriever_loss = generator_response.retriever_loss

            train_loss = generator_loss
            train_loss = train_loss / opt.accumulation_steps
            train_loss.backward()

            if step % opt.accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(generator_model.parameters(), opt.clip)
                generator_optimizer.step()
                generator_scheduler.step()
                generator_model.zero_grad()
                if retriever_loss is not None:
                    torch.nn.utils.clip_grad_norm_(retriever_model.parameters(), opt.clip)
                    retriever_optimizer.step()
                    retriever_scheduler.step()
                    retriever_model.zero_grad()

            train_loss = util.average_main(train_loss, opt)
            curr_loss += train_loss.item()
            curr_retrieval_loss += retriever_loss / opt.accumulation_steps if retriever_loss is not None else 0.0

            if (step - 1) % opt.eval_freq == 0 and step > opt.start_eval_step:
                if opt.is_main:
                    logger.warning("Start evaluation")
                dev_score, dev_metric = evaluate(generator_model, eval_dial_dataset,
                                                 dial_collator, generator_tokenizer, opt,
                                                 generator_all_dbs_ids, generator_all_dbs_mask,
                                                 retriever_model, retriever_all_dbs_embeddings, step)
                test_score, test_metric = evaluate(generator_model, test_dial_dataset,
                                                   dial_collator, generator_tokenizer, opt,
                                                   generator_all_dbs_ids, generator_all_dbs_mask,
                                                   retriever_model, retriever_all_dbs_embeddings, step)
                if opt.is_main:
                    logger.warning("Continue training")
                generator_model.train()
                retriever_model.train()
                if opt.is_main:
                    if dev_score > best_dev_score:
                        best_dev_score = dev_score
                        util.save(generator_model, generator_optimizer, generator_scheduler, step, best_dev_score,
                                  opt, checkpoint_path, 'generator_best_dev')
                        util.save(retriever_model, retriever_optimizer, retriever_scheduler, step, best_dev_score,
                                  opt, checkpoint_path, 'retriever_best_dev')
                        if opt.use_gt_dbs is False or (opt.use_gt_dbs is True and opt.use_retriever_for_gt is True):
                            np.save(checkpoint_path / "checkpoint" / "retriever_best_dev" /
                                    "retriever_all_dbs_embeddings.npy", retriever_all_dbs_embeddings.numpy())
                        metric_path = checkpoint_path / "checkpoint" / "generator_best_dev" / 'metric.json'
                        final_metric = {"val": copy.deepcopy(dev_metric), "test": copy.deepcopy(test_metric)}

                        with open(metric_path, 'w', encoding='utf-8') as fout:
                            json.dump(final_metric, fout, indent=4)
                    log = f"{step} / {training_steps} |"
                    # train
                    log += f" Train Loss: {curr_loss / opt.eval_freq * opt.accumulation_steps:.3f} |"
                    log += f" Train retrieval loss: {curr_retrieval_loss / opt.eval_freq * opt.accumulation_steps:.4f} \n|"

                    # dev
                    log += "Evaluation: "
                    for key, val in dev_metric.items():
                        log += f"res {key}: {val:.2f} |"

                    log += "retriever "
                    for i in range(3, opt.eval_top_k_dbs+1):
                        if f"RECALL@{i}_turn_level" in dev_metric:
                            retriever_recall_dev = dev_metric[f"RECALL@{i}_turn_level"]
                            log += f"RECALL@{i}_turn_level: {retriever_recall_dev:.2f} |"

                    # test
                    log += "\nTest: "
                    for key, val in test_metric.items():
                        log += f"res {key}: {val:.2f} |"

                    log += "retriever "
                    for i in range(3, opt.eval_top_k_dbs + 1):
                        if f"RECALL@{i}_turn_level" in test_metric:
                            retriever_recall_test = test_metric[f"RECALL@{i}_turn_level"]
                            log += f"RECALL@{i}_turn_level: {retriever_recall_test:.2f} |"

                    log += f" glr: {generator_scheduler.get_last_lr()[0]:.5f}"
                    log += f" rlr: {retriever_scheduler.get_last_lr()[0]:.5f}"
                    logger.warning(log)
                    if tb_logger is not None:
                        tb_logger.add_scalar("Training Loss", curr_loss / opt.eval_freq, step)

                        for key, val in dev_metric.items():
                            tb_logger.add_scalar("Evaluation res {}".format(key), val, step)
                        for i in range(3, opt.eval_top_k_dbs + 1):
                            if f"RECALL@{i}_turn_level" in dev_metric:
                                tb_logger.add_scalar("Evaluation retriever RECALL@{}_turn_level".format(i),
                                                     retriever_recall_dev, step)

                        for key, val in test_metric.items():
                            tb_logger.add_scalar("Test res {}".format(key), val, step)
                        for i in range(3, opt.eval_top_k_dbs + 1):
                            if f"RECALL@{i}_turn_level" in test_metric:
                                tb_logger.add_scalar("Test retriever RECALL@{}_turn_level".format(i),
                                                     retriever_recall_test, step)

                    curr_loss = 0.
                    curr_retrieval_loss = 0.

            if opt.is_main and (step - 1) % opt.save_freq == 0 and step > opt.start_eval_step:
                util.save(generator_model, generator_optimizer, generator_scheduler, step, best_dev_score,
                          opt, checkpoint_path, f"generator_step-{step}")
                util.save(retriever_model, retriever_optimizer, retriever_scheduler, step, best_dev_score,
                          opt, checkpoint_path, f"retriever_step-{step}")
                if opt.use_gt_dbs is False or (opt.use_gt_dbs is True and opt.use_retriever_for_gt is True):
                    np.save(checkpoint_path / "checkpoint" / "retriever_best_dev" /
                            "retriever_all_dbs_embeddings.npy", retriever_all_dbs_embeddings.numpy())

            if step == opt.joint_generator_retriever_start_step:  # just for reuse
                util.save(generator_model, generator_optimizer, generator_scheduler, step, best_dev_score,
                          opt, checkpoint_path, f"generator_step-{step}")
                util.save(retriever_model, retriever_optimizer, retriever_scheduler, step, best_dev_score,
                          opt, checkpoint_path, f"retriever_step-{step}")

            if step > training_steps:
                break


def evaluate(generator_model, eval_dial_dataset, dial_collator, generator_tokenizer, opt,
             generator_all_dbs_ids, generator_all_dbs_mask, retriever_model, retriever_all_dbs_embeddings, step):
    sampler = SequentialSampler(eval_dial_dataset)
    eval_dial_dataloader = DataLoader(eval_dial_dataset,
                                      sampler=sampler,
                                      batch_size=opt.per_gpu_eval_batch_size,
                                      drop_last=False,
                                      num_workers=10,
                                      collate_fn=dial_collator
                                      )
    generator_model.eval()
    retriever_model.eval()
    results = []
    raw_data = []
    retrieve_results = []
    generator_model = generator_model.module if hasattr(generator_model, "module") else generator_model
    retriever_model = retriever_model.module if hasattr(retriever_model, "module") else retriever_model

    with torch.no_grad():
        for i, batch in enumerate(tqdm(eval_dial_dataloader)):
            (index, resp_ori_input_ids, resp_ori_mask,\
            generator_context_input_ids, generator_context_mask, \
            retriever_context_input_ids, retriever_context_mask, retriever_context_token_type, \
            resp_delex_mask, gt_db_idx, gold_entities) = batch

            if opt.use_gt_dbs is False:
                retriever_context_embeddings = retriever_model(input_ids=retriever_context_input_ids.long().cuda(),
                                                               attention_mask=retriever_context_mask.long().cuda(),
                                                               token_type_ids=retriever_context_token_type.long().cuda(),
                                                               output_hidden_states=True,
                                                               return_dict=True,
                                                               sent_emb=True).pooler_output  # have grad
                retriever_all_dbs_scores = torch.einsum("bd,nd->bn", retriever_context_embeddings.detach().cpu(),
                                                        retriever_all_dbs_embeddings)  # (bs, all_db_num)
                retriever_top_k_dbs_index = retriever_all_dbs_scores.sort(-1, True)[1][:, :opt.top_k_dbs].unsqueeze(2)
                retrieve_results.append(retriever_all_dbs_scores.detach().cpu())
            else:
                if opt.use_retriever_for_gt:
                    top_k_dbs = opt.top_k_dbs
                    retriever_context_embeddings = retriever_model(input_ids=retriever_context_input_ids.long().cuda(),
                                                                   attention_mask=retriever_context_mask.long().cuda(),
                                                                   token_type_ids=retriever_context_token_type.long().cuda(),
                                                                   output_hidden_states=True,
                                                                   return_dict=True,
                                                                   sent_emb=True).pooler_output  # have grad
                    retriever_all_dbs_scores = torch.einsum("bd,nd->bn", retriever_context_embeddings.detach().cpu(),
                                                            retriever_all_dbs_embeddings)  # (bs, all_db_num)
                    retriever_gt_dbs_scores = torch.gather(retriever_all_dbs_scores, 1,
                                                           gt_db_idx.long())  # (bs, gt_db_num)

                    retriever_top_k_dbs_index = retriever_gt_dbs_scores.sort(-1, True)[1][:, :top_k_dbs]  # (bs, top_k)
                    retriever_top_k_dbs_index = torch.gather(gt_db_idx, 1, retriever_top_k_dbs_index.long())
                    retrieve_results.append(retriever_top_k_dbs_index.detach().cpu())
                    retriever_top_k_dbs_index = retriever_top_k_dbs_index.unsqueeze(2)  # (bs, top_k, 1)
                else:
                    retriever_top_k_dbs_index = gt_db_idx.unsqueeze(2)  # (bs, n_entity, 1)

            # get top-k db generator inputs and concat with context inputs and forward into generator model
            bsz = retriever_top_k_dbs_index.size(0)
            generator_db_len = generator_all_dbs_ids.size(-1)
            generator_top_k_dbs_ids = torch.gather(generator_all_dbs_ids.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                   retriever_top_k_dbs_index.long().repeat(1, 1, generator_db_len))
            generator_top_k_dbs_mask = torch.gather(generator_all_dbs_mask.unsqueeze(0).repeat(bsz, 1, 1), 1,
                                                    retriever_top_k_dbs_index.long().repeat(1, 1, generator_db_len))

            generator_context_top_k_dbs_input_ids = concat_context_and_dbs_input(generator_context_input_ids,
                                                                                 generator_top_k_dbs_ids)
            generator_context_top_k_dbs_mask = concat_context_and_dbs_input(generator_context_mask,
                                                                            generator_top_k_dbs_mask)

            retriever_top_k_dbs_scores = torch.gather(retriever_all_dbs_scores, 1,
                                                      retriever_top_k_dbs_index.long().squeeze(2))  # (bs, top_k)

            ans_outputs, res_outputs, node_prob = \
                generator_model.inference(input_ids=generator_context_top_k_dbs_input_ids.long().cuda(),
                                          attention_mask=generator_context_top_k_dbs_mask.cuda(),
                                          response_max_length=opt.response_maxlength,
                                          return_dict=True,
                                          entity_score=retriever_top_k_dbs_scores.cuda() if
                                          opt.joint_generator_retriever_cotrain or
                                          opt.joint_generator_retriever_cotrain_V2 else None)

            batch_size = generator_context_top_k_dbs_input_ids.size(0)
            for k in range(batch_size):
                result = []
                res = ''
                if res_outputs is not None:
                    res = generator_tokenizer.decode(res_outputs[k], skip_special_tokens=True).lower()
                    res = res.replace('<res> ', '')
                result.append(res)

                example = eval_dial_dataset.get_example(index[k])
                raw_data.append(example)
                if opt.dataset_name == "mwoz_gptke":
                    result.append(example["output_used"])
                    result.append(example["gold_entities"])
                    if "data1" in opt.eval_data:
                        result.append(example["kb"])
                    result.append(example["type"])

                elif opt.dataset_name == "camrest":
                    result.append(example["output_used"])
                    result.append(example["gold_entities"])

                elif opt.dataset_name == "smd":
                    result.append(example["output_used"])
                    result.append(example["gold_entities"])

                else:
                    raise NotImplementedError
                results.append(result)

    retrieve_results = torch.cat(retrieve_results, dim=0)
    if opt.is_distributed:
        output = [None for _ in range(opt.world_size)]
        dist.all_gather_object(output, results)
        new_results = []
        for r in output:
            new_results += r
        results = new_results

    if opt.dataset_name == "mwoz_gptke" and "data1" in opt.eval_data:
        METRIC = evaluation.Metric_data1_new1(results)
    elif opt.dataset_name == "camrest" and "data0" in opt.eval_data:
        METRIC = evaluation.Metric_data0_new1(results)
    elif opt.dataset_name == "smd" and "data0" in opt.eval_data:
        METRIC = evaluation.Metric_data0_new1(results)
    else:
        raise NotImplementedError
    reader_metrics = METRIC.baseline_reader_metric()
    model_select_metric = reader_metrics[opt.model_select_metric]

    if opt.dataset_name != "smd":
        RETRIEVE_METRIC = evaluation.Retrieve_Metric(retrieve_results, data=raw_data, db=data_loader.load_dbs(opt.dbs),
                                                     GTKB=opt.use_retriever_for_gt)

        for i in range(3, opt.eval_top_k_dbs+1):
            retrieve_metrics = RETRIEVE_METRIC.calc_recall(level="turn_level", top_k=i, first_turn_name=True)
            for k, v in retrieve_metrics.items():
                v, _ = util.weighted_average(v, len(raw_data), opt)
                retrieve_metrics[k] = v
            reader_metrics.update(retrieve_metrics)

    return model_select_metric, reader_metrics


def run(opt, checkpoint_path):
    if opt.dataset_name == "mwoz_gptke":
        special_tokens = ["<user>", "<sys>", "<api>", "<sys-api>", "<database>", "<sep_attributes>", "<ans>", "<res>"]
    elif opt.dataset_name == "camrest":
        special_tokens = ["<user>", "<sys>", "<database>", "<sep_attributes>"]
    elif opt.dataset_name == "smd":
        special_tokens = ["<user>", "<sys>", "<database>", "<sep_attributes>"]
    else:
        raise NotImplementedError
    # generator model
    generator_model_name = opt.model_size
    generator_model_class = model.FiDT5
    generator_tokenizer = transformers.T5Tokenizer.from_pretrained(generator_model_name)
    _ = generator_tokenizer.add_tokens(special_tokens)
    opt.decoder_start_ans_token_id = generator_tokenizer.encode("<ans>")[0]
    opt.decoder_start_res_token_id = generator_tokenizer.encode("<res>")[0]

    retriever_model_name = opt.retriever_model_name
    retriever_model_class = simcse_model.BertForCL
    retriever_tokenizer = transformers.BertTokenizer.from_pretrained(retriever_model_name)

    if opt.model_path:
        generator_model_path = f"{opt.model_path}/generator_step-{opt.joint_generator_retriever_start_step}"
        retriever_model_path = f"{opt.model_path}/retriever_step-{opt.joint_generator_retriever_start_step}"
        generator_model, generator_optimizer, generator_scheduler, opt_checkpoint, step, best_dev_score = \
            util.load(generator_model_class, generator_model_path, opt)
        retriever_model, retriever_optimizer, retriever_scheduler, opt_checkpoint, _, _ = \
            util.load(retriever_model_class, retriever_model_path, opt)

    else:
        # reader
        t5 = transformers.T5ForConditionalGeneration.from_pretrained(generator_model_name)
        t5.resize_token_embeddings(len(generator_tokenizer))  # len vocabulary is not absolutely correct.
        generator_model = generator_model_class(t5.config, model_args=opt)
        generator_model.load_t5(t5)
        generator_model = generator_model.to(opt.local_rank)
        generator_model.set_checkpoint(opt.use_checkpoint)
        generator_optimizer, generator_scheduler = util.set_optim(opt, generator_model)
        step, best_dev_score = 0, 0.0

        # retriever
        retriever_model = retriever_model_class.from_pretrained(retriever_model_name)  # dont need to add token
        retriever_model = retriever_model.to(opt.local_rank)
        retriever_optimizer, retriever_scheduler = util.set_retriever_optim(opt, retriever_model)

    if opt.is_distributed:
        generator_model = torch.nn.parallel.DistributedDataParallel(
            generator_model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )

        retriever_model = torch.nn.parallel.DistributedDataParallel(
            retriever_model,
            device_ids=[opt.local_rank],
            output_device=opt.local_rank,
            find_unused_parameters=False,
        )
    entity_pad_id = len(json.load(open(opt.dbs, 'r'))) - 1

    attributes = json.load(open(opt.entities, 'r'))
    attribute_value_dict = {}
    attribute_values = []
    for attr_name in attributes:
        for attr_val in attributes[attr_name]:
            if isinstance(attr_val, dict):
                for _attr_nm, _attr_val in attr_val.items():
                    if _attr_val not in attribute_value_dict:
                        attribute_values.append(_attr_val)
                        attribute_value_dict[_attr_val] = len(attribute_value_dict)
            else:
                if attr_val not in attribute_value_dict:
                    attribute_values.append(attr_val)
                    attribute_value_dict[attr_val] = len(attribute_value_dict)
    attribute_values = generator_tokenizer.batch_encode_plus(attribute_values, max_length=25,
                                                             return_tensors='pt', truncation=True, padding='max_length')

    attribute_values_id = attribute_values["input_ids"]
    attribute_values_mask = attribute_values["attention_mask"].bool()
    attribute_values = attribute_values_id.masked_fill(~attribute_values_mask, -100)

    # use global rank and world size to split the train set on multiple gpus
    train_dial_examples = data_loader.load_data(
        opt.train_data,
        local_rank=opt.local_rank,
        world_size=opt.world_size,
    )
    train_dial_dataset = data_loader.DialDataset(train_dial_examples, attribute_value_dict,
                                                     use_delex=opt.use_delex,
                                                     use_gt_dbs=opt.use_gt_dbs,
                                                     use_entity_pad=opt.use_entity_pad,
                                                     entity_pad_id=entity_pad_id)
    # use global rank and world size to split the eval set on multiple gpus
    eval_dial_examples = data_loader.load_data(
        opt.eval_data,
        local_rank=opt.local_rank,
        world_size=opt.world_size,
    )
    eval_dial_dataset = data_loader.DialDataset(eval_dial_examples, attribute_value_dict, use_gt_dbs=opt.use_gt_dbs,
                                                use_entity_pad=opt.use_entity_pad,
                                                entity_pad_id=entity_pad_id
                                                )
    # use global rank and world size to split the eval set on multiple gpus
    test_dial_examples = data_loader.load_data(
        opt.test_data,
        local_rank=opt.local_rank,
        world_size=opt.world_size,
    )
    test_dial_dataset = data_loader.DialDataset(test_dial_examples, attribute_value_dict, use_gt_dbs=opt.use_gt_dbs,
                                                use_entity_pad=opt.use_entity_pad,
                                                entity_pad_id=entity_pad_id
                                                )
    dial_collator = data_loader.DialCollator(generator_tokenizer, retriever_tokenizer, opt.generator_text_maxlength,
                                             opt.retriever_text_maxlength,
                                             opt.answer_maxlength, opt.response_maxlength)

    db_examples = data_loader.load_dbs(opt.dbs)
    db_dataset = data_loader.DBDataset(db_examples, opt.db_type, use_dk=opt.use_dk,
                                           dk_mask=opt.dk_mask, use_entity_pad=opt.use_entity_pad)
    generator_db_collator = data_loader.DBCollator(generator_tokenizer, opt.generator_db_maxlength,
                                                       opt.generator_text_maxlength,
                                                       attribute_type_num=opt.attribute_type_num,
                                                       type="generator")
    retriever_db_collator = data_loader.DBCollator(retriever_tokenizer, opt.retriever_db_maxlength,
                                                   opt.retriever_text_maxlength,
                                                   attribute_type_num=opt.attribute_type_num,
                                                   type="retriever")

    if opt.is_main:
        logger.warning("Start training")

    train(
        generator_model, generator_tokenizer, generator_optimizer, generator_scheduler, generator_db_collator,
        retriever_model, retriever_tokenizer, retriever_optimizer, retriever_scheduler, retriever_db_collator,
        step,
        train_dial_dataset,
        eval_dial_dataset,
        test_dial_dataset,
        dial_collator,
        db_dataset,
        opt,
        best_dev_score,
        checkpoint_path,
        attribute_values,
    )


if __name__ == "__main__":
    options = Options()
    options.add_reader_options()
    options.add_optim_options()
    options.add_eval_options()
    opt = options.parse()

    torch.manual_seed(opt.seed)

    add_exp_name = f"_seed-{opt.seed}_gtml-{opt.generator_text_maxlength}_gdml-{opt.generator_db_maxlength}" \
                   f"_rtml-{opt.retriever_text_maxlength}_rdml-{opt.retriever_db_maxlength}_topk-{opt.top_k_dbs}" \
                   f"_resml-{opt.response_maxlength}_dus-{opt.db_emb_update_steps}"

    if opt.end_eval_step != 32000:
        add_exp_name += f"_es-{opt.end_eval_step}"
    if opt.use_gt_dbs is True:
        add_exp_name += "_gtdb"
    if opt.model_select_metric != "MICRO-F1":
        add_exp_name += f"_msm-{opt.model_select_metric}"
    opt.name += add_exp_name

    opt.scheduler_steps = opt.total_steps // opt.accumulation_steps
    opt.warmup_steps = opt.warmup_steps // opt.accumulation_steps

    opt.retriever_total_steps = opt.total_steps - opt.joint_generator_retriever_start_step
    opt.retriever_scheduler_steps = opt.retriever_total_steps // opt.retriever_accumulation_steps
    opt.retriever_warmup_steps = opt.retriever_warmup_steps // opt.retriever_accumulation_steps \
        if opt.retriever_warmup_steps > 0 else 0

    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    checkpoint_exists = checkpoint_path.exists()

    if opt.is_distributed:
        opt.local_rank = int(os.environ['LOCAL_RANK'])
        if opt.local_rank == 0:
            opt.is_main = True
        else:
            opt.is_main = False
        opt.world_size = torch.cuda.device_count()
        torch.distributed.init_process_group(
            init_method='env://',
            backend='nccl',
            world_size=opt.world_size,
            rank=opt.local_rank
        )
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.barrier()
    else:
        opt.is_main = True
        opt.local_rank = 0
        opt.world_size = 1

    checkpoint_path.mkdir(parents=True, exist_ok=True)
    logger = util.init_logger(
        opt.is_main,
        opt.is_distributed,
        checkpoint_path / 'run.log'
    )
    if not checkpoint_exists and opt.is_main:
        options.print_options(opt)

    if opt.dataset_name == "mwoz_gptke":
        from data.evaluation.woz import evaluation
    elif opt.dataset_name == "camrest":
        from data.evaluation.camrest import evaluation
    elif opt.dataset_name == "smd":
        from data.evaluation.smd import evaluation
    else:
        raise NotImplementedError
    run(opt, checkpoint_path)
