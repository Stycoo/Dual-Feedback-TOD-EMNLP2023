"""data utils for turn-level batch, this means the input batch to the models are turns which might being the different
dialog's different turn."""
import torch
import json
import util


# --------------- data utils ---------------

def load_data(data_path=None, local_rank=-1, world_size=-1, num_examples=None):
    """distributed load data for train valid test set"""
    data = json.loads(open(data_path, 'r', encoding='utf-8').read().lower())  # use lower data for all
    if num_examples is not None:
        data = data[:num_examples]
    examples = []
    for k, example in enumerate(data):
        if local_rank > -1 and not k % world_size == local_rank:
            continue
        if not 'exp_id' in example:
            example['exp_id'] = k
        examples.append(example)
    return examples


def load_dbs(db_path=None, num_examples=None):
    """load all dbs in all gpus"""
    dbs = json.loads(open(db_path, 'r', encoding='utf-8').read().lower())
    if num_examples is not None:
        dbs = dbs[:num_examples]
    return dbs


# ---------------dataset & data_collector ---------------
def encode_passages(batch_text_passages, tokenizer, max_length, pad_id):
    """encode text with each db candidates"""
    passage_ids, passage_masks = [], []
    for k, text_passages in enumerate(batch_text_passages):
        p = tokenizer.batch_encode_plus(text_passages)
        p_input_ids = torch.tensor(util.padSeqs(p['input_ids'],
                                                maxlen=max_length, truncated='max_len',
                                                pad_method='post', trunc_method='pre', dtype='int32',
                                                value=pad_id))  # (num_passage. max_len)
        p_mask = torch.tensor(util.padSeqs(p['attention_mask'],
                                           maxlen=max_length, truncated='max_len',
                                           pad_method='post', trunc_method='pre', dtype='int32',
                                           value=pad_id))  # (num_passage, max_len)
        passage_ids.append(p_input_ids.unsqueeze(0))  # (1, num_passage, max_len)
        passage_masks.append(p_mask.unsqueeze(0))  # (1, num_passage, max_len)

    passage_ids = torch.cat(passage_ids, dim=0)  # (bs, num_passage, max_len)
    passage_masks = torch.cat(passage_masks, dim=0)  # (bs, num_passage, max_len)
    return passage_ids, passage_masks.bool()


def entity_to_text_w_dk_mask(entity, dk_mask=False):
    text = "<database>"
    mask = []
    for key, val in entity.items():
        text += f" {key} {val} <sep_attributes>"
        if val == "dontknow":
            mask.append(0)
        else:
            mask.append(1)

    if dk_mask is True:
        return text, mask
    else:
        return text, None


def entity_to_text_wo_dk(entity):
    text = "<database>"
    for key, val in entity.items():
        if val != "dontknow":
            text += f" {key} {val} <sep_attributes>"
    if not text.endswith(">"):  # for unused entity case
        text += ' dontknow'
    return text


def get_attribute_span(text_ids, attributes_value, tokenizer, attribute_type_num, context_maxlength):
    attribute_span = []
    pad_item = (0, 0, 0)

    for i, attr_val in enumerate(attributes_value):
        attr_span = []
        if len(attr_val) == 1 and attr_val[0][1] == "dontknow dontknow":
            attr_span = [pad_item] * attribute_type_num

        else:
            attr_text = [_attr_val[1] for _attr_val in attr_val]
            attr_text_ids = tokenizer.batch_encode_plus(attr_text)['input_ids']
            for ind, _attr_text_ids in enumerate(attr_text_ids):
                _attr_text_ids = _attr_text_ids[:-1]
                st_ind = find_sublist(_attr_text_ids, text_ids[i].tolist())

                assert len(st_ind) == 1, print(f"there exits attribute value overlap in :{attr_val}")
                st_ind = st_ind[0] + context_maxlength
                attr_span.append((attr_val[ind][0], st_ind, st_ind+len(_attr_text_ids)))

            attr_span += [pad_item] * (attribute_type_num - len(attr_span))
        attribute_span.append(attr_span)
    return attribute_span


def find_sublist(list1, list2):
    res = []
    for i in range(len(list2) - len(list1) + 1):
        if list1 == list2[i:i + len(list1)]:
            res.append(i)
    return res


class DialDataset(torch.utils.data.Dataset):
    """use for generate encoder"""

    def __init__(self, data, attribute_value_dict, use_delex=False, use_gt_dbs=False, use_entity_pad=False,
                 entity_pad_id=None):
        self.data = data
        self.use_delex = use_delex
        self.use_gt_dbs = use_gt_dbs
        self.attribute_value_dict = attribute_value_dict
        self.use_entity_pad = use_entity_pad
        self.entity_pad_id = entity_pad_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        context = example["context_used"]  # already have <user> prefix

        gold_entities = example['gold_entities']
        resp_ori = example["output_used"]

        gold_entities_id = []
        if not gold_entities:
            gold_entities_id = [-100]  # dontknow: no entity used in current dialog response
        else:
            for x in gold_entities:
                if '_' in x:
                    x = x.replace("_", " ")
                # assert x in self.attribute_value_dict, print(f"there exists OOD attribute value: {x}")
                if x not in self.attribute_value_dict:
                    continue
                else:
                    gold_entities_id.append(self.attribute_value_dict[x])
            if not gold_entities_id:
                gold_entities_id = [-100]

        return_dict = {'index': index, 'context': context, 'resp_ori': resp_ori, 'gold_entities': gold_entities_id}

        add_args = {}
        if self.use_delex is True:
            delex_entities = [x.replace("_", " ") for x in example["gold_entities"]]
            add_args["delex_entities"] = delex_entities
        if self.use_gt_dbs is True:
            gt_dbs = example["gt_db_idx"]
            if self.use_entity_pad:
                gt_dbs += [self.entity_pad_id]
            add_args["gt_db_idx"] = gt_dbs
        return_dict.update(add_args)
        return return_dict

    def get_example(self, index):
        return self.data[index]


class DialCollator(object):
    """use for generate encoder"""

    def __init__(self, generator_tokenizer, retriever_tokenizer, generator_text_maxlength, retriever_text_maxlength,
                 answer_maxlength, response_maxlength):
        self.generator_tokenizer = generator_tokenizer
        self.retriever_tokenizer = retriever_tokenizer
        self.generator_text_maxlength = generator_text_maxlength  # dialog context max length
        self.retriever_text_maxlength = retriever_text_maxlength  # dialog context max length
        self.answer_maxlength = answer_maxlength  # answer max length for QA
        self.response_maxlength = response_maxlength  # sys response max length
        self.pad_id = 0

    def __call__(self, batch):
        assert (batch[0]['resp_ori'] is not None)
        index = torch.tensor([ex['index'] for ex in batch])  # (bs, )
        gold_entities_id = [ex['gold_entities'] for ex in batch]
        gold_entities_num = [len(gt_ent_id) for gt_ent_id in gold_entities_id]
        gold_entities = torch.zeros(size=(len(gold_entities_id), max(gold_entities_num))) - 100
        for ind, gt_ent_id in enumerate(gold_entities_id):
            gold_entities[ind][:gold_entities_num[ind]] = torch.tensor(gt_ent_id)

        # RG response
        resp_ori = [ex['resp_ori'] for ex in batch]
        resp_ori = self.generator_tokenizer.batch_encode_plus(resp_ori)
        resp_ori_input_ids = torch.tensor(util.padSeqs(resp_ori['input_ids'],
                                                       maxlen=self.response_maxlength, truncated='max_len',
                                                       pad_method='post', trunc_method='pre', dtype='int32',
                                                       value=self.pad_id))  # (bs. ans_max_len)
        resp_ori_mask = torch.tensor(util.padSeqs(resp_ori['attention_mask'],
                                                  maxlen=self.response_maxlength, truncated='max_len',
                                                  pad_method='post', trunc_method='pre', dtype='int32',
                                                  value=self.pad_id)).bool()  # (bs, ans_max_len)
        resp_ori_input_ids = resp_ori_input_ids.masked_fill(~resp_ori_mask, -100)

        # dialog context for generator
        context = [ex["context"] for ex in batch]
        generator_context = self.generator_tokenizer.batch_encode_plus(context)
        generator_context_input_ids = torch.tensor(util.padSeqs(generator_context['input_ids'],
                                                                maxlen=self.generator_text_maxlength,
                                                                truncated='max_len',
                                                                pad_method='post', trunc_method='pre', dtype='int32',
                                                                value=self.pad_id))  # (bs, context_max_len)
        generator_context_mask = torch.tensor(util.padSeqs(generator_context['attention_mask'],
                                                           maxlen=self.generator_text_maxlength, truncated='max_len',
                                                           pad_method='post', trunc_method='pre', dtype='int32',
                                                           value=self.pad_id)).bool()  # (bs, context_max_len)
        # dialog context for retriever
        retriever_context = self.retriever_tokenizer.batch_encode_plus(context)
        retriever_context_input_ids = torch.tensor(util.padSeqs(retriever_context['input_ids'],
                                                                maxlen=self.retriever_text_maxlength,
                                                                truncated='max_len',
                                                                pad_method='post', trunc_method='pre', dtype='int32',
                                                                value=self.pad_id))  # (bs, context_max_len)
        retriever_context_mask = torch.tensor(util.padSeqs(retriever_context['attention_mask'],
                                                           maxlen=self.retriever_text_maxlength, truncated='max_len',
                                                           pad_method='post', trunc_method='pre', dtype='int32',
                                                           value=self.pad_id)).bool()  # (bs, context_max_len)
        retriever_context_token_type = torch.tensor(util.padSeqs(retriever_context['token_type_ids'],
                                                                 maxlen=self.retriever_text_maxlength,
                                                                 truncated='max_len',
                                                                 pad_method='post', trunc_method='pre', dtype='int32',
                                                                 value=1))  # (bs, context_max_len)

        if "delex_entities" in batch[0]:
            resp_delex_mask = torch.zeros_like(resp_ori_mask).bool()
            delex_entities = [ex["delex_entities"] for ex in batch]
            for i in range(len(batch)):
                resp_delex_mask[i][0] = True  # <sys> / <sys-api> token
                curr_delex_entities = delex_entities[i]
                if len(curr_delex_entities) == 0:
                    continue
                tknz_delex_entities = self.generator_tokenizer.batch_encode_plus(curr_delex_entities)["input_ids"]
                for e in tknz_delex_entities:
                    for delex_index in find_sublist(e, resp_ori_input_ids[i].tolist()):
                        resp_delex_mask[i][delex_index: delex_index + len(e)] = True
        else:
            resp_delex_mask = None

        if "gt_db_idx" in batch[0]:
            gt_db_idx = torch.tensor([ex["gt_db_idx"] for ex in batch])  # (bs, num_gt_db)
        else:
            gt_db_idx = None

        return index, resp_ori_input_ids, resp_ori_mask, \
               generator_context_input_ids, generator_context_mask, \
               retriever_context_input_ids, retriever_context_mask, retriever_context_token_type, \
               resp_delex_mask, gt_db_idx, gold_entities


class DBDataset(torch.utils.data.Dataset):
    """use for database text"""

    def __init__(self, dbs, db_type="entrance", use_dk=False, dk_mask=False,
                 use_entity_pad=True):
        self.dbs = dbs
        self.db_type = db_type
        self.use_dk = use_dk
        self.dk_mask = dk_mask
        self.use_entity_pad = use_entity_pad

    def __len__(self):
        return len(self.dbs)

    def __getitem__(self, index):
        example = self.dbs[index]
        if self.db_type == "entrance":
            # values = list(set(example.values()))
            # if len(values) == 1 and values[0] == 'dontknow':
            #     if self.use_entity_pad:
            #         text, _, attributes_value = entity_to_text_w_dk_mask(example, self.dbs_attribute_value_dict)
            #         return_dict = {"db_index": index, "db_text": text, "attributes_value": attributes_value}
            #     else:
            #         return_dict = {"db_index": index, "db_text": "<database>",
            #                        "attributes_value": [(0, 'dontknow dontknow')]}
            #
            # else:
            if self.use_dk is False:
                text = entity_to_text_wo_dk(example)
                return_dict = {"db_index": index, "db_text": text}
            else:
                text, mask = entity_to_text_w_dk_mask(example, dk_mask=self.dk_mask)
                if self.dk_mask is True:
                    return_dict = {"db_index": index, "db_text": text, "db_mask": mask}
                else:
                    return_dict = {"db_index": index, "db_text": text}

        else:
            raise ValueError
        return return_dict


class DBCollator(object):
    """use for database text"""

    def __init__(self, tokenizer, maxlength, context_maxlength, attribute_type_num, type='generator'):
        self.tokenizer = tokenizer
        self.maxlength = maxlength
        self.context_maxlength = context_maxlength
        self.attribute_type_num = attribute_type_num
        self.pad_id = 0
        if len(self.tokenizer.encode("<sep_attributes>")) == 1:
            self.sep_id = self.tokenizer.encode("<sep_attributes>")[0]
            self.db_id = self.tokenizer.encode("<database>")[0]
        else:
            self.sep_id = self.tokenizer.encode("<sep_attributes>")[1]
            self.db_id = self.tokenizer.encode("<database>")[1]
        self.type = type

    def __call__(self, batch):
        index = [x["db_index"] for x in batch]
        text = [x["db_text"] for x in batch]

        text = self.tokenizer.batch_encode_plus(text)
        text_ids = torch.tensor(util.padSeqs(text['input_ids'],
                                             maxlen=self.maxlength, truncated='max_len',
                                             pad_method='post', trunc_method='pre', dtype='int32',
                                             value=self.pad_id))
        text_mask = torch.tensor(util.padSeqs(text['attention_mask'],
                                              maxlen=self.maxlength, truncated='max_len',
                                              pad_method='post', trunc_method='pre', dtype='int32',
                                              value=self.pad_id)).bool()

        if "db_mask" in batch[0] and self.type == "generator":  # dk_mask only for generator
            attribute_mask = [x["db_mask"] for x in batch]
            attribute_mask = torch.tensor(attribute_mask).bool()
            sep_index = text_ids.eq(self.sep_id).nonzero()
            db_index = text_ids.eq(self.db_id).nonzero()
            attr_num = attribute_mask.size(1)
            assert len(sep_index) % attr_num == 0, print(len(sep_index), attr_num, text_ids)
            attr_true_mask = torch.ones_like(text_mask).bool()
            for i, idx in enumerate(sep_index):
                if i % attr_num == 0:
                    start_idx = db_index[sep_index[i][0]][1]
                else:
                    start_idx = sep_index[i - 1][1] + 1
                end_idx = sep_index[i][1] + 1
                attr_true_mask[sep_index[i][0], start_idx: end_idx] = attribute_mask[sep_index[i][0], i % attr_num]
            text_mask = text_mask * attr_true_mask
        else:
            attribute_mask = None
        if "token_type_ids" in text:
            text_token_type = torch.tensor(util.padSeqs(text['token_type_ids'],
                                                        maxlen=self.maxlength, truncated='max_len',
                                                        pad_method='post', trunc_method='pre', dtype='int32',
                                                        value=1))
        else:
            text_token_type = None
        return index, text_ids, text_mask, text_token_type, attribute_mask
