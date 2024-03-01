import string

import numpy as np
import os
import errno
import torch
import sys
import logging
import json
from pathlib import Path
import torch.distributed as dist
import csv
from nltk.tokenize import word_tokenize as tknz

logger = logging.getLogger(__name__)


def init_logger(is_main=True, is_distributed=False, filename=None):
    if is_distributed:
        torch.distributed.barrier()
    handlers = [logging.StreamHandler(sys.stdout)]
    if filename is not None:
        handlers.append(logging.FileHandler(filename=filename))
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)
    logging.getLogger('transformers.tokenization_utils_base').setLevel(logging.ERROR)
    return logger


def get_checkpoint_path(opt):
    checkpoint_path = Path(opt.checkpoint_dir) / opt.name
    checkpoint_exists = checkpoint_path.exists()
    if opt.is_distributed:
        torch.distributed.barrier()
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path, checkpoint_exists


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def save(model, optimizer, scheduler, step, best_eval_metric, opt, dir_path, name):
    model_to_save = model.module if hasattr(model, "module") else model
    path = os.path.join(dir_path, "checkpoint")
    epoch_path = os.path.join(path, name)  # "step-%s" % step)
    os.makedirs(epoch_path, exist_ok=True)
    model_to_save.save_pretrained(epoch_path)
    cp = os.path.join(path, "latest")
    fp = os.path.join(epoch_path, "optimizer.pth.tar")
    checkpoint = {
        "step": step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "opt": opt,
        "best_eval_metric": best_eval_metric,
    }
    torch.save(checkpoint, fp)
    symlink_force(epoch_path, cp)


def load(model_class, dir_path, opt, reset_params=False):
    epoch_path = os.path.realpath(dir_path)
    optimizer_path = os.path.join(epoch_path, "optimizer.pth.tar")
    logger.info("Loading %s" % epoch_path)
    model = model_class.from_pretrained(epoch_path, opt)
    model = model.to(opt.local_rank)
    logger.info("loading checkpoint %s" % optimizer_path)
    checkpoint = torch.load(optimizer_path, map_location=model.device)
    opt_checkpoint = checkpoint["opt"]
    step = checkpoint["step"]
    if "best_eval_metric" in checkpoint:
        best_eval_metric = checkpoint["best_eval_metric"]
    else:
        best_eval_metric = checkpoint["best_dev_em"]
    if not reset_params:
        optimizer, scheduler = set_optim(opt_checkpoint, model)
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        optimizer, scheduler = set_optim(opt, model)

    return model, optimizer, scheduler, opt_checkpoint, step, best_eval_metric


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, min_ratio, fixed_lr, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        self.min_ratio = min_ratio
        self.fixed_lr = fixed_lr
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return (1 - self.min_ratio) * step / float(max(1, self.warmup_steps)) + self.min_ratio

        if self.fixed_lr:
            return 1.0

        return max(0.0,
                   1.0 + (self.min_ratio - 1) * (step - self.warmup_steps) / float(
                       max(1.0, self.scheduler_steps - self.warmup_steps)),
                   )


class FixedScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, last_epoch=-1):
        super(FixedScheduler, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        return 1.0


def set_dropout(model, dropout_rate):
    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = dropout_rate


def set_optim(opt, model):
    if opt.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    if opt.scheduler == 'fixed':
        scheduler = FixedScheduler(optimizer)
    elif opt.scheduler == 'linear':
        if opt.scheduler_steps is None:
            scheduler_steps = opt.total_steps
        else:
            scheduler_steps = opt.scheduler_steps
        scheduler = WarmupLinearScheduler(optimizer, warmup_steps=opt.warmup_steps, scheduler_steps=scheduler_steps,
                                          min_ratio=0., fixed_lr=opt.fixed_lr)
    return optimizer, scheduler


def set_retriever_optim(opt, model):
    if opt.retriever_optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.retriever_lr)
    elif opt.retriever_optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.retriever_lr, weight_decay=opt.retriever_weight_decay)
    else:
        raise ValueError

    if opt.retriever_scheduler == 'fixed':
        scheduler = FixedScheduler(optimizer)
    elif opt.retriever_scheduler == 'linear':
        if opt.retriever_scheduler_steps is None:
            scheduler_steps = opt.retriever_total_steps
        else:
            scheduler_steps = opt.retriever_scheduler_steps
        scheduler = WarmupLinearScheduler(optimizer, warmup_steps=opt.retriever_warmup_steps, scheduler_steps=scheduler_steps,
                                          min_ratio=0., fixed_lr=opt.retriever_fixed_lr)
    else:
        raise ValueError
    return optimizer, scheduler


def set_ranker_optim(opt, model):
    if opt.ranker_optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.ranker_lr)
    elif opt.ranker_optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.ranker_lr, weight_decay=opt.ranker_weight_decay)
    if opt.ranker_scheduler == 'fixed':
        scheduler = FixedScheduler(optimizer)
    elif opt.ranker_scheduler == 'linear':
        if opt.ranker_scheduler_steps is None:
            scheduler_steps = opt.ranker_total_steps
        else:
            scheduler_steps = opt.ranker_scheduler_steps
        scheduler = WarmupLinearScheduler(optimizer, warmup_steps=opt.ranker_warmup_steps, scheduler_steps=scheduler_steps,
                                          min_ratio=0., fixed_lr=opt.ranker_fixed_lr)
    return optimizer, scheduler


def average_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
        if opt.is_main:
            x = x / opt.world_size
    return x


def sum_main(x, opt):
    if not opt.is_distributed:
        return x
    if opt.world_size > 1:
        dist.reduce(x, 0, op=dist.ReduceOp.SUM)
    return x


def weighted_average(x, count, opt):
    if not opt.is_distributed:
        return x, count
    t_loss = torch.tensor([x * count], device=opt.device)
    t_total = torch.tensor([count], device=opt.device)
    t_loss = sum_main(t_loss, opt)
    t_total = sum_main(t_total, opt)
    return (t_loss / t_total).item(), t_total.item()


def write_output(glob_path, output_path):
    files = list(glob_path.glob('*.txt'))
    files.sort()
    results = []
    for path in files:
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                idx, text_pred, text_gold = line.strip().split("\t")
                results.append([int(idx), text_pred, text_gold])
        path.unlink()
    results.sort(key=lambda x: x[0], reverse=False)
    writer = csv.writer(open(output_path, 'w', encoding='utf-8'), delimiter=',')
    writer.writerow(["exp_id", "response_pred", "response_gold"])
    writer.writerows(results)
    glob_path.rmdir()


def padSeqs(sequences, maxlen=None, truncated='max_len', pad_method='post', trunc_method='pre', dtype='int32',
            value=0., resOrAns=False):
    assert truncated in ['max_len', 'batch_max_len']
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    seq_maxlen = np.max(lengths)

    # if maxlen is not None and truncated:
    #     maxlen = min(seq_maxlen, maxlen)
    # else:
    #     maxlen = seq_maxlen
    if truncated == 'max_len':
        maxlen = maxlen
    else:
        maxlen = min(maxlen, seq_maxlen)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            print('empty list/array was found')
            continue  # empty list/array was found
        if trunc_method == 'pre':
            if resOrAns:
                trunc = s[0] + s[-maxlen+1:]
            else:
                trunc = s[-maxlen:]
        elif trunc_method == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % trunc_method)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if pad_method == 'post':
            x[idx, :len(trunc)] = trunc
        elif pad_method == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % pad_method)
    return x


def padLabel(labels):
    max_label_num = max([len(x) for x in labels])
    labels_padded = []
    for label in labels:
        labels_padded.append(label + [0] * (max_label_num - len(label)))
    return labels_padded


def get_pseudo_entity_label(adj_matrix, dbs_id, entityID_attributeID_map, gold_node_label):
    '''
    :param adj_matrix: n_attribute_value, n_attribute_value
    :param dbs_id: bsz, n_entity
    :param entityID_attributeID_map: full_db_size
    :param gold_node_label: bsz, n_gold_entity
    :return:
    '''
    batch_size, n_entity = dbs_id.size()
    n_attribute_value = adj_matrix.size(0)

    entity_name_id = torch.gather(entityID_attributeID_map.unsqueeze(0).expand(batch_size, -1), 1, dbs_id)
    adj_matrix_used = torch.gather(adj_matrix.unsqueeze(0).expand(batch_size, -1, -1), 1,
                                   entity_name_id.unsqueeze(-1).expand(-1, -1, n_attribute_value))

    # bsz, n_entity, n_gold_entity
    adj_matrix_used = torch.gather(adj_matrix_used, -1, gold_node_label.unsqueeze(1).expand(-1, n_entity, -1))
    pseudo_entity_label = torch.sum(adj_matrix_used, -1).gt(0).int()  # bsz, n_entity
    return pseudo_entity_label


def retrieval_precision(pseudo_entity_label):
    # pseudo_entity_label: bsz, n_entity
    retri_precision = torch.mean(pseudo_entity_label.float())
    return retri_precision


def clean_text_output(text, use_tknz=True, dataset="mwoz"):
    """
    clean special tokens and make space for every token
    :param text:
    :param use_tknz:
    :return:
    """
    if dataset == "mwoz_gptke":
        text = text.replace("<sys> ", "").replace("<sys-api> ", "").replace(" </s>", ""). \
            replace("<user>", "").replace("<api>", "").replace("<database>", "").replace("<sep_attributes>", "").strip()
    if use_tknz is True:
        text = ' '.join(tknz(text))
    return text


def preprocess_text(text):
    """Preprocess utterance and table value."""
    text = text.strip().replace("\t", " ").lower()
    for p in string.punctuation:
        text = text.replace(p, f" {p} ")
    text = " ".join(text.split())
    return text


def clean_gen_sentence(text):
    text = text.replace("<sys> ", "").replace("<sys-api> ", "").replace(" </s>", ""). \
        replace("<user>", "").replace("<api>", "").replace("<database>", "").replace("<sep_attributes>", "").strip()
    return text


def kg_node_init(nodes_span, lm_hidden_states, n_nodes):
    bsz = nodes_span.size(0)
    text_length = lm_hidden_states.size(1)

    # attribute value pooling mask
    pooling_mask = nodes_span.new_zeros(size=(bsz, n_nodes, text_length))
    for i, node_span_trip in enumerate(nodes_span):
        for j, triplet_span in enumerate(node_span_trip):
            pooling_mask[i][triplet_span[0]][triplet_span[1]: triplet_span[2]] = 1
    node_mask = pooling_mask.sum(-1).gt(0)
    node_len = pooling_mask.sum(-1).unsqueeze(-1)
    node_len = torch.where(node_len > 0, node_len,
                           (torch.ones_like(node_len) * torch.iinfo(node_len.dtype).min).cuda())

    pooling_mask = torch.div(pooling_mask, node_len)
    node_init = torch.matmul(pooling_mask, lm_hidden_states)  # bsz, n_nodes, hidden_dim
    return node_init, node_mask


def get_entity_score(node_score, node_span):
    # node_span: bsz, n_entity, n_attr, 3; node_score: bsz, n_node
    batch_size, n_entity, n_attribute, _ = node_span.size()
    entity_attribute_ids = node_span[:, :, :, 0].squeeze(-1)  # bsz, n_entity, n_attribute
    node_score = node_score[:, None, :].expand(-1, n_entity, -1)
    entity_score = node_score.gather(dim=-1, index=entity_attribute_ids)
    entity_score_mask = entity_score.gt(0).int()
    entity_score = (entity_score * entity_score_mask).sum(-1) / entity_score_mask.sum(-1)  # bsz, n_entity
    return entity_score
