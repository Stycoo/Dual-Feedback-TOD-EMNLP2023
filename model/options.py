import argparse
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def add_optim_options(self):
        self.parser.add_argument('--warmup_steps', type=int, default=0, help='the step meaning is same as total_step')
        self.parser.add_argument('--total_steps', type=int, default=32000)
        self.parser.add_argument('--scheduler_steps', type=int, default=None,
                                 help='the step meaning is same as total_step')
        self.parser.add_argument('--accumulation_steps', type=int, default=32)
        self.parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
        self.parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--optim', type=str, default='adamw')
        self.parser.add_argument('--scheduler', type=str, default='linear')
        self.parser.add_argument('--weight_decay', type=float, default=0.01)
        self.parser.add_argument('--fixed_lr', type=bool, default=False)

        self.parser.add_argument('--retriever_warmup_steps', type=int, default=0, help='the step meaning is same as total_step')
        self.parser.add_argument('--retriever_total_steps', type=int, default=32000)
        self.parser.add_argument('--retriever_scheduler_steps', type=int, default=None,
                                 help='the step meaning is same as total_step')
        self.parser.add_argument('--retriever_accumulation_steps', type=int, default=32)
        self.parser.add_argument('--retriever_dropout', type=float, default=0.1, help='dropout rate')
        self.parser.add_argument('--retriever_lr', type=float, default=0.0001, help='learning rate')
        self.parser.add_argument('--retriever_clip', type=float, default=1., help='gradient clipping')
        self.parser.add_argument('--retriever_optim', type=str, default='adamw')
        self.parser.add_argument('--retriever_scheduler', type=str, default='linear')
        self.parser.add_argument('--retriever_weight_decay', type=float, default=0.01)

    def add_eval_options(self):
        self.parser.add_argument('--test_data', type=str, default=None, help='path of test data')
        self.parser.add_argument('--model_path', type=str, default=None, help='path of test model')
        self.parser.add_argument('--num_beams', type=int, default=1, help='num beams')
        self.parser.add_argument('--repetition_penalty', type=float, default=1.0, help='repetition penalty')
        self.parser.add_argument('--write_generate_result', action='store_true', help='write generate result')
        self.parser.add_argument('--generate_result_path', type=str, default='none', help='generate result path')

    def add_reader_options(self):
        self.parser.add_argument('--train_data', type=str, default=None, help='path of train data')
        self.parser.add_argument('--eval_data', type=str, default=None, help='path of eval data')
        self.parser.add_argument('--entities', type=str, default=None)
        self.parser.add_argument('--dbs', type=str, default=None, help='path of dbs')

        self.parser.add_argument('--model_size', type=str, default='base')
        self.parser.add_argument('--use_checkpoint', action='store_true', help='use checkpoint in the t5 encoder')
        self.parser.add_argument('--generator_text_maxlength', type=int, default=200,
                                 help='maximum number of tokens in context')
        self.parser.add_argument('--retriever_text_maxlength', type=int, default=128,
                                 help='maximum number of tokens in context')
        self.parser.add_argument('--answer_maxlength', type=int, default=30,
                                 help='maximum number of tokens used of answer to train the model, no truncation if -1')
        self.parser.add_argument('--response_maxlength', type=int, default=64,
                                 help='maximum number of tokens used of answer to train the model, no truncation if -1')
        self.parser.add_argument('--generator_db_maxlength', type=int, default=100,
                                 help='maximum number of tokens in db text')
        self.parser.add_argument('--retriever_db_maxlength', type=int, default=128,
                                 help='maximum number of tokens in db text')
        self.parser.add_argument('--db_type', type=str, default="entrance", help='db type could be triplet or entrance')
        self.parser.add_argument('--db_emb_update_steps', type=int, default=100, help='step to update db text embedding')
        self.parser.add_argument('--top_k_dbs', type=int, default=7, help='top k db num')
        self.parser.add_argument('--eval_top_k_dbs', type=int, default=7, help='top k db num')
        self.parser.add_argument('--top_k_dbs_retGT', type=int, default=7, help='top k db num')

        self.parser.add_argument('--beam_size', type=int, default=10)
        self.parser.add_argument('--diversity_pen', type=float, default=1.0)
        self.parser.add_argument('--decoder_start_ans_token_id', type=int, default=0)
        self.parser.add_argument('--decoder_start_res_token_id', type=int, default=0)
        self.parser.add_argument('--attribute_type_num', type=int, default=12)
        self.parser.add_argument('--model_d', type=int, default=0)
        self.parser.add_argument('--use_entity_pad', action="store_true")
        self.parser.add_argument('--generator_warmup_steps', type=int, default=0)
        self.parser.add_argument('--joint_generator_ranker_target_grain', type=str, default=None, choices=['seq', 'entity_token'])
        self.parser.add_argument('--joint_generator_retriever_cotrain', action="store_true")
        self.parser.add_argument('--joint_generator_retriever_cotrain_kd_type', type=str, choices=['ent_margin', 'cross_attn'])
        self.parser.add_argument('--joint_generator_retriever_start_step', type=int, default=0)
        self.parser.add_argument('--joint_generator_retriever_end_step', type=int, default=0)
        self.parser.add_argument('--joint_generator_retriever_cotrain_PNFB', action="store_true")
        self.parser.add_argument('--hard_negative_mining_method', type=str, default=None,
                                 choices=['argmin_factEval', 'rankVariance_GP_FE', 'rankVariance_FEs'],
                                 help='GP: conditional generation prob, FE: factual evaluation')
        self.parser.add_argument('--joint_generator_retriever_cotrain_PNFB_loss_type', type=str,
                                 default=['ranking_loss', 'triplet_loss'])
        self.parser.add_argument('--joint_generator_retriever_cotrain_cand_eval_metric', type=str, default=['bleu', 'f1',
                                                                                                            'bleu_f1'])
        self.parser.add_argument('--joint_generator_retriever_eval_ngram', type=int, default=2)
        self.parser.add_argument('--joint_generator_retriever_eval_neg_threshold', type=float, default=0.5)
        self.parser.add_argument('--joint_generator_retriever_eval_pos_threshold', type=float, default=0.8)
        self.parser.add_argument('--joint_generator_ranker_consist_constrain', action="store_true")
        self.parser.add_argument('--joint_generator_ranker_consist_constrain_type', type=str, default=['kl', 'cosin'])
        self.parser.add_argument('--joint_generator_retriever_cotrain_V2', action="store_true")
        self.parser.add_argument('--joint_generator_retriever_sample_from_batch', action="store_true")
        self.parser.add_argument('--joint_generator_ranker_entRelevant_cal_type', type=str, default=['ml', 'cosin'])
        self.parser.add_argument('--low_topK_when_eval', action="store_true")

        self.parser.add_argument('--use_delex', action="store_true")
        self.parser.add_argument('--use_dk', action="store_true")
        self.parser.add_argument('--dk_mask', action="store_true")
        self.parser.add_argument('--use_gt_dbs', action="store_true")
        self.parser.add_argument('--use_retriever_for_gt', action="store_true")
        self.parser.add_argument('--generator_distill_retriever_pooling', type=str, default="avg_wo_context", help='cls/avg/avg_wo_context')


    def initialize_parser(self):
        # basic parameters
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment')
        self.parser.add_argument('--checkpoint_dir', type=str, default='/data/', help='models are saved here')
        self.parser.add_argument('--retriever_model_name', type=str, default='', help='path for retriever model')
        # dataset parameters
        self.parser.add_argument('--is_distributed', action="store_true")
        self.parser.add_argument("--per_gpu_batch_size", default=2, type=int,
                                 help="Batch size per GPU/CPU for training.")
        self.parser.add_argument("--per_gpu_eval_batch_size", default=2, type=int,
                                 help="Batch size per GPU/CPU for evaluation.")
        self.parser.add_argument('--maxload', type=int, default=-1)

        self.parser.add_argument("--local_rank", type=int, default=-1,
                                 help="For distributed training: local_rank")
        self.parser.add_argument("--main_port", type=int, default=-1,
                                 help="Main port (for multi-node SLURM jobs)")
        self.parser.add_argument('--seed', type=int, default=111, help="random seed for initialization")
        # training parameters
        self.parser.add_argument('--eval_freq', type=int, default=2000,
                                 help='evaluate model every <eval_freq> steps during training')
        self.parser.add_argument('--save_freq', type=int, default=80000,
                                 help='save model every <save_freq> steps during training')
        self.parser.add_argument('--start_eval_step', type=int, default=1,
                                 help='evaluate start step during training')
        self.parser.add_argument('--end_eval_step', type=int, default=32000,
                                 help='evaluate end step during training')
        self.parser.add_argument('--metric_record_file', type=str, default="metric_record.csv",
                                 help="file to write all metric")
        self.parser.add_argument('--dataset_name', type=str, default="mwoz_gptke", help="dataset name")
        self.parser.add_argument('--metric_version', type=str, default="old", help="metric version")
        self.parser.add_argument('--model_select_metric', type=str, default="MICRO-F1", help="model select metric")

    def print_options(self, opt):
        message = '\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default_value = self.parser.get_default(k)
            if v != default_value:
                comment = f'\t(default: {default_value})'
            message += f'{str(k):>30}: {str(v):<40}{comment}\n'

        expr_dir = Path(opt.checkpoint_dir) / opt.name
        with open(expr_dir / 'opt.log', 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        logger.info(message)

    def parse(self):
        opt = self.parser.parse_args()
        return opt
