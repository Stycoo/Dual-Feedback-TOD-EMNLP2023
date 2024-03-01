#!/bin/bash
export CUDA_VISIBLE_DEVICES=4
ES=48000
DATA=RRG_qtod_data0_times_gtdb_gesa_times-cr
RMN=./retriever_pretrained/retriever_camrest_cdnet_data0_times_gtdb_gesa_times-cr_retrieve1_seed-111_ep-15_lr-5e-5_wd-0.01_maxlen-128_bs-108_ngpu-_pln-128_tmp-0.05_hnw-0

python train.py \
    --name  \
    --train_data /data/${DATA}/train.json \
    --eval_data /data/${DATA}/val.json \
    --test_data /data/${DATA}/test.json \
    --entities /data/${DATA}/entities.json\
    --dbs /data/${DATA}/all_db.json \
    --checkpoint_dir /data/checkpoints/smd/ \
    --dataset_name smd \
    --retriever_model_name ${RMN} \
    --model_size base \
    --total_steps ${ES} \
    --eval_freq 2000 \
    --retriever_warmup_steps 0 \
    --lr 5e-5 \
    --retriever_lr 1e-5 \
    --response_maxlength 128 \
    --generator_db_maxlength 200 \
    --end_eval_step ${ES} \
    --metric_version new1 \
    --per_gpu_batch_size 1 \
    --top_k_dbs ${topk} \
    --eval_top_k_dbs ${topk} \
    --use_delex \
    --use_checkpoint \
    --joint_generator_retriever_cotrain \
    --joint_generator_retriever_cotrain_PNFB \
    --joint_generator_ranker_target_grain seq \
    --joint_generator_retriever_cotrain_kd_type ent_margin \
    --joint_generator_retriever_start_step 20000 \
    --joint_generator_retriever_end_step ${ES} \
    --joint_generator_retriever_cotrain_cand_eval_metric bleu \
    --hard_negative_mining_method rankVariance_GP_FE \
    --joint_generator_retriever_eval_neg_threshold 0.5 \
    --joint_generator_retriever_eval_pos_threshold 0.8 \
    --beam_size 5 \
    --joint_generator_retriever_cotrain_PNFB_loss_type triplet_loss \
    --retriever_scheduler fixed \
    --use_retriever_for_gt \
    --use_gt_dbs \