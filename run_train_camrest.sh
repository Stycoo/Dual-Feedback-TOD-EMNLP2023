#!/bin/bash
export CUDA_VISIBLE_DEVICES=7
ES=48000
DATA=RRG_cdnet_data0_times_gtdb_gesa_times-cr
RMN=./retriever_pretrained/retriever_camrest_cdnet_data0_times_gtdb_gesa_times-cr_retrieve1_seed-111_ep-15_lr-5e-5_wd-0.01_maxlen-128_bs-108_ngpu-_pln-128_tmp-0.05_hnw-0

python train.py \
    --name  \
    --train_data /data/${DATA}/train.json \
    --eval_data /data/${DATA}/val.json \
    --test_data /data/${DATA}/test.json \
    --entities /data/${DATA}/entities.json\
    --dbs /data/${DATA}/all_db.json \
    --checkpoint_dir /data/checkpoints/camrest \
    --lr 5e-5 \
    --model_size large \
    --dataset_name camrest \
    --retriever_model_name ${RMN} \
    --total_steps ${ES} \
    --eval_freq 2000 \
    --retriever_warmup_steps 0 \
    --retriever_lr 2e-5 \
    --end_eval_step ${ES} \
    --per_gpu_batch_size 1 \
    --top_k_dbs 4 \
    --eval_top_k_dbs 4 \
    --top_k_dbs_retGT 4 \
    --use_delex \
    --use_checkpoint \
    --joint_generator_retriever_cotrain \
    --joint_generator_retriever_cotrain_PNFB \
    --joint_generator_ranker_target_grain seq \
    --joint_generator_retriever_cotrain_kd_type cross_attn \
    --joint_generator_retriever_start_step 2 \
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