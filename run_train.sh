#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
ES=48000
DATA=RRG_data1_times_gtdb_gesa_times-cr-dyn
RMN=./retriever_pretrained/retriever_train_new_trunc_data_used_new_v0_seed-111_bert-base-uncased_ep-10_lr-5e-5_wd-0.01_maxlen-128_bs-32_ngpu-4_pln-128_tmp-0.05_hnw-0

python train.py \
    --name \
    --train_data /data/${DATA}/train.json \
    --eval_data /data/${DATA}/val.json \
    --test_data /data/${DATA}/test.json \
    --entities /data/${DATA}/entities.json \
    --dbs /data/${DATA}/all_db.json \
    --checkpoint_dir /data/checkpoints/mwoz \
    --dataset_name mwoz_gptke \
    --model_size base \
    --retriever_model_name ${RMN} \
    --total_steps ${ES} \
    --eval_freq 2000 \
    --retriever_warmup_steps 0 \
    --retriever_lr 2e-5 \
    --end_eval_step ${ES} \
    --per_gpu_batch_size 2 \
    --top_k_dbs 10 \
    --eval_top_k_dbs 10 \
    --use_delex \
    --use_checkpoint \
    --joint_generator_retriever_cotrain \
    --joint_generator_retriever_cotrain_PNFB \
    --joint_generator_ranker_target_grain seq \
    --joint_generator_retriever_cotrain_kd_type ent_margin\
    --joint_generator_retriever_start_step 20000 \
    --joint_generator_retriever_end_step 48000 \
    --joint_generator_retriever_cotrain_cand_eval_metric bleu \
    --hard_negative_mining_method rankVariance_GP_FE \
    --joint_generator_retriever_cotrain_PNFB_loss_type triplet_loss \
    --joint_generator_retriever_eval_neg_threshold 0.5 \
    --joint_generator_retriever_eval_pos_threshold 0.8 \
    --beam_size 5 \
    --retriever_scheduler fixed \
    --generator_distill_retriever_pooling avg_wo_context \
    --save_freq 40000 \
