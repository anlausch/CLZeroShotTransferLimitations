#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0;
export SQUAD_DIR=./finetune_data/xquad/;


max_sequence_length=384
batch_size=12

for learning_rate in 3e-5 2e-5; do
    for num_training_epochs in 2.0 3.0; do
        python run_squad.py \
          --model_type bert \
          --model_name_or_path bert-base-multilingual-cased \
          --do_train \
          --do_eval \
          --train_file $SQUAD_DIR/train-v1.1.json \
          --predict_file $SQUAD_DIR/dev-v1.1.json \
          --per_gpu_train_batch_size ${batch_size} \
          --learning_rate ${learning_rate} \
          --num_train_epochs ${num_training_epochs} \
          --max_seq_length ${max_sequence_length} \
          --doc_stride 128 \
          --output_dir ./data/xquad_english_${learning_rate}_${num_training_epochs}
    done;
done;