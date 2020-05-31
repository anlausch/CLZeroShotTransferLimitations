#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1;
export SQUAD_DIR=./finetune_data/xquad/;


max_sequence_length=384
batch_size=11

python run_xquad_xlmr.py \
  --model_type xlmr \
  --model_name_or_path ./data/xquad_xlmr_english_3e-5_3.0 \
  --do_eval \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size ${batch_size} \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --max_seq_length ${max_sequence_length} \
  --num_articles -1 \
  --language en \
  --doc_stride 128 \
  --output_dir ./data/xquad_xlmr_english_3e-5_3.0
