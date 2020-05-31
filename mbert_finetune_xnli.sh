#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0;
export XNLI_DIR=./finetune_data/XNLI/;


max_sequence_length=128
batch_size=32

for learning_rate in 5e-5 3e-5; do
    for num_training_epochs in 2.0 3.0; do
        train_language=en
        test_language=de

        ### training part
        python run_xnli.py \
          --model_type bert \
          --model_name_or_path bert-base-multilingual-cased \
          --language ${test_language} \
          --train_language ${train_language} \
          --do_train \
          --do_eval \
          --data_dir $XNLI_DIR \
          --per_gpu_train_batch_size ${batch_size} \
          --learning_rate ${learning_rate} \
          --num_train_epochs ${num_training_epochs} \
          --max_seq_length ${max_sequence_length} \
          --output_dir ./data/xnli_${train_language}_${test_language}_${learning_rate}_${num_training_epochs} \
          --save_steps -1

        model_path=./data/xnli_${train_language}_${test_language}_${learning_rate}_${num_training_epochs}

        for test_language2 in "fr" "es" "el" "bg" "ru" "tr" "ar" "vi" "th" "zh" "hi" "sw" "ur" "de"; do
        ### evaluation part
            python run_xnli.py \
              --model_type bert \
              --model_name_or_path ${model_path} \
              --language ${test_language2} \
              --train_language ${train_language} \
              --do_eval \
              --data_dir $XNLI_DIR \
              --per_gpu_train_batch_size ${batch_size} \
              --learning_rate ${learning_rate} \
              --num_train_epochs ${num_training_epochs} \
              --max_seq_length ${max_sequence_length} \
              --output_dir ./data/eval_xnli_${train_language}_${test_language2}_${learning_rate}_${num_training_epochs} \
              --save_steps -1
        done;
    done;
done;