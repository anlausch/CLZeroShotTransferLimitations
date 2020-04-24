#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0;
export XNLI_DIR=/work/anlausch/DebunkMLBERT/finetune_data/XNLI/;


max_sequence_length=128
batch_size=32

for iteration in 2 3 4 5; do
    for sampling_strategy in "k_first" "k_longest" "k_shortest"; do
        for learning_rate in 3e-5; do
            for num_training_epochs in 1.0; do
                #for num_examples in 10 100 150 200; do
                for num_examples in 10 50 100 200 500 1000; do
                    train_language_original=en
                    test_language_fake=de

                    model_path=./data/xnli_${train_language_original}_${test_language_fake}_${learning_rate}_2.0

                    for train_language in  "en" "fr" "es" "el" "bg" "ru" "tr" "ar" "vi" "th" "zh" "hi" "sw" "ur" "de"; do
                            ### evaluation part
                        python run_xnli.py \
                          --model_type bert \
                          --model_name_or_path ${model_path} \
                          --language ${train_language} \
                          --train_language ${train_language} \
                          --do_eval \
                          --do_train \
                          --overwrite_cache \
                          --num_examples ${num_examples} \
                          --sampling_strategy ${sampling_strategy} \
                          --data_dir $XNLI_DIR \
                          --seed ${iteration} \
                          --per_gpu_train_batch_size ${batch_size} \
                          --learning_rate ${learning_rate} \
                          --num_train_epochs ${num_training_epochs} \
                          --max_seq_length ${max_sequence_length} \
                          --output_dir ./data/eval_xnli_retrain_${num_examples}_${sampling_strategy}_${train_language}_${learning_rate}_${num_training_epochs}_${iteration} \
                          --save_steps -1
                    done;
                done;
            done;
        done;
    done;
done;