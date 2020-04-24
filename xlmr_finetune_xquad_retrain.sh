#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=1;
export SQUAD_DIR=/work/anlausch/DebunkMLBERT/finetune_data/xquad/;


max_sequence_length=384
batch_size=11
# best model on english is 3e-5 2.0
model_path=./data/xquad_xlmr_english_2e-5_2.0

## do not feed any examples and just predict on target languages
for learning_rate in 2e-5; do
    for num_training_epochs in 1.0; do
        for train_language in  "en" "zh" "vi" "tr" "th" "ru" "hi" "es" "el" "de" "ar"; do

            ### evaluation part
            python run_xquad_xlmr.py \
                --model_type xlm-roberta \
                --model_name_or_path ${model_path} \
                --do_eval \
                --predict_file $SQUAD_DIR/xquad-test.${train_language}.json \
                --per_gpu_train_batch_size ${batch_size} \
                --learning_rate ${learning_rate} \
                --num_train_epochs ${num_training_epochs} \
                --max_seq_length ${max_sequence_length} \
                --doc_stride 128 \
                --seed 1 \
                --overwrite_cache \
                --output_dir ./data/xquad_eval_xlmr_retrain_0_${train_language}_${learning_rate}_${num_training_epochs}_1 \
                --save_steps -1
        done;
    done;
done;

for iteration in 1 2 3 4 5; do
    for learning_rate in 2e-5; do
        for num_training_epochs in 1.0; do
            for num_articles in 2 4 6 8 10; do
                for train_language in  "en" "zh" "vi" "tr" "th" "ru" "hi" "es" "el" "de" "ar"; do
                    ### evaluation part
                    python run_xquad_xlmr.py \
                        --model_type xlm-roberta \
                        --model_name_or_path ${model_path} \
                        --do_train \
                        --do_eval \
                        --train_file $SQUAD_DIR/xquad-train.${train_language}.json \
                        --predict_file $SQUAD_DIR/xquad-test.${train_language}.json \
                        --per_gpu_train_batch_size ${batch_size} \
                        --learning_rate ${learning_rate} \
                        --num_train_epochs ${num_training_epochs} \
                        --max_seq_length ${max_sequence_length} \
                        --doc_stride 128 \
                        --seed ${iteration} \
                        --overwrite_cache \
                        --num_articles ${num_articles} \
                        --output_dir ./data/xquad_eval_xlmr_retrain_${num_articles}_${train_language}_${learning_rate}_${num_training_epochs}_${iteration} \
                        --save_steps -1
                done;
            done;
        done;
    done;
done;