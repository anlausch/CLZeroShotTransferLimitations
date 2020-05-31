#!/usr/bin/env bash
export XNLI_DIR=/work/anlausch/DebunkMLBERT/finetune_data/xquad/;
eval_script="${XNLI_DIR}evaluate-v1.1.py"
data_set="${XNLI_DIR}dev-v1.1.json"

for learning_rate in 3e-5 2e-5; do
    for num_training_epochs in 2.0 3.0; do
        prediction_path="./data/xquad_english_${learning_rate}_${num_training_epochs}/predictions_.json"
        python $eval_script ${data_set} ${prediction_path}
    done;
done;