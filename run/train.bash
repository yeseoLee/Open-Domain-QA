#!/bin/bash

# [참고]
# code/arguments.py
# https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/trainer#transformers.TrainingArguments

python code/train.py \
    --model_name_or_path klue/bert-base \
    --config_name None \
    --tokenizer_name None \
    \
    --dataset_name data/train_dataset \
    --max_seq_length 384 \
    --pad_to_max_length False \
    --doc_stride 128 \
    --max_answer_length 30 \
    --overwrite_cache True \
    --preprocessing_num_workers None \
    \
    --output_dir code/models/train_dataset \
    --overwrite_output_dir True \
    --do_train True \
    --do_eval True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 5e-5 \
    --weight_decay 0.0 \
    --num_train_epochs 5.0 \
    --warmup_ratio 0.0 \
    --logging_steps 500 \
    --save_steps 500 \
    --seed 42