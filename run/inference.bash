#!/bin/bash

# [참고]
# code/arguments.py
# https://huggingface.co/docs/transformers/v4.37.2/en/main_classes/trainer#transformers.TrainingArguments

python code/inference.py \
    --model_name_or_path code/models/train_dataset/ \
    \
    --dataset_name data/test_dataset/ \
    --overwrite_cache True \
    --max_seq_length 384 \
    --pad_to_max_length False \
    --doc_stride 128 \
    --max_answer_length 30 \
    --eval_retrieval True \
    --num_clusters 64 \
    --top_k_retrieval 20 \
    --use_faiss False \
    \
    --output_dir code/outputs/test_dataset/ \
    --overwrite_output_dir True \
    --do_eval False \
    --do_predict True \
    --seed 42