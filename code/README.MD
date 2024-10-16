# 실행 방법
## 통합 파이프라인으로 실행
```bash
# train-eval-inference 통합 파이프라인
python integration_pipline.py ./config/integration.json
```

## config.json으로 실행
```bash
# 학습
python train.py ./config/train.json
# 평가
python train.py ./config/eval.json
# 추론
python inference.py ./config/inference.json
```

## 쉘 스크립트로 실행
### 학습 또는 평가 방법
```bash
python train.py \
    --model_name_or_path klue/bert-base \
    --config_name None \
    --tokenizer_name None \
    \
    --dataset_name ../data/train_dataset \
    --max_seq_length 384 \
    --pad_to_max_length False \
    --doc_stride 128 \
    --max_answer_length 30 \
    --overwrite_cache True \
    --preprocessing_num_workers None \
    \
    --output_dir ../models/train_dataset \
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
```

### 추론 방법
```bash
python inference.py \
    --model_name_or_path ../models/train_dataset/ \
    \
    --dataset_name ../data/test_dataset/ \
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
``` 