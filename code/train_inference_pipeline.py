"""
훈련부터 추론까지 전 과정을 수행하는 하나의 파이프라인 코드
"""

import train
import inference
from utils.utils_qa import load_arguments


def main():
    """
    TODO: config.json 파일 하나로 train-eval-inference 통합 시스템 구축
    - arguments.py 수정 (train용, inference용 분리)
    - load_arguments 수정
    """

    model_args, data_args, training_args = load_arguments()
    train.run_mrc(train.load_mrc_resources((model_args, data_args, training_args)))
    inference.run_mrc(
        inference.load_mrc_resources((model_args, data_args, training_args))
    )


if __name__ == "__main__":
    main()

"""
# ess-exp-roberta

# python train.py --model_name_or_path klue/roberta-large --base_model roberta --output_dir ./models/base_roberta_ess --save_steps 1000 --per_device_train_batch_size 16 --gradient_accumulation_steps 4 --do_train

# python train.py --base_model roberta --output_dir ./outputs/base_roberta_ess --model_name_or_path ./models/base_roberta_ess/ --do_eval 

# python inference_es.py --base_model roberta --output_dir ./outputs/base_roberta_ess/ --dataset_name ../data/test_dataset/ --model_name_or_path ./models/base_roberta_ess/ --per_device_eval_batch_size 16 --do_predict
"""
