"""
훈련부터 추론까지 전 과정을 수행하는 하나의 파이프라인 코드
"""

import train
import inference
from arguments import ModelArguments, DataTrainingArguments, CustomTrainingArguments
from utils.utils_qa import load_arguments, set_seed


def train_to_eval(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: CustomTrainingArguments,
):
    # train output을 eval model로 이동
    model_args.model_name_or_path = training_args.output_dir
    # eval output은 model이 아닌 prediction
    training_args.output_dir = training_args.output_dir.replace("models/", "outputs/")
    # do_train -> do_eval
    training_args.do_train = False
    training_args.do_eval = True
    return model_args, data_args, training_args


def eval_to_inference(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: CustomTrainingArguments,
):
    # dataset 변경: train -> test
    data_args.dataset_name = data_args.dataset_name.replace("train", "test")
    if data_args.retriever_from_file is not None:
        data_args.retriever_from_file = data_args.retriever_from_file.replace(
            "train", "test"
        )
    # do_eval -> do_predict
    training_args.do_eval = False
    training_args.do_predict = True
    return model_args, data_args, training_args


def main():
    """
    config.json 파일 하나로 train-eval-inference 통합 파이프라인 구축
    """
    model_args, data_args, training_args = load_arguments()
    train.run_mrc(*train.load_mrc_resources(model_args, data_args, training_args))

    model_args, data_args, training_args = train_to_eval(
        model_args, data_args, training_args
    )
    train.run_mrc(*train.load_mrc_resources(model_args, data_args, training_args))
    inference.run_mrc(
        *inference.load_mrc_resources(model_args, data_args, training_args)
    )
    model_args, data_args, training_args = eval_to_inference(
        model_args, data_args, training_args
    )
    inference.run_mrc(
        *inference.load_mrc_resources(model_args, data_args, training_args)
    )


if __name__ == "__main__":
    set_seed()
    main()
