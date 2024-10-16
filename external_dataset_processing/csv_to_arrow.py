import os
import json
import pandas as pd
from numpy import array
from datasets import Dataset, DatasetDict


def csv_to_arrow(train_filename, valid_filename, output_dir):
    train_df = pd.read_csv(train_filename)
    valid_df = pd.read_csv(valid_filename)

    # type str -> object(dict)로 변환
    train_df["answers"] = train_df["answers"].apply(lambda x: eval(x))
    valid_df["answers"] = valid_df["answers"].apply(lambda x: eval(x))

    # Dataset으로 변환 (from pandas DataFrame)
    train_dataset = Dataset.from_pandas(train_df)

    # 동일하게 validation과 test도 처리 가능
    valid_dataset = Dataset.from_pandas(valid_df)

    # DatasetDict로 결합
    dataset_dict = DatasetDict(
        {
            "train": train_dataset,
            "validation": valid_dataset,
        }
    )

    # Arrow 파일로 저장
    dataset_dict.save_to_disk(output_dir)
    print(f"Augmented dataset saved to {output_dir}")


def modify_filename(output_dir):
    TARGET_FILE_NAME = "dataset.arrow"

    # JSON 파일 경로
    train_state = f"{output_dir}/train/state.json"
    valid_state = f"{output_dir}/validation/state.json"

    # arrow 파일 경로
    train_arrow = f"{output_dir}/train/data-00000-of-00001.arrow"
    valid_arrow = f"{output_dir}/validation/data-00000-of-00001.arrow"

    def modify_json(json_filename):
        # JSON 파일 읽기
        with open(json_filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        # filename 변경
        data["_data_files"][0]["filename"] = TARGET_FILE_NAME

        # 수정된 JSON 파일 덮어쓰기
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)

    def modify_arrow(arrow_filename):
        if os.path.exists(arrow_filename):
            old_filepath, _ = os.path.split(arrow_filename)
            os.rename(arrow_filename, os.path.join(old_filepath, TARGET_FILE_NAME))
        else:
            print(f"파일 '{arrow_filename}'이 존재하지 않습니다.")

    modify_json(train_state)
    modify_json(valid_state)

    modify_arrow(train_arrow)
    modify_arrow(valid_arrow)


if __name__ == "__main__":
    import argparse

    # Arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--train_filename", metavar="train_dataset.csv", type=str, help=""
    )
    parser.add_argument(
        "--valid_filename", metavar="train_validation_dataset.csv", type=str, help=""
    )
    parser.add_argument("--output_dir", metavar="train_dataset_v0", type=str, help="")
    args = parser.parse_args()

    csv_to_arrow(args.train_filename, args.valid_filename, args.output_dir)
    modify_filename(args.output_dir)

    # python csv_to_arrow.py --train_filename train_dataset.csv --valid_filename train_validation_dataset.csv --output_dir train_dataset_v0
