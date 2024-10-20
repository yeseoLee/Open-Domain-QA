import ast
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForSequenceClassification, AutoTokenizer, set_seed


def rerank(
    df: pd.DataFrame,
    topk: int = 1,
    model_path: str = "Dongjin-kr/ko-reranker",
    batch_size: int = 16,
    max_length: int = 512,
):
    def exp_normalize(x):
        y = np.exp(x - x.max(axis=1, keepdims=True))
        return y / y.sum(axis=1, keepdims=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    all_questions = df["question"].tolist()
    all_contexts = df["context"].tolist()
    all_pairs = [
        [question, context]
        for question, contexts in zip(all_questions, all_contexts)
        for context in contexts
    ]

    all_scores = []

    for i in tqdm(range(0, len(all_pairs), batch_size), desc="Processing batches"):
        batch_pairs = all_pairs[i : i + batch_size]

        with torch.no_grad():
            inputs = tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            batch_scores = (
                model(**inputs, return_dict=True).logits.view(-1).float().cpu().numpy()
            )
            all_scores.extend(batch_scores)

    all_scores = np.array(all_scores)

    start = 0
    new_contexts = []
    for contexts in all_contexts:
        end = start + len(contexts)
        scores = all_scores[start:end]
        scores = exp_normalize(scores.reshape(1, -1)).flatten()
        top_indices = np.argsort(scores)[-topk:][::-1]
        top_contexts = [contexts[i] for i in top_indices]
        new_contexts.append(" ".join(top_contexts))
        start = end

    df["context"] = new_contexts

    return df


if __name__ == "__main__":
    import argparse

    # Arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--df_path", default="es_topk_20.csv", type=str, help="")
    parser.add_argument("--topk", default=5, type=int, help="")

    args = parser.parse_args()
    print(args)

    set_seed(42)
    df_for_rerank = pd.read_csv(args.df_path)
    df_for_rerank["context"] = df_for_rerank["context"].apply(
        lambda x: ast.literal_eval(x)
    )

    df_rerank = rerank(df=df_for_rerank, topk=args.topk)
    df_rerank.to_csv(f"df_rerank_{args.topk}.csv", index=False)
    print("rerank 결과 저장됨")
