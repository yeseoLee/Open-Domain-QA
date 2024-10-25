if __name__ == "__main__":
    import sys, os

    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import os
import pickle
import sys
from datasets import Dataset
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm
from retrieval.base import Retrieval
from utils.utils_qa import timer


class BM25Retrieval(Retrieval):
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        use_title=True,
        *args,
        **kwargs,
    ) -> None:
        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            use_title:
                contexts에 title을 추가할지 정합니다.

        Summary:
            Passage 파일을 불러오고 BM25Okapi를 선언하는 기능을 합니다.
        """
        super().__init__(tokenize_fn, data_path, context_path, use_title)

        self.get_sparse_embedding()

        # Tokenize
        # self.tokenized_corpus = [self.tokenize_fn(doc) for doc in self.contexts]

        # Transform by vectorizer
        # self.bm25 = BM25Okapi(self.tokenized_corpus, k1=1.5, b=0.75)

    def get_sparse_embedding(self) -> None:
        """Summary:
        Passage Embedding을 만들고
        TFIDF와 Embedding을 pickle로 저장합니다.
        만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 불러오거나 저장합니다.
        pickle_name = f"bm25_sparse_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.bm25 = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")

            tokenized_corpus = [self.tokenize_fn(doc) for doc in self.contexts]
            self.bm25 = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)

            with open(emd_path, "wb") as file:
                pickle.dump(self.bm25, file)
            print("Embedding pickle saved.")

    def retrieve_for_reranker(
        self, query_or_dataset: Dataset, topk: Optional[int] = 1
    ) -> pd.DataFrame:
        # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
        total = []
        with timer("query exhaustive search using BM25"):
            doc_scores, doc_indices = self.get_relevant_doc_bulk(
                query_or_dataset["question"], k=topk
            )
        for idx, example in enumerate(
            tqdm(query_or_dataset, desc="Sparse retrieval: ")
        ):
            tmp = {
                # Query와 해당 id를 반환합니다.
                "question": example["question"],
                "id": example["id"],
                # Retrieve한 Passage의 id, context를 반환합니다.
                "context": [self.contexts[pid] for pid in doc_indices[idx]],
            }
            if "context" in example.keys() and "answers" in example.keys():
                # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                tmp["original_context"] = example["context"]
                tmp["answers"] = example["answers"]
            total.append(tmp)

        return pd.DataFrame(total)

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                query를 포함한 HF.Dataset을 받으면 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[query exhaustive search using BM25]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search using BM25"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        with timer("transform"):
            tokenized_query = self.tokenize_fn(query)

        with timer("query exhaustive search"):
            result = self.bm25.get_scores(tokenized_query)
        if not isinstance(result, np.ndarray):
            result = result.toarray()

        sorted_result = np.argsort(result.squeeze())[::-1]
        doc_score = result.squeeze()[sorted_result].tolist()[:k]
        doc_indices = sorted_result.tolist()[:k]
        return doc_score, doc_indices

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        """
        Arguments:
            queries (List):
                여러 개의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """

        tokenized_query = [self.tokenize_fn(query) for query in queries]

        doc_scores = []
        doc_indices = []

        for i in range(len(tokenized_query)):
            query = tokenized_query[i]
            result = self.bm25.get_scores(query)
            if not isinstance(result, np.ndarray):
                result = result.toarray()
            sorted_result = np.argsort(result.squeeze())[::-1]
            doc_scores.append(result.squeeze()[sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])

        return doc_scores, doc_indices


if __name__ == "__main__":
    import argparse
    from datasets import concatenate_datasets, load_from_disk

    # Arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", default="../data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        default="monologg/koelectra-base-v3-finetuned-korquad",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", default="../data", type=str, help="")
    parser.add_argument(
        "--context_path", default="wikipedia_documents.json", type=str, help=""
    )
    parser.add_argument(
        "--topk",
        default=20,
        type=int,
        help="retrieve할 topk 문서의 개수를 설정해주세요",
    )
    parser.add_argument(
        "--is_for_reranker",
        default=False,
        type=bool,
        help="reranker 용도일 때 설정해주세요",
    )

    args = parser.parse_args()
    print(args)

    # Test sparse
    org_dataset = load_from_disk(args.dataset_name)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
    )

    retriever = BM25Retrieval(
        tokenize_fn=tokenizer.tokenize,
        data_path=args.data_path,
        context_path=args.context_path,
        use_title=False,
    )

    if args.is_for_reranker:
        df_for_reranker = retriever.retrieve_for_reranker(
            org_dataset["validation"], args.topk
        )
        df_for_reranker.to_csv(f"bm25_{args.topk}.csv", index=False)
        print("rerank를 위한 bm25 retriver 결과 저장됨")
    else:
        full_ds = concatenate_datasets(
            [
                org_dataset["train"].flatten_indices(),
                org_dataset["validation"].flatten_indices(),
            ]
        )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
        print("*" * 40, "query dataset", "*" * 40)
        print(full_ds)
        query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"

        # test single query
        with timer("single query by exhaustive search using bm25"):
            scores, indices = retriever.retrieve(query)

        # test bulk
        with timer("bulk query by exhaustive search using bm25"):
            df = retriever.retrieve(full_ds)
            df["correct"] = df["original_context"] == df["context"]
            print(
                "correct retrieval result by exhaustive search",
                df["correct"].sum() / len(df),
            )
