import json
import os, sys
import pprint
import warnings
import re
import argparse
from typing import List, Optional, Tuple, Union
import pandas as pd
from datasets import Dataset
from tqdm.auto import tqdm
from retrieval.base import Retrieval
from elasticsearch import Elasticsearch, ElasticsearchWarning
from utils.utils_qa import timer

# ElasticsearchWarning 무시
warnings.filterwarnings("ignore", category=ElasticsearchWarning)


class ElasticRetrieval(Retrieval):
    def __init__(
        self, index_name, setting_path, data_path, context_path, *args, **kwargs
    ):
        self.ec = ElasticClient(
            index_name=index_name,
            setting_path=setting_path,
            dataset_path=os.path.join(data_path, context_path),
        )

    def retrieve_for_reranker(
        self, query_or_dataset: Dataset, topk: Optional[int] = 1
    ) -> pd.DataFrame:
        # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
        total = []
        with timer("query exhaustive search"):
            doc_scores, doc_indices, docs = self.get_relevant_doc_bulk(
                query_or_dataset["question"], k=topk
            )

        for idx, example in enumerate(
            tqdm(query_or_dataset, desc="Sparse retrieval with Elasticsearch: ")
        ):
            retrieved_context = []
            for i in range(min(topk, len(docs[idx]))):
                retrieved_context.append(docs[idx][i]["_source"]["document_text"])

            tmp = {
                "question": example["question"],
                "id": example["id"],
                "context": retrieved_context,
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
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices, docs = self.get_relevant_doc(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(min(topk, len(docs))):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(doc_indices[i])
                print(docs[i]["_source"]["document_text"])

            return (doc_scores, [doc_indices[i] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):
            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices, docs = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=topk
                )

            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval with Elasticsearch: ")
            ):
                # retrieved_context 구하는 부분 수정
                retrieved_context = []
                for i in range(min(topk, len(docs[idx]))):
                    retrieved_context.append(docs[idx][i]["_source"]["document_text"])

                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    # "context_id": doc_indices[idx], # type 에러 발생하여 주석처리
                    "context": " ".join(retrieved_context),  # 수정
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        doc_score = []
        doc_index = []
        res = self.ec.es_search(query, k)
        docs = res["hits"]["hits"]

        for hit in docs:
            doc_score.append(hit["_score"])
            doc_index.append(hit["_id"])
            print("Doc ID: %3r  Score: %5.2f" % (hit["_id"], hit["_score"]))

        return doc_score, doc_index, docs

    def get_relevant_doc_bulk(
        self, queries: List, k: Optional[int] = 1
    ) -> Tuple[List, List]:
        total_docs = []
        doc_scores = []
        doc_indices = []

        for query in queries:
            doc_score = []
            doc_index = []
            res = self.ec.es_search(query, k)
            docs = res["hits"]["hits"]

            for hit in docs:
                doc_score.append(hit["_score"])
                doc_indices.append(hit["_id"])

            doc_scores.append(doc_score)
            doc_indices.append(doc_index)
            total_docs.append(docs)

        return doc_scores, doc_indices, total_docs


class ElasticClient:
    def __init__(self, index_name, setting_path, dataset_path, manual_mode=False):
        self.index_name = index_name
        self.setting_path = setting_path
        self.dataset_path = dataset_path
        self.client = self.connect()
        # 직접 메서드 호출해서 사용하는 경우
        if manual_mode:
            return
        if self.create_index():
            self.insert_data()

    # elasticsearch 서버 세팅
    def connect(self):
        es = Elasticsearch(
            "http://localhost:9200", timeout=30, max_retries=10, retry_on_timeout=True
        )
        print("Ping Elasticsearch :", es.ping())
        return es

    # 인덱스 생성
    def create_index(self):
        # 이미 인덱스가 존재하는 경우
        if self.client.indices.exists(index=self.index_name):
            print("Index already exists.")
            return False

        with open(self.setting_path, "r") as f:
            setting = json.load(f)
        self.client.indices.create(index=self.index_name, body=setting)
        print("Index creation has been completed")
        return True

    # 인덱스 삭제
    def delete_index(self):
        # 인덱스가 존재하지 않는 경우
        if not self.client.indices.exists(index=self.index_name):
            print("Index doesn't exist.")
            return

        self.client.indices.delete(index=self.index_name)
        print("Index deletion has been completed")

    # 인덱스에 데이터 삽입
    def insert_data(self):
        # 삽입할 데이터 전처리
        def _preprocess(text):
            text = re.sub(r"\n", " ", text)
            text = re.sub(r"\\n", " ", text)
            text = re.sub(r"#", " ", text)
            text = re.sub(
                r"[^A-Za-z0-9가-힣.?!,()~‘’“”" ":%&《》〈〉''㈜·\-'+\s一-龥サマーン]",
                "",
                text,
            )  # サマーン 는 predictions.json에 있었음
            text = re.sub(
                r"\s+", " ", text
            ).strip()  # 두 개 이상의 연속된 공백을 하나로 치환
            # text = re.sub(r"[^A-Za-z0-9가-힣.?!,()~‘’“”"":%&《》〈〉''㈜·\-\'+\s一-龥]", "", text)

            return text

        # 위키피디아 데이터 로드
        def _load_data(dataset_path: str = "../data/wikipedia_documents.json"):
            with open(dataset_path, "r") as f:
                wiki = json.load(f)

            wiki_texts = list(dict.fromkeys([v["text"] for v in wiki.values()]))
            wiki_texts = [_preprocess(text) for text in wiki_texts]
            wiki_corpus = [
                {"document_text": wiki_texts[i]} for i in range(len(wiki_texts))
            ]
            return wiki_corpus

        wiki_corpus = _load_data(self.dataset_path)
        for i, text in enumerate(tqdm(wiki_corpus)):
            try:
                self.client.index(index=self.index_name, id=i, body=text)
            except:
                print(f"Unable to load document {i}.")

        n_records = self.client.count(index=self.index_name)["count"]
        print(f"Succesfully loaded {n_records} into {self.index_name}")
        print("@@@@@@@ 데이터 삽입 완료 @@@@@@@")

    # 삽입한 데이터 확인
    def check_data(self, doc_id=0):
        print("샘플 데이터:")
        doc = self.client.get(index=self.index_name, id=doc_id)
        pprint.pprint(doc)

    def es_search(self, question, topk):
        query = {"query": {"bool": {"must": [{"match": {"document_text": question}}]}}}
        res = self.client.search(index=self.index_name, body=query, size=topk)
        return res


if __name__ == "__main__":
    import sys
    import argparse
    from datasets import concatenate_datasets, load_from_disk

    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

    # Arguments
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", default="../data/train_dataset", type=str, help=""
    )
    parser.add_argument(
        "--model_name_or_path",
        default="klue/roberta-large",
        type=str,
        help="",
    )
    parser.add_argument("--data_path", default="../data", type=str, help="")
    parser.add_argument(
        "--context_path", default="wikipedia_documents.json", type=str, help=""
    )
    parser.add_argument("--use_faiss", default=False, type=bool, help="")
    parser.add_argument(
        "--index_name",
        default="origin-wiki",
        type=str,
        help="테스트할 index name을 설정해주세요",
    )
    parser.add_argument(
        "--setting_path",
        default="../config/elastic_setting.json",
        type=str,
        help="테스트할 setting_path를 설정해주세요",
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

    # dataset 로드
    org_dataset = load_from_disk(args.dataset_name)

    # Elasticsearch index 초기화
    ec = ElasticClient(
        index_name=args.index_name,
        setting_path=args.setting_path,
        dataset_path="../data/wikipedia_documents.json",
        manual_mode=True,
    )

    # Elasticsearch 사용
    retriever = ElasticRetrieval(
        index_name=args.index_name,
        setting_path=args.setting_path,
        data_path="../data/",
        context_path="wikipedia_documents.json",
    )

    if args.is_for_reranker:
        df_for_reranker = retriever.retrieve_for_reranker(
            org_dataset["validation"], args.topk
        )
        df_for_reranker.to_csv(f"es_{args.topk}.csv", index=False)
        print("rerank를 위한 es retriver 결과 저장됨")
    else:
        # Test sparse
        full_ds = concatenate_datasets(
            [
                org_dataset["train"].flatten_indices(),
                org_dataset["validation"].flatten_indices(),
            ]
        )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
        print("*" * 40, "query dataset", "*" * 40)
        print(full_ds)
        print(len(org_dataset["train"]), len(org_dataset["validation"]))

        query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
        if args.use_faiss:

            # test single query
            with timer("single query by faiss"):
                scores, indices = retriever.retrieve_faiss(query)

            # test bulk
            with timer("bulk query by exhaustive search"):
                df = retriever.retrieve_faiss(full_ds)
                df["correct"] = df["original_context"] == df["context"]

                print(
                    "correct retrieval result by faiss", df["correct"].sum() / len(df)
                )

        else:
            with timer("bulk query by exhaustive search"):
                df = retriever.retrieve(full_ds, topk=args.topk)
                df["correct"] = [
                    original_context in context
                    for original_context, context in zip(
                        df["original_context"], df["context"]
                    )
                ]
                print(
                    "correct retrieval result by exhaustive search",
                    f"{df['correct'].sum()}/{len(df)}",
                    df["correct"].sum() / len(df),
                )

            with timer("single query by exhaustive search"):
                scores, indices = retriever.retrieve(query)
