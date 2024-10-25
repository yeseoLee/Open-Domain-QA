import json
import os
from typing import List, Optional, Union, Callable
from datasets import Dataset


class Retrieval:
    def __init__(
        self,
        tokenize_fn: Optional[Callable] = None,
        data_path: Optional[str] = None,
        context_path: Optional[str] = None,
        use_title: Optional[bool] = None,
        index_name: Optional[str] = None,
        setting_path: Optional[str] = None,
    ):
        self.data_path = data_path
        with open(
            os.path.join(data_path, context_path),
            "r",
            encoding="utf-8",
        ) as f:
            wiki = json.load(f)
        self.contexts = list(
            dict.fromkeys(
                [
                    f'{v["title"]}: {v["text"]}' if use_title else v["text"]
                    for v in wiki.values()
                ]
            )
        )
        self.ids = list(range(len(self.contexts)))
        self.tokenize_fn = tokenize_fn
        if index_name:
            self.index_name = index_name
        if setting_path:
            self.setting_path = setting_path

    def retrieve(self, query_or_dataset: Union[str, Dataset], topk: Optional[int]):
        raise NotImplementedError

    def get_embedding(self):
        raise NotImplementedError

    def get_relevant_doc(self, query: str, k: Optional[int]):
        raise NotImplementedError

    def get_relevant_doc_bulk(self, queries: List, k: Optional[int]):
        raise NotImplementedError
