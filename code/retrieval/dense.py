import json
import os
import torch
import pickle
from typing import List, Tuple, Optional
from datasets import load_from_disk
from transformers import AutoTokenizer
from pprint import pprint

class DenseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_documents.json",
        model_checkpoint: str = "klue/bert-base",
        passage_encoder_path: str = "dense_retrieval/p_klue_bert_base.bin_3",
        query_encoder_path: str = "dense_retrieval/q_klue_bert_base.bin_3",
        use_title: bool = True,
    ) -> None:
        self.data_path = data_path
        self.context_path = context_path
        self.use_title = use_title
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        # Load pre-trained encoders (passage and query encoders)
        with open(passage_encoder_path, "rb") as f:
            self.p_encoder = pickle.load(f)
        with open(query_encoder_path, "rb") as f:
            self.q_encoder = pickle.load(f)

        # Ensure the models are in eval mode
        self.p_encoder.eval()
        self.q_encoder.eval()

    def retrieve(self, query: str, passages: List[str], k: int = 5) -> List[int]:
        """Retrieve the top-k relevant passages for the given query."""
        # Tokenize query
        with torch.no_grad():
            q_inputs = self.tokenizer([query], padding="max_length", truncation=True, return_tensors="pt").to("cuda")
            q_emb = self.q_encoder(**q_inputs).to("cpu")  # (num_query, emb_dim)

            # Encode passages and compute similarity scores
            p_embs = []
            for passage in passages:
                p_inputs = self.tokenizer(passage, padding="max_length", truncation=True, return_tensors="pt").to("cuda")
                p_emb = self.p_encoder(**p_inputs).to("cpu").numpy()  # (1, emb_dim)
                p_embs.append(p_emb)

            p_embs = torch.Tensor(p_embs).squeeze()  # (num_passage, emb_dim)
            dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))  # (num_query, num_passage)
            ranked_indices = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze().tolist()

        return ranked_indices[:k]

    def get_relevant_doc(self, query: str, k: int = 5) -> Tuple[List[str], List[int]]:
        """Retrieve the top-k relevant passages and their indices for a given query."""
        # Load the context (Wikipedia passages)
        context_file = os.path.join(self.data_path, self.context_path)
        with open(context_file, "r", encoding="utf-8") as f:
            wiki_data = json.load(f)
        contexts = list(dict.fromkeys([value["text"] for value in wiki_data.values()]))

        # Retrieve relevant passages
        ranked_indices = self.retrieve(query=query, passages=contexts, k=k)
        retrieved_passages = [contexts[idx] for idx in ranked_indices]

        return retrieved_passages, ranked_indices

    def get_topk_passages_for_dataset(self, dataset, k: int = 5):
        """Retrieve top-k passages for all queries in a dataset."""
        queries = dataset["question"]
        contexts = dataset["context"]

        for query in queries[:5]:  # Example: retrieve for the first 5 queries
            retrieved_passages, indices = self.get_relevant_doc(query=query, k=k)
            print(f"[Query] {query}")
            for i, (passage, idx) in enumerate(zip(retrieved_passages, indices)):
                print(f"Top-{i + 1} Passage (Index {idx}):")
                pprint(passage)

if __name__ == "__main__":
    # Load dataset (train_dataset assumed to be at data_path)
    data_path = "../data/"
    dataset = load_from_disk(os.path.join(data_path, "train_dataset"))
    val_dataset = dataset["validation"]

    # Initialize DenseRetrieval object
    dense_retriever = DenseRetrieval(
        tokenize_fn=None,  # You can pass a custom tokenize_fn if needed
        data_path=data_path,
        context_path="wikipedia_documents.json",
        model_checkpoint="klue/bert-base",
        passage_encoder_path="dense_retrieval/p_klue_bert_base.bin_3",
        query_encoder_path="dense_retrieval/q_klue_bert_base.bin_3",
    )

    # Example: retrieve top-5 passages for the validation dataset
    dense_retriever.get_topk_passages_for_dataset(val_dataset, k=5)