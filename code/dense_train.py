import json
import os
import numpy as np
import random
from datetime import datetime
from tqdm import tqdm, trange
import pickle
import wandb

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AdamW, get_linear_schedule_with_warmup
)
from BertEncoder import BertEncoder


class DenseRetrieval:
    def __init__(self,
        args,
        dataset,
        val_dataset,
        num_neg,
        tokenizer,
        p_encoder,
        q_encoder
    ):
        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder.to(args.device)
        self.q_encoder = q_encoder.to(args.device)

        self.prepare_in_batch_negative(num_neg=num_neg)
        self.prepare_validation_dataloader(val_dataset)

    def prepare_in_batch_negative(self,
        dataset=None,
        num_neg=2,
        tokenizer=None
    ):
        # negative sampling
        training_dataset = list(set([example for example in self.dataset["context"]]))

        corpus = np.array(training_dataset)
        p_with_neg = []

        for c in self.dataset["context"]:
            while True:
                neg_idxs = np.random.randint(len(corpus), size=num_neg)

                if not c in corpus[neg_idxs]:
                    p_with_neg.append(c)
                    p_with_neg.extend(corpus[neg_idxs])
                    break

        # tokenizer
        q_seqs = self.tokenizer(
            self.dataset["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        p_seqs = self.tokenizer(
            p_with_neg,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 1, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(-1, num_neg + 1, max_len)
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(-1, num_neg + 1, max_len)

        train_dataset = TensorDataset(
            p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"],
            q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
        )

        self.batch_size = self.args.per_device_train_batch_size
        # Dataloader
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, drop_last=True)


    # Validation 데이터 전처리 및 dataloader 만들기
    def prepare_validation_dataloader(self, dataset):
        # negative sampling은 하지 않고, 검증 데이터의 question과 context를 사용
        q_seqs = self.tokenizer(
            dataset["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        p_seqs = self.tokenizer(
            dataset["context"],
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        val_dataset = TensorDataset(
            p_seqs["input_ids"], p_seqs["attention_mask"], p_seqs["token_type_ids"],
            q_seqs["input_ids"], q_seqs["attention_mask"], q_seqs["token_type_ids"]
        )

        self.val_dataloader = DataLoader(val_dataset, batch_size=self.args.per_device_eval_batch_size)


    def train(self,
        p_encoder_name,
        q_encoder_name,
        args=None
    ):
        torch.cuda.empty_cache()
        
        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.p_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.p_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
            {"params": [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)], "weight_decay": self.args.weight_decay},
            {"params": [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon
        )
        t_total = len(self.train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=t_total
        )

        # Start
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        for t in train_iterator:

            # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    self.p_encoder.train()
                    self.q_encoder.train()

                    targets = torch.zeros(self.batch_size).long() # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(self.args.device)

                    p_inputs = {
                        "input_ids": batch[0].view(self.batch_size * (self.num_neg + 1), -1).to(self.args.device),
                        "attention_mask": batch[1].view(self.batch_size * (self.num_neg + 1), -1).to(self.args.device),
                        "token_type_ids": batch[2].view(self.batch_size * (self.num_neg + 1), -1).to(self.args.device)
                    }

                    q_inputs = {
                        "input_ids": batch[3].to(self.args.device),
                        "attention_mask": batch[4].to(self.args.device),
                        "token_type_ids": batch[5].to(self.args.device)
                    }

                    del batch
                    torch.cuda.empty_cache()
                    # (batch_size * (num_neg + 1), emb_dim)
                    p_outputs = self.p_encoder(**p_inputs)
                    # (batch_size, emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(self.batch_size, self.num_neg + 1, -1) # (batch_size, sample_num, emb_size)
                    q_outputs = q_outputs.view(self.batch_size, 1, -1) # (batch_size, 1, emb_size)

                    # (batch_size, num_neg + 1)
                    sim_scores = torch.bmm(q_outputs, torch.transpose(p_outputs, 1, 2)).squeeze()
                    sim_scores = sim_scores.view(self.batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets) # -log(sim_scores - targets) → 결국 sim_scores가 클수록 loss 값은 0에 가까워짐. sim_scores가 0에 가까울수록 loss는 무한히 (-)로 커짐.
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    wandb.log({"loss": loss})

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.q_encoder.zero_grad()
                    self.p_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()
                    del p_inputs, q_inputs

            val_loss = self.evaluate(self.val_dataloader)
            print(f"Validation Loss: {val_loss}")

            wandb.log({"validation_loss": val_loss})
            
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(self.args.output_dir + p_encoder_name, "wb") as f:
               pickle.dump(self.p_encoder, f)

            with open(self.args.output_dir + q_encoder_name, "wb") as f:
               pickle.dump(self.q_encoder, f)


    def evaluate(self, dataloader):
        self.p_encoder.eval()
        self.q_encoder.eval()
        total_loss = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"): # batch[0].size() == (batch_size, max_length)
                targets = torch.zeros(batch[0].size(0)).long().to(self.args.device)

                p_inputs = {
                    "input_ids": batch[0].to(self.args.device),
                    "attention_mask": batch[1].to(self.args.device),
                    "token_type_ids": batch[2].to(self.args.device),
                }
                q_inputs = {
                    "input_ids": batch[3].to(self.args.device),
                    "attention_mask": batch[4].to(self.args.device),
                    "token_type_ids": batch[5].to(self.args.device),
                }

                p_outputs = self.p_encoder(**p_inputs).unsqueeze(1) # (batch_size, max_length, emb_size)
                q_outputs = self.q_encoder(**q_inputs).unsqueeze(1)


                sim_scores = torch.bmm(q_outputs, p_outputs.transpose(1, 2)).squeeze() # (batch_size, 1)
                sim_scores = sim_scores.view(self.batch_size, -1)
                sim_scores = F.log_softmax(sim_scores, dim=1)

                loss = F.nll_loss(sim_scores, targets)
                total_loss += loss.item()

            return total_loss / len(dataloader)


# wandB 초기화 및 저장 폴더 생성
def wandb_init(run_name, base_path="../wandb"):
    # 현재 시각
    current_time = datetime.now().strftime("%m%d_%H%M")
    wandb_name = f"klue/bert-base_{current_time}"

    # WandB 초기화
    wandb.init(project="wandb_logs", name=wandb_name, dir=base_path,
                config={
                "learning_rate" : 5e-5,
                "batch_size" : 8,
                "epoch" : 3
        })  # 프로젝트 이름과 실험 이름 설정

    # set run name
    wandb.run.name = run_name
    wandb.run.save()


if __name__=="__main__":
    def set_seed(random_seed):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
        random.seed(random_seed)
        np.random.seed(random_seed)

    set_seed(42) # magic number :)

    # WandB 초기화
    wandb_init("klue/bert-base_1")

    # "/bin_name.bin"
    p_encoder_name = "/p_test.bin"
    q_encoder_name = "/q_test.bin"

    # load dataset
    dataset = load_from_disk("../data/train_dataset")
    with open('../data/korquad_train_dataset.json', "r") as f:
        train_dataset = json.load(f)

    # train_dataset = dataset["train"]
    val_dataset = dataset["validation"]

    # args set
    args = TrainingArguments(
        output_dir="./retrieval/dense_encoder",
        evaluation_strategy="epoch",
        learning_rate=4e-5,
        per_device_train_batch_size=5,
        per_device_eval_batch_size=5,
        num_train_epochs=2,
        weight_decay=0.01
    )
    model_checkpoint = "klue/bert-base"

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint).to(args.device)

    # Retrieval 객체 생성
    retriever = DenseRetrieval(
        args=args,
        dataset=train_dataset,
        val_dataset=val_dataset,
        num_neg=2,
        tokenizer=tokenizer,
        p_encoder=p_encoder,
        q_encoder=q_encoder
    )

    # train
    retriever.train(p_encoder_name, q_encoder_name)