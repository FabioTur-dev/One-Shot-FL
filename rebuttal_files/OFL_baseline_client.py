#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client-side FAIR implementation of Dense and Co-Boost baselines
(Data-free, no-cheating, only client-local train data is used)

Outputs:
Each client trains a LinearHead on its own Z_train subset,
and saves head.pt under:

  ./oneshot_bench/{DATASET}/{ALPHA}/clients/DenseCB_fair/client_XX/head.pt
"""

import os
import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# ============================================================
# CONFIG
# ============================================================
DATASETS = ["BANKING77", "CLINC150", "AG_NEWS", "DBPEDIA_14", "SST2"]
ALPHAS = [0.5, 0.01]
NUM_CLIENTS = 10
SEED = 42

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64
BATCH = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

K_RP = 256
RP_SEED = 12345

LR = 5e-3
WD = 1e-4
EPOCHS = 30

OUT_ROOT = "./oneshot_bench/{DS}/{ALPHA}/clients/DenseCB_fair"

torch.set_grad_enabled(True)


# ============================================================
# UTILITIES
# ============================================================
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def rp_matrix(d, k, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(d, k, generator=g) / math.sqrt(d)


def dirichlet_split(labels, C, K, alpha):
    labels = torch.tensor(labels)
    splits = [[] for _ in range(K)]
    for c in range(C):
        idx = torch.where(labels == c)[0]
        if len(idx) == 0:
            continue
        idx = idx[torch.randperm(len(idx))]

        dist = torch.distributions.Dirichlet(
            torch.full((K,), float(alpha))
        ).sample()
        counts = (dist * len(idx)).long().tolist()

        diff = len(idx) - sum(counts)
        for i in range(abs(diff)):
            counts[i % K] += 1 if diff > 0 else -1

        s = 0
        for k in range(K):
            e = s + counts[k]
            if e > s:
                splits[k].extend(idx[s:e].tolist())
            s = e

    for k in range(K):
        random.shuffle(splits[k])
    return splits


# ============================================================
# LOADERS (dataset → train/test)
# ============================================================
def load_splits(dataset_name):
    name = dataset_name.upper()

    if name == "SST2":
        ds = load_dataset("glue", "sst2")
        train = ds["train"]
        test = ds["validation"]
        return (
            list(train["sentence"]),
            list(train["label"]),
            list(test["sentence"]),
            list(test["label"]),
            ["negative", "positive"],
            2,
        )

    if name == "AG_NEWS":
        ds = load_dataset("ag_news")
        train = ds["train"]
        test = ds["test"]
        class_names = ["World", "Sports", "Business", "Sci/Tech"]
        return (
            list(train["text"]),
            list(train["label"]),
            list(test["text"]),
            list(test["label"]),
            class_names,
            4,
        )

    if name == "DBPEDIA_14":
        ds = load_dataset("dbpedia_14")
        train = ds["train"]
        test = ds["test"]
        class_names = [str(i) for i in range(14)]
        return (
            list(train["content"]),
            list(train["label"]),
            list(test["content"]),
            list(test["label"]),
            class_names,
            14,
        )

    if name == "BANKING77":
        data_files = {
            "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
            "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
        }
        d = load_dataset("csv", data_files=data_files)
        train, test = d["train"], d["test"]

        texts_train = list(train["text"])
        labels_train_str = list(train["category"])
        texts_test = list(test["text"])
        labels_test_str = list(test["category"])

        class_names = sorted(set(labels_train_str + labels_test_str))
        label2id = {c: i for i, c in enumerate(class_names)}

        y_train = [label2id[x] for x in labels_train_str]
        y_test = [label2id[x] for x in labels_test_str]

        return (
            texts_train,
            y_train,
            texts_test,
            y_test,
            class_names,
            len(class_names),
        )

    if name in ("CLINC150", "CLINIC150"):
        data_files = "https://huggingface.co/datasets/contemmcm/clinc150/resolve/main/data_full.csv"
        d = load_dataset("csv", data_files=data_files)
        full = d["train"]

        # Identify columns
        cols = full.column_names

        if "text" in cols:
            text_col = "text"
        else:
            text_col = [c for c in cols if "text" in c.lower() or "utterance" in c.lower()][0]

        if "intent" in cols:
            intent_col = "intent"
        else:
            intent_col = [c for c in cols if "intent" in c.lower() or "label" in c.lower()][0]

        if "split" in cols:
            split_col = "split"
        else:
            split_col = [c for c in cols if "split" in c.lower()][0]

        # Build label map
        class_names = sorted(set(full[intent_col]))
        label2id = {x: i for i, x in enumerate(class_names)}

        texts_train, labels_train = [], []
        texts_test, labels_test = [], []

        for txt, sp, lab in zip(full[text_col], full[split_col], full[intent_col]):
            if sp == "train":
                texts_train.append(txt)
                labels_train.append(label2id[lab])
            else:
                texts_test.append(txt)
                labels_test.append(label2id[lab])

        return (
            texts_train,
            labels_train,
            texts_test,
            labels_test,
            class_names,
            len(class_names),
        )

    raise ValueError(f"Dataset {dataset_name} non supportato.")


# ============================================================
# ENCODER
# ============================================================
def build_encoder():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = AutoModel.from_pretrained(MODEL_NAME)
    enc.eval().to(DEVICE)
    for p in enc.parameters():
        p.requires_grad_(False)
    return tok, enc


def encode_batch(tok, xs):
    return tok(xs, padding=True, truncation=True,
               max_length=MAX_LEN, return_tensors="pt")


@torch.no_grad()
def extract_cls(enc, batch_inputs):
    out = enc(**batch_inputs)
    return out.last_hidden_state[:, 0, :]


def encode_many(tok, enc, xs, R):
    outs = []
    dl = DataLoader(xs, batch_size=128, shuffle=False)
    for batch in dl:
        tb = encode_batch(tok, batch)
        tb = {k: v.to(DEVICE) for k, v in tb.items()}
        x = extract_cls(enc, tb)
        z = x @ R
        outs.append(z.cpu())
    return torch.cat(outs)


# ============================================================
# HEAD MODEL
# ============================================================
class LinearHead(torch.nn.Module):
    def __init__(self, k, C):
        super().__init__()
        self.fc = torch.nn.Linear(k, C)

    def forward(self, z):
        return self.fc(z)


# ============================================================
# TRAIN HEAD
# ============================================================
def train_head(Zc, yc, Z_test, y_test, C, cid):
    if Zc.size(0) == 0:
        print(f"[CLIENT {cid:02d}] dataset vuoto → skip")
        return None

    ds = TensorDataset(Zc, yc)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True)

    head = LinearHead(K_RP, C).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    ce = torch.nn.CrossEntropyLoss()

    for ep in range(1, EPOCHS + 1):
        head.train()
        for zb, yb in dl:
            zb, yb = zb.to(DEVICE), yb.to(DEVICE)
            loss = ce(head(zb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # evaluation (solo logging)
    with torch.no_grad():
        head.eval()
        pred = head(Z_test.to(DEVICE)).argmax(1)
        acc = (pred.cpu() == y_test).float().mean().item() * 100
        print(f"[CLIENT {cid:02d}] test-acc ~ {acc:.2f}%")

    return head


# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(SEED)
    tok, enc = build_encoder()

    for dset in DATASETS:
        print(f"\n=== DATASET {dset} ===")
        texts_train, labels_train, texts_test, labels_test, class_names, C = load_splits(dset)

        y_train = torch.tensor(labels_train)
        y_test = torch.tensor(labels_test)

        print("[ENC] encoding...")
        d = enc.config.hidden_size
        R = rp_matrix(d, K_RP, RP_SEED).to(DEVICE)

        Z_train = encode_many(tok, enc, texts_train, R)
        Z_test = encode_many(tok, enc, texts_test, R)

        for alpha in ALPHAS:
            print(f"\n[ALPHA={alpha}] split clients...")
            splits = dirichlet_split(labels_train, C, NUM_CLIENTS, alpha)

            out_root = OUT_ROOT.format(DS=dset.upper(), ALPHA=alpha)
            os.makedirs(out_root, exist_ok=True)

            for cid, idxs in enumerate(splits):
                Zc = Z_train[idxs]
                yc = y_train[idxs]

                head = train_head(Zc, yc, Z_test, y_test, C, cid)
                if head is None:
                    continue

                client_dir = f"{out_root}/client_{cid:02d}"
                os.makedirs(client_dir, exist_ok=True)
                torch.save(
                    {"state_dict": head.state_dict(), "C": C},
                    f"{client_dir}/head.pt"
                )


if __name__ == "__main__":
    main()
