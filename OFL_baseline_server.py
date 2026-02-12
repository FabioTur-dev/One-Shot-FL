#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FAIR One-Shot FL baselines (server-side)
========================================

Implementa tre metodi server-side data-free:

  1. Ensemble dei client heads (teacher)
  2. Dense Student (MLP distillato sui teacher logits)
  3. Co-Boost Students (S1 + S2 + Ensemble)

FAIR e 100% NON-cheating:
 - il server vede SOLO il test set pubblico
 - non accede a nessun dato raw dei client
 - non allena sui dati dei client
 - usa SOLO i logits delle head client come teacher
"""

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel


# ============================================================
# CONFIG
# ============================================================

DATASETS = ["BANKING77", "CLINC150", "AG_NEWS", "DBPEDIA_14", "SST2"]
ALPHAS = [0.5, 0.01]

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

K_RP = 256
RP_SEED = 12345

BATCH = 128
SEED = 42

LR = 5e-3
WD = 1e-4

EPOCHS_DENSE = 60
EPOCHS_S1 = 60
EPOCHS_S2 = 40

torch.set_grad_enabled(True)


# ============================================================
# UTILITIES (identiche al client-side)
# ============================================================

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def rp_matrix(d, k, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(d, k, generator=g) / math.sqrt(d)


def load_splits(dataset_name):
    name = dataset_name.upper()

    if name == "SST2":
        ds = load_dataset("glue", "sst2")
        return (
            list(ds["train"]["sentence"]),
            list(ds["train"]["label"]),
            list(ds["validation"]["sentence"]),
            list(ds["validation"]["label"]),
            ["negative", "positive"],
            2,
        )

    if name == "AG_NEWS":
        ds = load_dataset("ag_news")
        return (
            list(ds["train"]["text"]),
            list(ds["train"]["label"]),
            list(ds["test"]["text"]),
            list(ds["test"]["label"]),
            ["World", "Sports", "Business", "Sci/Tech"],
            4,
        )

    if name == "DBPEDIA_14":
        ds = load_dataset("dbpedia_14")
        return (
            list(ds["train"]["content"]),
            list(ds["train"]["label"]),
            list(ds["test"]["content"]),
            list(ds["test"]["label"]),
            [str(i) for i in range(14)],
            14,
        )

    if name == "BANKING77":
        data_files = {
            "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
            "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
        }
        d = load_dataset("csv", data_files=data_files)
        train, test = d["train"], d["test"]

        labels_train = list(train["category"])
        labels_test = list(test["category"])

        class_names = sorted(set(labels_train + labels_test))
        label2id = {c: i for i, c in enumerate(class_names)}

        return (
            list(train["text"]),
            [label2id[x] for x in labels_train],
            list(test["text"]),
            [label2id[x] for x in labels_test],
            class_names,
            len(class_names),
        )

    if name in ("CLINC150", "CLINIC150"):
        data_files = "https://huggingface.co/datasets/contemmcm/clinc150/resolve/main/data_full.csv"
        d = load_dataset("csv", data_files=data_files)
        full = d["train"]

        cols = full.column_names
        if "text" in cols: text_col = "text"
        else: text_col = [c for c in cols if "text" in c.lower()][0]

        if "intent" in cols: intent_col = "intent"
        else: intent_col = [c for c in cols if "intent" in c.lower()][0]

        if "split" in cols: split_col = "split"
        else: split_col = [c for c in cols if "split" in c.lower()][0]

        class_names = sorted(set(full[intent_col]))
        label2id = {x: i for i, x in enumerate(class_names)}

        texts_train, y_train = [], []
        texts_test, y_test = [], []

        for txt, sp, lab in zip(full[text_col], full[split_col], full[intent_col]):
            if sp == "train":
                texts_train.append(txt)
                y_train.append(label2id[lab])
            else:
                texts_test.append(txt)
                y_test.append(label2id[lab])

        return texts_train, y_train, texts_test, y_test, class_names, len(class_names)

    raise ValueError(f"Dataset {dataset_name} non supportato.")


def build_encoder():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = AutoModel.from_pretrained(MODEL_NAME)
    enc.eval().to(DEVICE)
    for p in enc.parameters():
        p.requires_grad_(False)
    return tok, enc


def encode_batch(tok, xs):
    return tok(xs, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")


@torch.no_grad()
def extract_cls(enc, batch_inputs):
    out = enc(**batch_inputs)
    return out.last_hidden_state[:, 0, :]


def encode_many(tok, enc, xs, R):
    outs = []
    dl = DataLoader(xs, batch_size=128, shuffle=False, num_workers=0)
    for batch in dl:
        tb = encode_batch(tok, batch)
        tb = {k: v.to(DEVICE) for k, v in tb.items()}
        x = extract_cls(enc, tb)
        z = x @ R
        outs.append(z.cpu())
    return torch.cat(outs)


# ============================================================
# LINEAR HEAD
# ============================================================

class LinearHead(nn.Module):
    def __init__(self, k, C):
        super().__init__()
        self.fc = nn.Linear(k, C)

    def forward(self, z):
        return self.fc(z)


# ============================================================
# STUDENT MLP
# ============================================================

class StudentMLP(nn.Module):
    def __init__(self, in_dim, C, h=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h),
            nn.ReLU(),
            nn.Linear(h, C)
        )
    def forward(self, z):
        return self.net(z)


# ============================================================
# LOADING CLIENT HEADS
# ============================================================

def load_client_heads(dataset, alpha):
    root = f"./oneshot_bench/{dataset}/{alpha}/clients/DenseCB_fair"
    heads = []
    for cid in range(100):
        p = os.path.join(root, f"client_{cid:02d}", "head.pt")
        if not os.path.isfile(p):
            break
        payload = torch.load(p, map_location="cpu")
        C = payload["C"]
        head = LinearHead(K_RP, C)
        head.load_state_dict(payload["state_dict"])
        head.to(DEVICE)
        head.eval()
        heads.append(head)
    print(f"[INFO] Loaded {len(heads)} heads.")
    return heads


# ============================================================
# TEACHER LOGITS (ensemble)
# ============================================================

@torch.no_grad()
def ensemble_logits(heads, Z):
    Z = Z.to(DEVICE)
    C = heads[0].fc.out_features
    res = torch.zeros(Z.size(0), C, device=DEVICE)
    for h in heads:
        res += h(Z)
    return res.cpu()


# ============================================================
# EVAL HELPERS
# ============================================================

@torch.no_grad()
def eval_model(model, Z, y):
    dl = DataLoader(TensorDataset(Z, y), batch_size=128, shuffle=False)
    correct = total = 0
    for zb, yb in dl:
        zb, yb = zb.to(DEVICE), yb.to(DEVICE)
        pred = model(zb).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return 100 * correct / total


@torch.no_grad()
def eval_ensemble(s1, s2, Z, y):
    dl = DataLoader(TensorDataset(Z, y), batch_size=128, shuffle=False)
    correct = total = 0
    for zb, yb in dl:
        zb, yb = zb.to(DEVICE), yb.to(DEVICE)
        pred = (s1(zb) + s2(zb)).argmax(1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return 100 * correct / total


# ============================================================
# DENSE STUDENT
# ============================================================

def train_dense(Z, teacher_logits, C):
    T = 2.0
    student = StudentMLP(K_RP, C).to(DEVICE)

    opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=WD)
    kd_loss = nn.KLDivLoss(reduction="batchmean")

    with torch.no_grad():
        soft = F.softmax(teacher_logits / T, dim=1)

    dl = DataLoader(TensorDataset(Z, torch.arange(Z.size(0))), batch_size=256, shuffle=True)

    for ep in range(1, EPOCHS_DENSE + 1):
        student.train()
        for zb, idx in dl:
            zb = zb.to(DEVICE)
            pt = soft[idx].to(DEVICE)

            logits = student(zb) / T
            loss = kd_loss(F.log_softmax(logits, dim=1), pt) * (T*T)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if ep % 10 == 0:
            print(f"[DENSE] epoch {ep}/{EPOCHS_DENSE}")

    return student


# ============================================================
# CO-BOOST (S1 + S2)
# ============================================================

def train_cobo(Z, teacher_logits, C):
    T = 2.0
    kd_loss = nn.KLDivLoss(reduction="batchmean")

    with torch.no_grad():
        soft_all = F.softmax(teacher_logits / T, dim=1)
        hard_teacher = teacher_logits.argmax(1)

    # -------- Student 1 ----------
    s1 = StudentMLP(K_RP, C).to(DEVICE)
    opt1 = torch.optim.AdamW(s1.parameters(), lr=LR, weight_decay=WD)

    dl_all = DataLoader(TensorDataset(Z, torch.arange(Z.size(0))), batch_size=256, shuffle=True)

    for ep in range(1, EPOCHS_S1 + 1):
        s1.train()
        for zb, idx in dl_all:
            zb = zb.to(DEVICE)
            pt = soft_all[idx].to(DEVICE)

            logits = s1(zb) / T
            loss = kd_loss(F.log_softmax(logits, dim=1), pt) * (T*T)

            opt1.zero_grad()
            loss.backward()
            opt1.step()

        if ep % 10 == 0:
            print(f"[COBOOST S1] epoch {ep}/{EPOCHS_S1}")

    # -------- Identifica mismatches ----------
    with torch.no_grad():
        preds = []
        for (zb,) in DataLoader(TensorDataset(Z), batch_size=256):
            preds.append(s1(zb.to(DEVICE)).argmax(1).cpu())
        preds = torch.cat(preds)

    mismatch_idx = (preds != hard_teacher).nonzero().flatten()

    if mismatch_idx.numel() == 0:
        print("[COBOOST] No mismatches → S2 = S1")
        return s1, s1

    # -------- Student 2 ----------
    Zm = Z[mismatch_idx]
    soft_m = soft_all[mismatch_idx]

    s2 = StudentMLP(K_RP, C).to(DEVICE)
    opt2 = torch.optim.AdamW(s2.parameters(), lr=LR, weight_decay=WD)

    dl_m = DataLoader(TensorDataset(Zm, torch.arange(Zm.size(0))), batch_size=256, shuffle=True)

    for ep in range(1, EPOCHS_S2 + 1):
        s2.train()
        for zb, idx in dl_m:
            zb = zb.to(DEVICE)
            pt = soft_m[idx].to(DEVICE)

            logits = s2(zb) / T
            loss = kd_loss(F.log_softmax(logits, dim=1), pt) * (T*T)

            opt2.zero_grad()
            loss.backward()
            opt2.step()

        if ep % 10 == 0:
            print(f"[COBOOST S2] epoch {ep}/{EPOCHS_S2}")

    return s1, s2


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(SEED)

    for ds in DATASETS:
        print(f"\n===============================")
        print(f"### DATASET: {ds}")
        print("===============================")

        # ---- load dataset
        X_train, y_train, X_test, y_test, class_names, C = load_splits(ds)
        y_test = torch.tensor(y_test, dtype=torch.long)

        # ---- encoder + RP
        tok, enc = build_encoder()
        R = rp_matrix(enc.config.hidden_size, K_RP, RP_SEED).to(DEVICE)

        print("[ENC] encoding test set...")
        Z_test = encode_many(tok, enc, X_test, R)

        # ---- concat for students
        # (students are trained on Z_test only → strictly data-free)
        Z_all = Z_test.clone()

        for alpha in ALPHAS:
            print(f"\n--- α = {alpha} ---")

            heads = load_client_heads(ds.upper(), alpha)
            if len(heads) == 0:
                print("No client heads found.")
                continue

            # ----- 1) Ensemble Teacher -----
            teacher_logits = ensemble_logits(heads, Z_all)
            ense_acc = (teacher_logits.argmax(1) == y_test).float().mean().item() * 100
            print(f"[ENSEMBLE] test acc = {ense_acc:.2f}%")

            # ----- 2) Dense Student -----
            dense = train_dense(Z_all, teacher_logits, C)
            dense_acc = eval_model(dense, Z_test, y_test)
            print(f"[DENSE] test acc = {dense_acc:.2f}%")

            # ----- 3) Co-Boost Students -----
            s1, s2 = train_cobo(Z_all, teacher_logits, C)
            s1_acc = eval_model(s1, Z_test, y_test)
            s2_acc = eval_model(s2, Z_test, y_test)
            ens2_acc = eval_ensemble(s1, s2, Z_test, y_test)

            print(f"[COBOOST S1] test acc = {s1_acc:.2f}%")
            print(f"[COBOOST S2] test acc = {s2_acc:.2f}%")
            print(f"[COBOOST ENS] test acc = {ens2_acc:.2f}%")


if __name__ == "__main__":
    main()
