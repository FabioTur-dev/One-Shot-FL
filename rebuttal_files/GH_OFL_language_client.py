#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GH-OFL — Client-side (NLP)
Datasets: SST2, AG_NEWS, DBPEDIA_14, BANKING77, CLINC150
Backbone: distilbert-base-uncased (CLS embedding, encoder frozen)

Output stats (RP space z):
  - A_per_class_z       [C,k] float64   (sum z per class)
  - SUMSQ_per_class_z   [C,k] float64   (sum element-wise z⊙z per class)
  - N_per_class         [C]   int64
  - B_global_z          [k,k] float64   (sum z^T z over all samples)

Salvataggio (per α scelto):
  ./oneshot_bench/{DATASET}/{ALPHA}/clients/FedCGS/client_XX/client.pt
"""

import os
import math
import time
import random
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# =========================
# Config
# =========================
DATASETS = ["SST2", "AG_NEWS", "DBPEDIA_14", "BANKING77", "CLINC150"]
ALPHA = 0.5          # Dirichlet α (solo per lo split client, non per le stats globali)
NUM_CLIENTS = 10
SEED = 42

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 128
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = torch.cuda.is_available()

# Random Projection (GH-OFL)
RP_SEED = 12345
K_RP = 256  # dimensione k di z = x @ R

# Cartella output (coerente col server GH-OFL)
ROOT_TPL = "./oneshot_bench/{DATASET}/{ALPHA}/clients/FedCGS"

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
torch.set_grad_enabled(False)


# =========================
# Meta info
# =========================
@dataclass
class MetaInfo:
    dataset_name: str
    num_clients: int
    dirichlet_alpha: float
    feature_dim: int            # d (hidden size DistilBERT)
    rp_dim: int                 # k (RP)
    rp_seed: int
    backbone: str               # "distilbert"
    weights_tag: str            # "distilbert-base-uncased"
    normalization_mean: Tuple[float, ...]  # non usati in NLP
    normalization_std: Tuple[float, ...]   # non usati in NLP
    class_names: Tuple[str, ...]
    has_sumsq_per_class: bool


# =========================
# Utils
# =========================
def set_seed(s: int):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def build_rp_matrix(d: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.randn(d, k, generator=g) / math.sqrt(d)  # JL scaling
    return R  # [d,k] on CPU


def dirichlet_split(labels: List[int], num_classes: int, num_clients: int,
                    alpha: float, seed: int) -> List[List[int]]:
    # stesso algoritmo che usi nelle baseline Dense/Co-Boost
    set_seed(seed)
    labels_t = torch.tensor(labels, dtype=torch.long)
    per_client: List[List[int]] = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx = torch.where(labels_t == c)[0]
        if idx.numel() == 0:
            continue
        idx = idx[torch.randperm(len(idx))]
        dist = torch.distributions.Dirichlet(
            torch.full((num_clients,), float(alpha))
        ).sample()
        counts = (dist * len(idx)).round().to(torch.long).tolist()

        diff = len(idx) - sum(counts)
        for k in range(abs(diff)):
            counts[k % num_clients] += 1 if diff > 0 else -1
        counts = [max(0, int(x)) for x in counts]

        while sum(counts) > len(idx):
            for k in range(num_clients):
                if sum(counts) <= len(idx):
                    break
                if counts[k] > 0:
                    counts[k] -= 1
        while sum(counts) < len(idx):
            counts[sum(counts) % num_clients] += 1

        s = 0
        for i in range(num_clients):
            e = s + counts[i]
            if e > s:
                per_client[i].extend(idx[s:e].tolist())
            s = e

    for i in range(num_clients):
        rng = random.Random(SEED + i)
        rng.shuffle(per_client[i])
    return per_client


# =========================
# Dataset wrapper
# =========================
class TextClassDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = list(texts)
        self.labels = list(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], int(self.labels[i])


# =========================
# Encoder DistilBERT (frozen)
# =========================
def build_encoder(model_name=MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name)
    enc.eval().to(DEVICE)
    return tok, enc


def batch_encode(tok, texts):
    return tok(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )


@torch.no_grad()
def extract_cls(enc: AutoModel, batch_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    out = enc(**batch_inputs)
    return out.last_hidden_state[:, 0, :]  # [B, d]


# =========================
# Accumulo statistiche (RP z)
# =========================
@torch.no_grad()
def accumulate_stats_z(dataset: Dataset, tok, enc, R: torch.Tensor, num_classes: int) -> Dict[str, torch.Tensor]:
    k = R.shape[1]
    A = torch.zeros(num_classes, k, dtype=torch.float64, device=DEVICE)
    SUMSQ = torch.zeros(num_classes, k, dtype=torch.float64, device=DEVICE)
    N = torch.zeros(num_classes, dtype=torch.long, device=DEVICE)
    B = torch.zeros(k, k, dtype=torch.float64, device=DEVICE)

    dl = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )

    total = 0
    Rt = R.to(DEVICE, dtype=torch.float32)

    for texts, y in dl:
        t = batch_encode(tok, list(texts))
        t = {k: v.to(DEVICE, non_blocking=True) for k, v in t.items()}
        y = torch.tensor(y, dtype=torch.long, device=DEVICE)

        x = extract_cls(enc, t).to(torch.float32)    # [B,d]
        z = x @ Rt                                   # [B,k]

        B += z.t().mm(z).to(torch.float64)
        total += z.size(0)

        for cls in y.unique():
            cls_i = int(cls.item())
            mask = (y == cls_i)
            if mask.any():
                zc = z[mask].to(torch.float64)
                A[cls_i] += zc.sum(dim=0)
                SUMSQ[cls_i] += (zc * zc).sum(dim=0)
                N[cls_i] += int(mask.sum().item())

    return {
        "A_per_class_z": A.detach().cpu(),
        "SUMSQ_per_class_z": SUMSQ.detach().cpu(),
        "N_per_class": N.detach().cpu(),
        "B_global_z": B.detach().cpu(),
        "num_samples": int(total),
    }


# =========================
# Helpers dataset HF / CSV
# =========================
def load_texts_labels(dataset_name: str):
    name = dataset_name.upper()

    # --- dataset GLUE / HF già usati ---
    if name == "SST2":
        ds = load_dataset("glue", "sst2")
        train = ds["train"]        # ~67k
        texts = train["sentence"]
        labels = train["label"]
        class_names = ("negative", "positive")
        C = 2

    elif name == "AG_NEWS":
        ds = load_dataset("ag_news")
        train = ds["train"]        # ~120k
        texts = train["text"]
        labels = train["label"]    # 0..3
        class_names = ("World", "Sports", "Business", "Sci/Tech")
        C = 4

    elif name == "DBPEDIA_14":
        ds = load_dataset("dbpedia_14")
        train = ds["train"]        # ~560k
        texts = train["content"]
        labels = train["label"]    # 0..13
        class_names = tuple(str(i) for i in range(14))
        C = 14

    # --- nuovi: BANKING77 + CLINC150 (stesso schema delle baseline) ---
    elif name == "BANKING77":
        data_files = {
            "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
            "test":  "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
        }
        dsd = load_dataset("csv", data_files=data_files)
        train_ds = dsd["train"]
        test_ds  = dsd["test"]

        cols = train_ds.column_names
        text_col = "text" if "text" in cols else cols[0]
        label_candidates = [c for c in cols if c != text_col]
        if not label_candidates:
            raise ValueError(f"[BANKING77] Nessuna colonna label trovata. Colonne: {cols}")
        label_col = label_candidates[0]

        all_labels_str = list(train_ds[label_col]) + list(test_ds[label_col])
        class_names = sorted(set(all_labels_str))
        label2id = {lab: i for i, lab in enumerate(class_names)}
        C = len(class_names)

        texts = list(train_ds[text_col])                     # SOLO train per le stats GH-OFL
        labels = [label2id[l] for l in train_ds[label_col]]

    elif name in ("CLINC150", "CLINIC150"):
        data_files = "https://huggingface.co/datasets/contemmcm/clinc150/resolve/main/data_full.csv"
        dsd = load_dataset("csv", data_files=data_files)
        full = dsd["train"]

        cols = full.column_names
        if "text" in cols:
            text_col = "text"
        else:
            text_like = [c for c in cols if "text" in c.lower() or "utterance" in c.lower()]
            text_col = text_like[0] if text_like else cols[0]

        if "intent" in cols:
            intent_col = "intent"
        else:
            intent_like = [c for c in cols if "intent" in c.lower() or "label" in c.lower()]
            if not intent_like:
                raise ValueError(f"[CLINC150] Nessuna colonna intent trovata. Colonne: {cols}")
            intent_col = intent_like[0]

        if "split" in cols:
            split_col = "split"
        else:
            split_like = [c for c in cols if "split" in c.lower() or "partition" in c.lower()]
            if not split_like:
                raise ValueError(f"[CLINC150] Nessuna colonna split trovata. Colonne: {cols}")
            split_col = split_like[0]

        all_intents = list(full[intent_col])
        class_names = sorted(set(all_intents))
        label2id = {lab: i for i, lab in enumerate(class_names)}
        C = len(class_names)

        texts = []
        labels = []
        for txt, split, intent in zip(full[text_col], full[split_col], full[intent_col]):
            if split == "train":          # SOLO split train per GH-OFL
                texts.append(txt)
                labels.append(label2id[intent])

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return list(texts), list(labels), class_names, C


# =========================
# Main
# =========================
def main():
    set_seed(SEED)
    tok, enc = build_encoder(MODEL_NAME)
    d = enc.config.hidden_size  # 768 for DistilBERT
    R = build_rp_matrix(d, K_RP, RP_SEED)  # cpu

    for dset in DATASETS:
        print(f"\n[DATASET {dset}] === GH-OFL client stats ===")
        texts, labels, class_names, C = load_texts_labels(dset)
        print(f"[INFO] {dset}: |train|={len(texts)}, C={C}")

        splits = dirichlet_split(labels, num_classes=C, num_clients=NUM_CLIENTS,
                                 alpha=ALPHA, seed=SEED)

        out_dir = ROOT_TPL.format(DATASET=dset.upper(), ALPHA=str(ALPHA))
        os.makedirs(out_dir, exist_ok=True)

        meta = MetaInfo(
            dataset_name=dset.upper(),
            num_clients=NUM_CLIENTS,
            dirichlet_alpha=ALPHA,
            feature_dim=d,
            rp_dim=K_RP,
            rp_seed=RP_SEED,
            backbone="distilbert",
            weights_tag=MODEL_NAME,
            normalization_mean=tuple(),  # not used in NLP
            normalization_std=tuple(),   # not used in NLP
            class_names=tuple(class_names),
            has_sumsq_per_class=True,
        )

        for i, idxs in enumerate(splits):
            subset = TextClassDataset([texts[j] for j in idxs],
                                      [labels[j] for j in idxs])

            stats = accumulate_stats_z(subset, tok, enc, R, num_classes=C)

            payload = {
                "meta": asdict(meta),
                "client_id": i,
                "A_per_class_z": stats["A_per_class_z"],
                "SUMSQ_per_class_z": stats["SUMSQ_per_class_z"],
                "N_per_class": stats["N_per_class"],
                "B_global_z": stats["B_global_z"],
                "num_samples": stats["num_samples"],
                "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
            }

            client_dir = os.path.join(out_dir, f"client_{i:02d}")
            os.makedirs(client_dir, exist_ok=True)
            out_path = os.path.join(client_dir, "client.pt")
            torch.save(payload, out_path)

            print(f"[OK] {dset}/{MODEL_NAME} | client {i:02d} -> {out_path} | "
                  f"samples={payload['num_samples']}")

    print("\n[DONE] GH-OFL NLP client stats saved under:", os.path.abspath("../oneshot_bench"))


if __name__ == "__main__":
    main()

