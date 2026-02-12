#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
language_client.py  (Dense / Co-Boost baselines — client-side POTENZIATO)

Client-side per Dense / Co-Boosting su intent NLP:
- Datasets:
    * BANKING77
    * CLINC150
    * AG_NEWS
    * DBPEDIA_14
    * SST2  (da GLUE, split: train / validation)
- Backbone: distilbert-base-uncased (CLS embedding, encoder frozen)
- RP: k=256, stesso schema GH-OFL

Pipeline:
  1) Per ogni dataset:
       - carica train/test con mapping unico delle classi
       - calcola z_train, z_test con DistilBERT CLS + RP
  2) Per ogni alpha in ALPHAS:
       - Split Dirichlet α di z_train in NUM_CLIENTS client
       - Per ogni client:
           - logga distribuzione delle classi
           - allena LinearHead(K_RP, C) in RP space
           - logga loss e train-acc per epoca
           - valuta su train locale e test globale
           - salva head:
             ./oneshot_bench/{DATASET}/{ALPHA}/clients/DenseCB/client_XX/head.pt
"""

import os
import math
import time
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# =========================
# Config
# =========================
DATASETS = ["BANKING77", "CLINC150", "AG_NEWS", "DBPEDIA_14", "SST2"]

# Nuovo: più alpha
ALPHAS = [0.5, 0.01]
# Per compatibilità con il server che importa ALPHA:
ALPHA = 0.5  # alpha "di default" (es. per il server-side attuale)

NUM_CLIENTS = 10
SEED = 42

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 64
NUM_WORKERS = 0  # lavoriamo in RP space, no bisogno di worker extra
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = False

RP_SEED = 12345
K_RP = 256

# training più aggressivo sui client
LR = 5e-3
WD = 1e-4
EPOCHS = 40      # aumentato rispetto a prima
MAX_NORM = 5.0
LS = 0.0         # niente label smoothing

ROOT_TPL = "./oneshot_bench/{DATASET}/{ALPHA}/clients/DenseCB"

torch.backends.cudnn.benchmark = True
torch.set_grad_enabled(True)


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
    class_names: Tuple[str, ...]
    num_classes: int


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
    R = torch.randn(d, k, generator=g) / math.sqrt(d)
    return R  # [d,k] su CPU


def dirichlet_split(labels: List[int], num_classes: int, num_clients: int,
                    alpha: float, seed: int) -> List[List[int]]:
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
# Encoder DistilBERT (frozen)
# =========================
def build_encoder(model_name=MODEL_NAME):
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name)
    enc.eval().to(DEVICE)
    for p in enc.parameters():
        p.requires_grad_(False)
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
    return out.last_hidden_state[:, 0, :]  # [B,d]


# =========================
# Dataset loaders (train + test, mapping unico)
# =========================
def load_banking77_splits():
    data_files = {
        "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
        "test":  "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
    }
    dsd = load_dataset("csv", data_files=data_files)
    train_ds = dsd["train"]
    test_ds = dsd["test"]

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

    texts_train = list(train_ds[text_col])
    labels_train = [label2id[l] for l in train_ds[label_col]]

    texts_test = list(test_ds[text_col])
    labels_test = [label2id[l] for l in test_ds[label_col]]

    return texts_train, labels_train, texts_test, labels_test, class_names, C


def load_clinc150_splits():
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

    texts_train = []
    labels_train = []
    texts_test = []
    labels_test = []

    for text, split, intent in zip(full[text_col], full[split_col], full[intent_col]):
        lid = label2id[intent]
        if split == "train":
            texts_train.append(text)
            labels_train.append(lid)
        elif split == "test":
            texts_test.append(text)
            labels_test.append(lid)

    return texts_train, labels_train, texts_test, labels_test, class_names, C


def load_agnews_splits():
    """
    AG_NEWS da HuggingFace: load_dataset("ag_news")
    Colonne tipiche: "text", "label"
    """
    dsd = load_dataset("ag_news")
    train_ds = dsd["train"]
    test_ds = dsd["test"]

    cols = train_ds.column_names
    # testo
    if "text" in cols:
        text_col = "text"
    else:
        text_like = [c for c in cols if "text" in c.lower() or "title" in c.lower()]
        text_col = text_like[0] if text_like else cols[0]
    # label
    if "label" in cols:
        label_col = "label"
    else:
        lab_like = [c for c in cols if "label" in c.lower() or "class" in c.lower()]
        label_col = lab_like[0] if lab_like else cols[-1]

    all_labels = list(train_ds[label_col]) + list(test_ds[label_col])
    class_names = sorted(set(all_labels))
    label2id = {lab: i for i, lab in enumerate(class_names)}
    C = len(class_names)

    texts_train = list(train_ds[text_col])
    labels_train = [label2id[l] for l in train_ds[label_col]]

    texts_test = list(test_ds[text_col])
    labels_test = [label2id[l] for l in test_ds[label_col]]

    return texts_train, labels_train, texts_test, labels_test, class_names, C


def load_dbpedia14_splits():
    """
    DBPEDIA_14 da HuggingFace: load_dataset("dbpedia_14")
    Colonne tipiche: "content", "label"
    """
    dsd = load_dataset("dbpedia_14")
    train_ds = dsd["train"]
    test_ds = dsd["test"]

    cols = train_ds.column_names
    # testo
    if "content" in cols:
        text_col = "content"
    else:
        text_like = [c for c in cols if "content" in c.lower() or "text" in c.lower()]
        text_col = text_like[0] if text_like else cols[0]
    # label
    if "label" in cols:
        label_col = "label"
    else:
        lab_like = [c for c in cols if "label" in c.lower() or "class" in c.lower()]
        label_col = lab_like[0] if lab_like else cols[-1]

    all_labels = list(train_ds[label_col]) + list(test_ds[label_col])
    class_names = sorted(set(all_labels))
    label2id = {lab: i for i, lab in enumerate(class_names)}
    C = len(class_names)

    texts_train = list(train_ds[text_col])
    labels_train = [label2id[l] for l in train_ds[label_col]]

    texts_test = list(test_ds[text_col])
    labels_test = [label2id[l] for l in test_ds[label_col]]

    return texts_train, labels_train, texts_test, labels_test, class_names, C


def load_sst2_splits():
    """
    SST-2 da GLUE: load_dataset("glue", "sst2")
    Usiamo train come train, validation come test (il test ufficiale non ha label).
    Colonne: "sentence", "label"
    """
    dsd = load_dataset("glue", "sst2")
    train_ds = dsd["train"]
    val_ds = dsd["validation"]  # usato come test

    cols = train_ds.column_names
    if "sentence" in cols:
        text_col = "sentence"
    else:
        text_like = [c for c in cols if "sentence" in c.lower() or "text" in c.lower()]
        text_col = text_like[0] if text_like else cols[0]

    if "label" in cols:
        label_col = "label"
    else:
        lab_like = [c for c in cols if "label" in c.lower() or "class" in c.lower()]
        label_col = lab_like[0] if lab_like else cols[-1]

    all_labels = list(train_ds[label_col]) + list(val_ds[label_col])
    class_names = sorted(set(all_labels))
    label2id = {lab: i for i, lab in enumerate(class_names)}
    C = len(class_names)

    texts_train = list(train_ds[text_col])
    labels_train = [label2id[l] for l in train_ds[label_col]]

    texts_test = list(val_ds[text_col])
    labels_test = [label2id[l] for l in val_ds[label_col]]

    return texts_train, labels_train, texts_test, labels_test, class_names, C


def load_splits(dataset_name: str):
    name = dataset_name.upper()
    if name == "BANKING77":
        return load_banking77_splits()
    elif name in ("CLINC150", "CLINIC150"):
        return load_clinc150_splits()
    elif name in ("AG_NEWS", "AGNEWS"):
        return load_agnews_splits()
    elif name in ("DBPEDIA_14", "DBPEDIA-14", "DBPEDIA14"):
        return load_dbpedia14_splits()
    elif name in ("SST2", "SST-2"):
        return load_sst2_splits()
    else:
        raise ValueError(f"Dataset non supportato: {dataset_name}")


# =========================
# Precompute z_train, z_test
# =========================
@torch.no_grad()
def compute_z_matrix(texts: List[str],
                     tok,
                     enc,
                     R: torch.Tensor,
                     desc: str) -> torch.Tensor:
    print(f"[ENC] Computing z for {desc} ({len(texts)} samples)...")
    dl = DataLoader(
        texts,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda batch: list(batch),
    )
    Rt = R.to(DEVICE, dtype=torch.float32)
    zs = []
    for i, batch_texts in enumerate(dl):
        t = batch_encode(tok, batch_texts)
        t = {k: v.to(DEVICE, non_blocking=True) for k, v in t.items()}
        x = extract_cls(enc, t).to(torch.float32)  # [B,d]
        z = x @ Rt                                 # [B,k]
        zs.append(z.cpu())
        if (i + 1) % 50 == 0:
            print(f"    [ENC {desc}] batch {i+1}/{len(dl)}")
    Z = torch.cat(zs, dim=0)  # [N,k]
    print(f"[ENC] Done {desc}: Z shape = {Z.shape}")
    return Z


# =========================
# Head model
# =========================
class LinearHead(torch.nn.Module):
    def __init__(self, in_dim, C):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, C)

    def forward(self, z):
        return self.fc(z)


# =========================
# Eval helper in RP space
# =========================
@torch.no_grad()
def eval_head_on_tensors(head: torch.nn.Module,
                         Z: torch.Tensor,
                         y: torch.Tensor,
                         desc: str) -> float:
    head.eval()
    dl = DataLoader(
        TensorDataset(Z, y),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )
    tot = 0
    correct = 0
    for zb, yb in dl:
        zb = zb.to(DEVICE, non_blocking=True)
        yb = yb.to(DEVICE, non_blocking=True)
        logits = head(zb)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        tot += yb.numel()
    acc = 100.0 * correct / float(tot)
    print(f"        [EVAL {desc}] acc = {acc:.2f}%  ({correct}/{tot})")
    return acc


# =========================
# Training one client head in RP space
# =========================
def train_client_head_rp(client_id: int,
                         Z_client: torch.Tensor,
                         y_client: torch.Tensor,
                         Z_test: torch.Tensor,
                         y_test: torch.Tensor,
                         C: int,
                         class_names: List[str]):
    ds_client = TensorDataset(Z_client, y_client)
    dl = DataLoader(
        ds_client,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    head = LinearHead(K_RP, C).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    ce = torch.nn.CrossEntropyLoss(label_smoothing=LS)

    # logging distribuzione classi del client
    labels_np = y_client.numpy()
    unique, counts = np.unique(labels_np, return_counts=True)
    freq = sorted(zip(unique.tolist(), counts.tolist()),
                  key=lambda x: x[1], reverse=True)
    n_classes_client = len(unique)
    print(f"[CLIENT {client_id:02d}] samples={len(Z_client)}, distinct_classes={n_classes_client}")
    print("    Top-5 classi (id, nome, count):")
    for cid, cnt in freq[:5]:
        name = class_names[cid] if cid < len(class_names) else cid
        name_str = str(name)
        print(f"        class {cid:3d} ({name_str[:30]}): {cnt}")

    # training loop
    for ep in range(1, EPOCHS + 1):
        head.train()
        tot_loss = 0.0
        seen = 0
        correct_train = 0

        for zb, yb in dl:
            zb = zb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)

            logits = head(zb)
            loss = ce(logits, yb)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), MAX_NORM)
            opt.step()

            bs = zb.size(0)
            tot_loss += loss.item() * bs
            seen += bs

            pred = logits.argmax(dim=1)
            correct_train += int((pred == yb).sum().item())

        avg_loss = tot_loss / max(1, seen)
        train_acc_ep = 100.0 * correct_train / float(seen)
        print(f"    [CLIENT {client_id:02d}] ep {ep:02d}/{EPOCHS} | loss={avg_loss:.4f} | train_acc={train_acc_ep:.2f}%")

    # Eval finale su train locale e su test globale
    train_acc = eval_head_on_tensors(head, Z_client, y_client, desc=f"CLIENT{client_id:02d}-TRAIN")
    test_acc = eval_head_on_tensors(head, Z_test, y_test, desc=f"CLIENT{client_id:02d}-TEST")

    return head, train_acc, test_acc


def main():
    set_seed(SEED)
    tok, enc = build_encoder(MODEL_NAME)
    d = enc.config.hidden_size
    R = build_rp_matrix(d, K_RP, RP_SEED)

    for dset in DATASETS:
        print(f"\n[DATASET] {dset}")
        texts_train, labels_train, texts_test, labels_test, class_names, C = load_splits(dset)
        print(f"[INFO] {dset}: {len(texts_train)} train samples, {len(texts_test)} test samples, C={C} classes")
        print(f"[INFO] log(C) = {math.log(C):.4f}")

        # precompute z_train, z_test UNA SOLA VOLTA per dataset
        Z_train = compute_z_matrix(texts_train, tok, enc, R, desc=f"{dset}-train")
        Z_test = compute_z_matrix(texts_test, tok, enc, R, desc=f"{dset}-test")
        y_train = torch.tensor(labels_train, dtype=torch.long)
        y_test = torch.tensor(labels_test, dtype=torch.long)

        # Per ogni alpha generiamo un set di client heads
        for alpha in ALPHAS:
            print(f"\n[DATASET {dset}] === ALPHA = {alpha} ===")

            # 1) split Dirichlet
            splits = dirichlet_split(
                labels_train,
                num_classes=C,
                num_clients=NUM_CLIENTS,
                alpha=alpha,
                seed=SEED,
            )

            # 2) FIX: assicuriamoci che nessun client sia completamente vuoto
            total_before = sum(len(s) for s in splits)
            assert total_before == len(labels_train), "Split ha perso/duplicato sample!"

            empty_clients = [i for i, s in enumerate(splits) if len(s) == 0]
            if empty_clients:
                print(f"[WARN] {dset} | alpha={alpha}: {len(empty_clients)} client vuoti dalla Dirichlet, riassegno 1 sample a ciascuno.")
                for cid in empty_clients:
                    # troviamo un donatore con più sample
                    donor = max(range(len(splits)), key=lambda j: len(splits[j]))
                    if len(splits[donor]) <= 1:
                        # in caso estremissimo non possiamo fare molto di meglio
                        continue
                    moved_idx = splits[donor].pop()  # togli un indice dal donatore
                    splits[cid].append(moved_idx)    # e assegnalo al client vuoto

            # sanity check: stessa cardinalità di prima
            total_after = sum(len(s) for s in splits)
            assert total_after == len(labels_train), "Dopo il fix il numero di sample è cambiato!"

            out_root = ROOT_TPL.format(DATASET=dset.upper(), ALPHA=str(alpha))
            os.makedirs(out_root, exist_ok=True)

            meta = MetaInfo(
                dataset_name=dset.upper(),
                num_clients=NUM_CLIENTS,
                dirichlet_alpha=alpha,
                feature_dim=d,
                rp_dim=K_RP,
                rp_seed=RP_SEED,
                backbone="distilbert",
                weights_tag=MODEL_NAME,
                class_names=tuple(class_names),
                num_classes=C,
            )

            all_train_acc = []
            all_test_acc = []

            for i, idxs in enumerate(splits):
                idxs = np.array(idxs, dtype=np.int64)
                if idxs.size == 0:
                    # dovrebbe essere impossibile dopo il fix, ma per sicurezza logghiamo
                    print(f"[WARN] {dset} | alpha={alpha} | client {i:02d} ancora vuoto, lo salto.")
                    continue

                Zc = Z_train[idxs].clone()
                yc = y_train[idxs].clone()
                head, tr_acc, te_acc = train_client_head_rp(
                    client_id=i,
                    Z_client=Zc,
                    y_client=yc,
                    Z_test=Z_test,
                    y_test=y_test,
                    C=C,
                    class_names=class_names,
                )
                all_train_acc.append(tr_acc)
                all_test_acc.append(te_acc)

                payload = {
                    "meta": asdict(meta),
                    "client_id": i,
                    "state_dict": head.state_dict(),
                    "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
                }

                client_dir = os.path.join(out_root, f"client_{i:02d}")
                os.makedirs(client_dir, exist_ok=True)
                out_path = os.path.join(client_dir, "head.pt")
                torch.save(payload, out_path)

                print(f"[OK] {dset}/{MODEL_NAME} | alpha={alpha} | client {i:02d} -> {out_path}")

            # riepilogo per dataset + alpha
            if all_train_acc and all_test_acc:
                mean_tr = sum(all_train_acc) / len(all_train_acc)
                mean_te = sum(all_test_acc) / len(all_test_acc)
                min_tr = min(all_train_acc)
                max_tr = max(all_train_acc)
                min_te = min(all_test_acc)
                max_te = max(all_test_acc)
                print(f"[SUMMARY {dset} | alpha={alpha}] TRAIN acc over clients: mean={mean_tr:.2f}% | min={min_tr:.2f}% | max={max_tr:.2f}%")
                print(f"[SUMMARY {dset} | alpha={alpha}] TEST  acc over clients: mean={mean_te:.2f}% | min={min_te:.2f}% | max={max_te:.2f}%")

    print("\n[DONE] Dense/CoBoost client heads saved under:", os.path.abspath("./oneshot_bench"))



if __name__ == "__main__":
    main()



