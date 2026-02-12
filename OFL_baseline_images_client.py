#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client-side OFL baselines (Ensemble, Dense, Co-Boost) for SVHN
- Backbone: ResNet-18 pretrained on ImageNet
- Embeddings: 512-d from global average pooling layer
- Dirichlet client splits: α ∈ {0.05, 0.10, 0.50}
- Saves per-client linear heads in:
    ./oneshot_bench_svhn/{ALPHA}/client_xx/head.pt
"""

import os
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms, models


# ===============================================================
# CONFIG
# ===============================================================
ALPHAS = [0.05, 0.10, 0.50]
NUM_CLIENTS = 10
SEED = 42

BATCH = 128
LR = 5e-3
WD = 1e-4
EPOCHS = 30

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_ROOT = "./oneshot_bench_svhn/{ALPHA}/client_{CID}"

torch.set_grad_enabled(True)


# ===============================================================
# UTILITIES
# ===============================================================
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def dirichlet_split(labels, C, K, alpha):
    labels = torch.tensor(labels)
    splits = [[] for _ in range(K)]

    for c in range(C):
        idx = torch.where(labels == c)[0]
        if len(idx) == 0:
            continue

        idx = idx[torch.randperm(len(idx))]
        dist = torch.distributions.Dirichlet(torch.full((K,), float(alpha))).sample()
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


# ===============================================================
# SVHN WRAPPER (robust version)
# ===============================================================
class SVHNWrap(Dataset):
    def __init__(self, ds, transform):
        self.ds = ds
        self.T = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        # SVHN gives CHW arrays, sometimes not uint8 → enforce robust conversion
        x = self.ds.data[i].transpose(1, 2, 0)
        x = np.ascontiguousarray(x).astype(np.uint8)
        x = Image.fromarray(x)

        y = int(self.ds.labels[i])
        return self.T(x), y


# ===============================================================
# BACKBONE (ResNet18 pretrained)
# ===============================================================
def build_backbone():
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()    # 512-d embeddings
    net.eval().to(DEVICE)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


@torch.no_grad()
def extract_embeddings(net, loader):
    outs = []
    for x, _ in loader:
        x = x.to(DEVICE)
        z = net(x).cpu()
        outs.append(z)
    return torch.cat(outs)


# ===============================================================
# LINEAR HEAD
# ===============================================================
class LinearHead(nn.Module):
    def __init__(self, d, C):
        super().__init__()
        self.fc = nn.Linear(d, C)

    def forward(self, x):
        return self.fc(x)


# ===============================================================
# TRAIN HEAD (Dense-style student)
# ===============================================================
def train_head(Zc, yc, Z_test, y_test, C, cid):
    if Zc.size(0) == 0:
        print(f"[CLIENT {cid:02d}] empty dataset → skip")
        return None

    ds = TensorDataset(Zc, yc)
    dl = DataLoader(ds, batch_size=64, shuffle=True)

    head = LinearHead(Zc.size(1), C).to(DEVICE)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, EPOCHS + 1):
        head.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = ce(head(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    # quick sanity check accuracy
    with torch.no_grad():
        head.eval()
        pred = head(Z_test.to(DEVICE)).argmax(1)
        acc = (pred.cpu() == y_test).float().mean().item() * 100
        print(f"[CLIENT {cid:02d}] test-acc ~ {acc:.2f}%")

    return head


# ===============================================================
# MAIN
# ===============================================================
def main():
    set_seed(SEED)

    print("\n========== DATASET SVHN ==========")

    # Good ImageNet-pretrained transform
    Timg = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Load raw SVHN
    train_raw = datasets.SVHN("./data", split="train", download=True)
    test_raw  = datasets.SVHN("./data", split="test", download=True)

    train_ds = SVHNWrap(train_raw, Timg)
    test_ds  = SVHNWrap(test_raw,  Timg)

    dl_train = DataLoader(train_ds, batch_size=BATCH, shuffle=False)
    dl_test  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

    C = 10
    y_train = torch.tensor(train_raw.labels)
    y_test  = torch.tensor(test_raw.labels)

    print("[ENC] extracting embeddings...")
    net = build_backbone()
    Z_train = extract_embeddings(net, dl_train)
    Z_test  = extract_embeddings(net, dl_test)

    # For each alpha split
    for alpha in ALPHAS:
        print(f"\n[ALPHA={alpha}] splitting clients...")
        splits = dirichlet_split(y_train, C, NUM_CLIENTS, alpha)

        for cid, idxs in enumerate(splits):
            Zc = Z_train[idxs]
            yc = y_train[idxs]

            head = train_head(Zc, yc, Z_test, y_test, C, cid)
            if head is None:
                continue

            out_dir = OUT_ROOT.format(ALPHA=alpha, CID=f"{cid:02d}")
            os.makedirs(out_dir, exist_ok=True)

            torch.save(
                {"state_dict": head.state_dict(), "C": C},
                f"{out_dir}/head.pt"
            )

    print("\nDone.")


if __name__ == "__main__":
    main()
