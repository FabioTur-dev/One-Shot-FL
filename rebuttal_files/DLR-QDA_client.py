#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GH-OFL — CLIENT SIDE (NO RANDOM PROJECTION)
Genera statistiche (A, N, B, S, D) per ogni client.
Backbone: ResNet-18 o backbone scelto (pretrained ImageNet).
"""

import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128


# ------------------------------------------------------------
# Backbone (no RP)
# ------------------------------------------------------------
def load_backbone(name="resnet18"):
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Unsupported backbone")

    # Remove final FC (we want 512-d features)
    model.fc = nn.Identity()

    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    return model


# ------------------------------------------------------------
# Dirichlet split
# ------------------------------------------------------------
def dirichlet_split(labels, alpha, n_clients):
    num_classes = labels.max() + 1
    class_idxs = [np.where(labels == i)[0] for i in range(num_classes)]

    client_indices = [[] for _ in range(n_clients)]
    for c in range(num_classes):
        idxs = class_idxs[c]
        np.random.shuffle(idxs)
        proportions = np.random.dirichlet(np.repeat(alpha, n_clients))
        proportions = (proportions * len(idxs)).astype(int)
        proportions[-1] = len(idxs) - proportions[:-1].sum()
        start = 0
        for i, p in enumerate(proportions):
            client_indices[i].extend(idxs[start:start+p])
            start += p

    return [np.array(ci) for ci in client_indices]


# ------------------------------------------------------------
# Feature extraction (NO RP)
# ------------------------------------------------------------
def extract_stats(model, loader, num_classes):
    d = 512  # ResNet18 embedding dim
    A = torch.zeros((num_classes, d), dtype=torch.float64)
    N = torch.zeros(num_classes, dtype=torch.int64)
    B = torch.zeros((d, d), dtype=torch.float64)
    S = torch.zeros((num_classes, d, d), dtype=torch.float64)
    D = torch.zeros((num_classes, d), dtype=torch.float64)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            feats = model(x).detach().cpu().double()
            for i in range(len(y)):
                c = int(y[i])
                v = feats[i]

                A[c] += v
                N[c] += 1
                B += torch.outer(v, v)
                S[c] += torch.outer(v, v)
                D[c] += v * v

    return {"A": A, "N": N, "B": B, "S": S, "D": D}


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=int, required=True)
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="svhn",
                        choices=["svhn", "cifar10", "cifar100"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--backbone", type=str, default="resnet18")
    args = parser.parse_args()

    # Dataset
    if args.dataset == "svhn":
        transform = transforms.ToTensor()
        train = datasets.SVHN(root="./data", split="train", download=True, transform=transform)
        labels = train.labels
        num_classes = 10
    else:
        transform = transforms.ToTensor()
        train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        labels = np.array(train.targets)
        num_classes = 10

    # Partition
    client_splits = dirichlet_split(labels, args.alpha, args.n_clients)
    idx = client_splits[args.client_id]
    subset = Subset(train, idx)

    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    print(f"[Client {args.client_id}] Loading backbone: {args.backbone}")
    model = load_backbone(args.backbone)

    print(f"[Client {args.client_id}] Extracting stats for {len(idx)} samples...")
    stats = extract_stats(model, loader, num_classes)

    os.makedirs("../client_stats", exist_ok=True)
    out = f"client_stats/client_{args.client_id}.pt"
    torch.save(stats, out)
    print(f"[Client {args.client_id}] Done. Saved to {out}")


if __name__ == "__main__":
    main()
