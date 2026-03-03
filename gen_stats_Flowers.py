#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==============================================================
GH-OFL — CLIENT STATS GENERATOR
Dataset : Oxford Flowers-102
Space   : X-space (NO RP)
==============================================================

Generates per-client statistics:

A_per_class_x      [C,d]
SUMSQ_per_class_x  [C,d]
S_per_class_x      [C,d,d]
N_per_class        [C]
B_global_x         [d,d]

Compatible with:
GH-OFL SERVER X-space pipeline
"""

import os
import random
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import build_backbone


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------
# Transform
# --------------------------------------------------
IMAGENET_MEAN = (0.485,0.456,0.406)
IMAGENET_STD  = (0.229,0.224,0.225)

def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# --------------------------------------------------
# Dirichlet partition
# --------------------------------------------------
def dirichlet_split(labels, num_clients, alpha, seed):

    rng = np.random.default_rng(seed)

    labels = np.array(labels)
    classes = np.unique(labels)

    client_indices = [[] for _ in range(num_clients)]

    for c in classes:
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)

        proportions = rng.dirichlet(
            alpha * np.ones(num_clients)
        )

        splits = (
            np.cumsum(proportions) * len(idx)
        ).astype(int)[:-1]

        parts = np.split(idx, splits)

        for i in range(num_clients):
            client_indices[i].extend(parts[i])

    return client_indices


# --------------------------------------------------
# Stats computation
# --------------------------------------------------
@torch.no_grad()
def compute_stats(model, loader, num_classes, device):

    d = model.feature_dim

    A = torch.zeros(num_classes, d, dtype=torch.float64)
    SUMSQ = torch.zeros_like(A)
    S = torch.zeros(num_classes, d, d, dtype=torch.float64)
    N = torch.zeros(num_classes, dtype=torch.long)
    B = torch.zeros(d, d, dtype=torch.float64)

    for x,y in loader:

        x = x.to(device, non_blocking=True)
        y = y.to(device)

        feats = model(x).double().cpu()

        for i in range(feats.size(0)):
            c = int(y[i])

            f = feats[i]

            A[c] += f
            SUMSQ[c] += f * f
            S[c] += torch.outer(f,f)
            N[c] += 1
            B += torch.outer(f,f)

    return dict(
        A_per_class_x=A,
        SUMSQ_per_class_x=SUMSQ,
        S_per_class_x=S,
        N_per_class=N,
        B_global_x=B
    )


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--num_clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)

    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    tfm = build_transform()

    trainset = datasets.Flowers102(
        root="./data",
        split="train",
        download=True,
        transform=tfm
    )

    labels = trainset._labels

    num_classes = 102

    # --------------------------------------------------
    # Backbone
    # --------------------------------------------------
    model, feat_dim, weights_tag = build_backbone(
        "resnet18",
        device
    )

    model.eval()
    model.feature_dim = feat_dim

    # --------------------------------------------------
    # Dirichlet clients
    # --------------------------------------------------
    client_indices = dirichlet_split(
        labels,
        args.num_clients,
        args.alpha,
        args.seed
    )

    alpha_tag = str(args.alpha).replace(".","p")

    save_root = (
        f"./client_stats_X/FLOWERS102/"
        f"resnet18-{weights_tag}_TRAIN_A{alpha_tag}_X{feat_dim}"
    )

    os.makedirs(save_root, exist_ok=True)

    print("\nGenerating Flowers102 stats")
    print("save_dir:", save_root)

    # --------------------------------------------------
    # Clients
    # --------------------------------------------------
    for cid, idxs in enumerate(client_indices):

        subset = Subset(trainset, idxs)

        loader = DataLoader(
            subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        stats = compute_stats(
            model,
            loader,
            num_classes,
            device
        )

        stats["meta"] = dict(
            dataset="flowers102",
            backbone="resnet18",
            weights_tag=weights_tag,
            feature_dim=feat_dim,
            alpha=args.alpha,
            client_id=cid
        )

        torch.save(
            stats,
            os.path.join(
                save_root,
                f"client_{cid:02d}.pt"
            )
        )

        print(f"client {cid:02d} done")

    print("\n✅ DONE")


if __name__ == "__main__":
    main()