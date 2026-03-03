#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
===========================================================
GH-OFL CLIENT — UNIVERSAL FGVC STATS GENERATOR
===========================================================

NO YAML.
Server controls experiments.

Supported datasets:
    flowers102
    cub200
    aircraft
    domainnet

Outputs:
    ./client_stats_X/{DATASET}/
"""

import os
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import build_backbone


# --------------------------------------------------
# Repro
# --------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------
# Transform
# --------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


# --------------------------------------------------
# CUB200
# --------------------------------------------------
class CUB200(datasets.ImageFolder):
    def __init__(self, root):
        super().__init__(
            os.path.join(root, "CUB_200_2011", "images"),
            transform=transform
        )


# --------------------------------------------------
# Dataset Loader
# --------------------------------------------------
def load_dataset(name, root):

    name = name.lower()

    if name == "flowers102":
        ds = datasets.Flowers102(
            root=root,
            split="train",
            download=True,
            transform=transform
        )
        C = 102

    elif name == "aircraft":
        ds = datasets.FGVCAircraft(
            root=root,
            split="trainval",
            download=True,
            transform=transform
        )
        C = 100

    elif name == "cub200":
        ds = CUB200(root)
        C = 200

    elif name == "domainnet":
        ds = datasets.ImageFolder(
            os.path.join(root, "domainnet", "real"),
            transform=transform
        )
        C = len(ds.classes)

    else:
        raise ValueError(name)

    return ds, C


# --------------------------------------------------
# Dirichlet Split
# --------------------------------------------------
def dirichlet_partition(labels, num_clients, alpha):

    labels = np.array(labels)
    classes = np.unique(labels)

    client_idx = [[] for _ in range(num_clients)]

    for c in classes:

        idx = np.where(labels == c)[0]
        np.random.shuffle(idx)

        proportions = np.random.dirichlet(
            alpha * np.ones(num_clients)
        )

        cuts = (np.cumsum(proportions) * len(idx)).astype(int)[:-1]
        splits = np.split(idx, cuts)

        for i in range(num_clients):
            client_idx[i].extend(splits[i])

    return client_idx


# --------------------------------------------------
# Stats computation
# --------------------------------------------------
@torch.no_grad()
def compute_stats(loader, fe, C, d, device):

    A = torch.zeros(C, d, dtype=torch.float64)
    SUMSQ = torch.zeros(C, d, dtype=torch.float64)
    N = torch.zeros(C, dtype=torch.long)
    B = torch.zeros(d, d, dtype=torch.float64)
    S = torch.zeros(C, d, d, dtype=torch.float64)

    for x, y in tqdm(loader):

        x = x.to(device)
        feats = fe(x).cpu().to(torch.float64)

        for f, label in zip(feats, y):

            c = int(label)

            A[c] += f
            SUMSQ[c] += f * f
            B += torch.outer(f, f)
            S[c] += torch.outer(f, f)
            N[c] += 1

    return A, SUMSQ, N, B, S


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", required=True)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--data_root", default="./data")

    args = parser.parse_args()

    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset, C = load_dataset(args.dataset, args.data_root)

    labels = [y for _, y in dataset]

    splits = dirichlet_partition(
        labels,
        args.clients,
        args.alpha
    )

    fe, d, weights = build_backbone(
        args.backbone,
        device
    )

    alpha_tag = str(args.alpha).replace(".", "p")

    save_root = (
        f"./client_stats_X/"
        f"{args.dataset.upper()}/"
        f"{args.backbone}-{weights}"
        f"_TRAIN_A{alpha_tag}_X{d}"
    )

    os.makedirs(save_root, exist_ok=True)

    print("\nSaving stats →", save_root)

    for cid, idx in enumerate(splits):

        subset = Subset(dataset, idx)

        loader = DataLoader(
            subset,
            batch_size=128,
            shuffle=False,
            num_workers=4
        )

        A, SUMSQ, N, B, S = compute_stats(
            loader, fe, C, d, device
        )

        payload = dict(
            A_per_class_x=A,
            SUMSQ_per_class_x=SUMSQ,
            N_per_class=N,
            B_global_x=B,
            S_per_class_x=S,
            meta=dict(
                backbone=args.backbone,
                weights_tag=weights,
                feature_dim=d
            )
        )

        torch.save(
            payload,
            os.path.join(save_root, f"client_{cid:02d}.pt")
        )

        print(f"client {cid} done")

    print("\n✅ DONE")


if __name__ == "__main__":
    main()