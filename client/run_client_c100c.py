#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=====================================================================================
GH-OFL | CLIENT | X-SPACE (NO RP) | CAMERA-READY (ICLR 2026)
=====================================================================================

Paper:
  "The Gaussian-Head OFL Family: One-Shot Federated Learning from Client Global Statistics"
  arXiv:2602.01186

What this script does (CLIENT-SIDE, TRAIN SPLIT ONLY):
  1) Loads a vision dataset (SVHN / CIFAR-10 / CIFAR-100).
  2) Creates a deterministic Dirichlet client partition (class-conditional) for a given alpha.
  3) Uses a frozen ResNet-18 ImageNet feature extractor (d=512) with ImageNet preprocessing.
  4) For each client, computes and saves *only* global statistics in x-space (no raw data, no grads):
       - A_per_class_x         [C,512]     float64   sum(x) per class
       - SUMSQ_per_class_x     [C,512]     float64   sum(x ⊙ x) per class  (diagonal moments)
       - N_per_class           [C]         int64     counts per class
       - B_global_x            [512,512]   float64   sum(x^T x) global
       - S_per_class_x         [C,512,512] float64   sum(x^T x) per class  (FULL QDA moments)

Design notes:
  - X-space only (no random projection), exactly as in your "full" SVHN/CIFAR X-space client.
  - Accumulation in float64 for stability; features extracted in float32 then cast.
  - Preprocessing matches strong server convention: Resize(224) + ImageNet Normalize.
  - Deterministic split with SEED; no hidden randomness.

Output layout (repo-relative):
  ./client_stats_X/{DATASET}/resnet18-IMAGENET1K_V1_TRAIN_A{alpha}_X512/client_XX.pt

Config:
  Use --config configs/{dataset}.yaml
  Required YAML keys:
    dataset: "svhn" | "cifar10" | "cifar100"
    num_clients: int
    dirichlet_alpha: float
    seed: int
    batch_size: int
    data_root: str
    out_root: str

Example:
  python client/run_client.py --config configs/svhn.yaml

=====================================================================================
"""

from __future__ import annotations

import os
import math
import time
import random
import argparse
import platform
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models


# -----------------------------
# Constants
# -----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

FEATURE_DIM = 512  # ResNet-18 penultimate features (fc replaced with Identity)


# -----------------------------
# Meta payload (saved in each .pt)
# -----------------------------
@dataclass
class MetaInfo:
    paper: str
    arxiv: str
    dataset_name: str
    dataset_split: str
    num_classes: int
    num_clients: int
    dirichlet_alpha: float
    seed: int

    backbone: str
    weights_tag: str
    feature_dim: int

    preprocessing: str
    image_size: int
    normalization_mean: Tuple[float, float, float]
    normalization_std: Tuple[float, float, float]

    stats: str
    notes: str


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic split + consistent feature extraction (as in your SVHN client)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Preprocessing / Backbone
# -----------------------------
def _resize_224():
    # antialias=True not available in older torchvision; keep robust.
    try:
        return transforms.Resize(224, antialias=True)
    except TypeError:
        return transforms.Resize(224)


def build_resnet18_imagenet(device: torch.device) -> Tuple[nn.Module, str]:
    """
    Frozen ResNet-18 ImageNet feature extractor, fc = Identity.
    Returns model and weights_tag.
    """
    weights_tag = "IMAGENET1K_V1"
    try:
        from torchvision.models import ResNet18_Weights
        w = ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=w)
    except Exception:
        model = models.resnet18(pretrained=True)
        weights_tag = "pretrained_fallback"

    model.fc = nn.Identity()
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    return model, weights_tag


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        _resize_224(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# -----------------------------
# Dataset helpers
# -----------------------------
def load_train_dataset(dataset_name: str, data_root: str, tfm) -> Tuple[torch.utils.data.Dataset, int]:
    dataset_name = dataset_name.lower()

    if dataset_name == "svhn":
        ds = datasets.SVHN(root=data_root, split="train", transform=tfm, download=True)
        num_classes = 10
        return ds, num_classes

    if dataset_name == "cifar10":
        ds = datasets.CIFAR10(root=data_root, train=True, transform=tfm, download=True)
        num_classes = 10
        return ds, num_classes

    if dataset_name == "cifar100":
        ds = datasets.CIFAR100(root=data_root, train=True, transform=tfm, download=True)
        num_classes = 100
        return ds, num_classes

    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_labels(ds, dataset_name: str) -> List[int]:
    dataset_name = dataset_name.lower()
    if dataset_name == "svhn":
        # SVHN labels may include 10 for digit '0' -> map with %10 (paper-coherent)
        y = np.array(ds.labels, dtype=np.int64) % 10
        return y.tolist()
    else:
        # CIFAR datasets
        y = np.array(ds.targets, dtype=np.int64)
        return y.tolist()


# -----------------------------
# Dirichlet split (class-conditional) — deterministic
# -----------------------------
def dirichlet_split(
    labels: List[int],
    num_classes: int,
    num_clients: int,
    alpha: float,
    seed: int
) -> List[List[int]]:
    """
    Class-conditional Dirichlet split:
      For each class c, distribute indices across clients with Dirichlet(alpha).

    Deterministic given seed.
    """
    set_seed(seed)

    labels_t = torch.tensor(labels, dtype=torch.long)
    per_client: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx = torch.where(labels_t == c)[0]
        idx = idx[torch.randperm(len(idx))]

        dist = torch.distributions.Dirichlet(torch.full((num_clients,), float(alpha))).sample()
        counts = (dist * len(idx)).round().to(torch.long).tolist()

        # Fix counts to sum exactly to len(idx)
        diff = len(idx) - sum(counts)
        for k in range(abs(diff)):
            counts[k % num_clients] += 1 if diff > 0 else -1

        counts = [max(0, int(x)) for x in counts]

        # Guard for any drift due to rounding fixes
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

    # Shuffle each client deterministically
    for i in range(num_clients):
        rng = random.Random(seed + 1000 + i)
        rng.shuffle(per_client[i])

    return per_client


# -----------------------------
# Stats accumulation in x-space (float64)
# -----------------------------
@torch.no_grad()
def aggregate_client_stats_x(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    dataset_name: str,
) -> Dict[str, torch.Tensor]:
    """
    Computes x-space statistics for a single client subset.
    Features extracted by model: x in R^512 (float32), accumulated in float64.
    """
    C = num_classes
    d = FEATURE_DIM

    A = torch.zeros(C, d, dtype=torch.float64, device=device)
    SUMSQ = torch.zeros(C, d, dtype=torch.float64, device=device)
    N = torch.zeros(C, dtype=torch.long, device=device)
    B = torch.zeros(d, d, dtype=torch.float64, device=device)
    S = torch.zeros(C, d, d, dtype=torch.float64, device=device)

    for imgs, y in loader:
        imgs = imgs.to(device, non_blocking=True)

        # Labels handling (SVHN: y%10)
        if dataset_name.lower() == "svhn":
            y = (y.to(device, non_blocking=True) % 10).to(torch.long)
        else:
            y = y.to(device, non_blocking=True).to(torch.long)

        x32 = model(imgs).to(torch.float32)   # [B,512]
        x64 = x32.to(torch.float64)

        # Global second moment
        B += x64.t().mm(x64)

        # Per-class moments
        for cls in y.unique():
            ci = int(cls.item())
            mask = (y == cls)
            if mask.any():
                xc = x64[mask]  # [n_c,512]
                A[ci] += xc.sum(dim=0)
                SUMSQ[ci] += (xc * xc).sum(dim=0)
                N[ci] += int(mask.sum().item())
                S[ci] += xc.t().mm(xc)

    return {
        "A_per_class_x": A.detach().cpu(),
        "SUMSQ_per_class_x": SUMSQ.detach().cpu(),
        "N_per_class": N.detach().cpu(),
        "B_global_x": B.detach().cpu(),
        "S_per_class_x": S.detach().cpu(),
    }


# -----------------------------
# Config
# -----------------------------
def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_alpha(alpha: float) -> str:
    # Pretty folder naming: 0.1 -> 0p1, 0.05 -> 0p05, 0.5 -> 0p5
    s = f"{alpha:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="GH-OFL Client | X-space statistics (no RP)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., configs/svhn.yaml)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    dataset_name: str = str(cfg["dataset"]).lower()
    base_dataset = str(cfg.get("base_dataset", dataset_name)).lower()
    num_clients: int = int(cfg["num_clients"])
    alpha: float = float(cfg["dirichlet_alpha"])
    seed: int = int(cfg.get("seed", 42))
    batch_size: int = int(cfg.get("batch_size", 512))
    data_root: str = str(cfg.get("data_root", "./data"))
    out_root: str = str(cfg.get("out_root", "./client_stats_X"))

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_win = platform.system().lower().startswith("win")
    pin_memory = torch.cuda.is_available() and (not is_win)
    num_workers = int(cfg.get("num_workers", 2 if is_win else 4))

    tfm = build_transform()
    # CIFAR-100-C uses CIFAR-100 train
    train_dataset_name = base_dataset if dataset_name == "cifar100c" else dataset_name

    ds, num_classes = load_train_dataset(train_dataset_name, data_root, tfm)
    labels = get_labels(ds, train_dataset_name)

    # Backbone
    model, weights_tag = build_resnet18_imagenet(device)

    # Output dir
    alpha_tag = format_alpha(alpha)
    out_dir = os.path.join(
        out_root,
        dataset_name.upper(),
        f"resnet18-{weights_tag}_TRAIN_A{alpha_tag}_X512"
    )
    os.makedirs(out_dir, exist_ok=True)

    # Create deterministic splits
    splits = dirichlet_split(
        labels=labels,
        num_classes=num_classes,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed
    )

    # Meta shared
    meta = MetaInfo(
        paper="The Gaussian-Head OFL Family: One-Shot Federated Learning from Client Global Statistics",
        arxiv="https://arxiv.org/abs/2602.01186",
        dataset_name=dataset_name.upper(),
        dataset_split="train",
        num_classes=num_classes,
        num_clients=num_clients,
        dirichlet_alpha=float(alpha),
        seed=int(seed),

        backbone="resnet18",
        weights_tag=str(weights_tag),
        feature_dim=int(FEATURE_DIM),

        preprocessing="Resize(224) + ToTensor() + ImageNet Normalize",
        image_size=224,
        normalization_mean=tuple(float(x) for x in IMAGENET_MEAN),
        normalization_std=tuple(float(x) for x in IMAGENET_STD),

        stats="A_per_class_x, SUMSQ_per_class_x, N_per_class, B_global_x, S_per_class_x (FULL QDA)",
        notes="X-space only (no RP). float64 accumulation. SVHN labels mapped with y%10.",
    )

    print("\n" + "=" * 90)
    print("GH-OFL | CLIENT | X-SPACE (NO RP) — CAMERA-READY")
    print("=" * 90)
    print(f"[INFO] dataset      : {meta.dataset_name} (train)")
    print(f"[INFO] num_classes  : {num_classes}")
    print(f"[INFO] num_clients  : {num_clients}")
    print(f"[INFO] dirichlet α  : {alpha}")
    print(f"[INFO] seed         : {seed}")
    print(f"[INFO] device       : {device.type}")
    print(f"[INFO] backbone     : resnet18 ({weights_tag}) | d={FEATURE_DIM}")
    print(f"[INFO] batch_size   : {batch_size} | workers={num_workers} | pin_memory={pin_memory}")
    print(f"[INFO] out_dir      : {os.path.abspath(out_dir)}")
    print("=" * 90 + "\n")

    # Per-client processing
    for cid, idxs in enumerate(splits):
        subset = Subset(ds, idxs)
        loader = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        )

        t1 = time.time()
        stats = aggregate_client_stats_x(
            model=model,
            loader=loader,
            num_classes=num_classes,
            device=device,
            dataset_name=dataset_name,
        )
        dt = time.time() - t1

        payload = {
            "meta": asdict(meta),
            "client_id": int(cid),
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
            **stats,
        }

        out_path = os.path.join(out_dir, f"client_{cid:02d}.pt")
        torch.save(payload, out_path)

        n_client = int(stats["N_per_class"].sum().item())
        print(f"[OK] client {cid:02d} -> {out_path} | samples={n_client} | time={dt/60.0:.2f} min")

    print("\n[DONE] All client stats written to:")
    print(f"       {os.path.abspath(out_dir)}\n")


if __name__ == "__main__":
    main()