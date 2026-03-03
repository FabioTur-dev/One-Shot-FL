#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=====================================================================================
GH-OFL | CLIENT | X-SPACE (NO RP) | CAMERA-READY (ICLR 2026)
=====================================================================================

Paper:
  "The Gaussian-Head OFL Family: One-Shot Federated Learning from Client Global Statistics"
  arXiv:2602.01186

Client-side:
  - Frozen ImageNet backbone (configurable)
  - Deterministic Dirichlet split
  - X-space sufficient statistics (float64 accumulation)
  - No raw data, no gradients

Supported backbones:
  resnet18, resnet50, mobilenet_v2, efficientnet_b0, vgg16

CIFAR-100-C:
  - TRUE legacy_holdout protocol
  - Train stats from severities [1,2,3,4]
  - All corruptions by default (unless specified)

=====================================================================================
"""

from __future__ import annotations

import os
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
from torchvision import datasets, transforms

from models import build_backbone


# =============================================================================
# CONSTANTS
# =============================================================================

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

ALL_CORRUPTIONS = [
    "brightness","contrast","defocus_blur","elastic_transform",
    "fog","frost","gaussian_blur","gaussian_noise",
    "glass_blur","impulse_noise","jpeg_compression",
    "motion_blur","pixelate","saturate",
    "shot_noise","snow","spatter",
    "speckle_noise","zoom_blur"
]


# =============================================================================
# META PAYLOAD
# =============================================================================

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


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# PREPROCESSING
# =============================================================================

def _resize_224():
    try:
        return transforms.Resize(224, antialias=True)
    except TypeError:
        return transforms.Resize(224)


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        _resize_224(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# =============================================================================
# CIFAR-100-C DATASET
# =============================================================================

class CIFAR100CSubset(torch.utils.data.Dataset):

    def __init__(self, root, corruptions, severities, transform):
        self.root = root
        self.transform = transform
        self.labels_all = np.load(os.path.join(root, "labels.npy"))
        self.items = []
        self.cache = {}

        for corr in corruptions:
            arr = np.load(os.path.join(root, f"{corr}.npy"), mmap_mode="r")
            for s in severities:
                lo = (s - 1) * 10000
                hi = s * 10000
                for p in range(lo, hi):
                    self.items.append((corr, p))

    def __len__(self):
        return len(self.items)

    def _get(self, c):
        if c not in self.cache:
            self.cache[c] = np.load(
                os.path.join(self.root, f"{c}.npy"),
                mmap_mode="r"
            )
        return self.cache[c]

    def __getitem__(self, idx):
        from PIL import Image
        corr, p = self.items[idx]
        img = self._get(corr)[p]
        img = Image.fromarray(np.array(img))

        if self.transform:
            img = self.transform(img)

        y = int(self.labels_all[p])
        return img, y


# =============================================================================
# CLEAN DATASETS
# =============================================================================

def load_train_dataset(dataset_name: str, data_root: str, tfm):
    dataset_name = dataset_name.lower()

    if dataset_name == "svhn":
        ds = datasets.SVHN(root=data_root, split="train", transform=tfm, download=True)
        return ds, 10

    if dataset_name == "cifar10":
        ds = datasets.CIFAR10(root=data_root, train=True, transform=tfm, download=True)
        return ds, 10

    if dataset_name == "cifar100":
        ds = datasets.CIFAR100(root=data_root, train=True, transform=tfm, download=True)
        return ds, 100

    raise ValueError(f"Unknown dataset: {dataset_name}")


def get_labels(ds, dataset_name: str) -> List[int]:
    dataset_name = dataset_name.lower()

    if dataset_name == "svhn":
        y = np.array(ds.labels, dtype=np.int64) % 10
    else:
        y = np.array(ds.targets, dtype=np.int64)

    return y.tolist()


# =============================================================================
# DIRICHLET SPLIT
# =============================================================================

def dirichlet_split(
    labels: List[int],
    num_classes: int,
    num_clients: int,
    alpha: float,
    seed: int
) -> List[List[int]]:

    set_seed(seed)

    labels_t = torch.tensor(labels, dtype=torch.long)
    per_client: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx = torch.where(labels_t == c)[0]
        idx = idx[torch.randperm(len(idx))]

        dist = torch.distributions.Dirichlet(
            torch.full((num_clients,), float(alpha))
        ).sample()

        counts = (dist * len(idx)).round().to(torch.long).tolist()

        diff = len(idx) - sum(counts)
        for k in range(abs(diff)):
            counts[k % num_clients] += 1 if diff > 0 else -1

        counts = [max(0, int(x)) for x in counts]

        s = 0
        for i in range(num_clients):
            e = s + counts[i]
            if e > s:
                per_client[i].extend(idx[s:e].tolist())
            s = e

    for i in range(num_clients):
        rng = random.Random(seed + 1000 + i)
        rng.shuffle(per_client[i])

    return per_client


# =============================================================================
# STATS AGGREGATION
# =============================================================================

@torch.no_grad()
def aggregate_client_stats_x(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    feature_dim: int,
    device: torch.device,
    dataset_name: str,
) -> Dict[str, torch.Tensor]:

    C = num_classes
    d = feature_dim

    A = torch.zeros(C, d, dtype=torch.float64, device=device)
    SUMSQ = torch.zeros(C, d, dtype=torch.float64, device=device)
    N = torch.zeros(C, dtype=torch.long, device=device)
    B = torch.zeros(d, d, dtype=torch.float64, device=device)
    S = torch.zeros(C, d, d, dtype=torch.float64, device=device)

    for imgs, y in loader:
        imgs = imgs.to(device, non_blocking=True)

        if dataset_name.lower() == "svhn":
            y = (y.to(device) % 10).long()
        else:
            y = y.to(device).long()

        x32 = model(imgs).float()
        x64 = x32.double()

        B += x64.t().mm(x64)

        for cls in y.unique():
            ci = int(cls.item())
            mask = (y == cls)
            if mask.any():
                xc = x64[mask]
                A[ci] += xc.sum(dim=0)
                SUMSQ[ci] += (xc * xc).sum(dim=0)
                N[ci] += int(mask.sum().item())
                S[ci] += xc.t().mm(xc)

    return {
        "A_per_class_x": A.cpu(),
        "SUMSQ_per_class_x": SUMSQ.cpu(),
        "N_per_class": N.cpu(),
        "B_global_x": B.cpu(),
        "S_per_class_x": S.cpu(),
    }


# =============================================================================
# CONFIG
# =============================================================================

def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_alpha(alpha: float) -> str:
    s = f"{alpha:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    dataset_name = str(cfg["dataset"]).lower()
    backbone_name = str(cfg.get("backbone", "resnet18")).lower()
    num_clients = int(cfg["num_clients"])
    alpha = float(cfg["dirichlet_alpha"])
    seed = int(cfg.get("seed", 42))
    batch_size = int(cfg.get("batch_size", 512))
    data_root = str(cfg.get("data_root", "./data"))
    out_root = str(cfg.get("out_root", "./client_stats_X"))

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_win = platform.system().lower().startswith("win")
    pin_memory = torch.cuda.is_available() and not is_win
    num_workers = int(cfg.get("num_workers", 2 if is_win else 4))

    tfm = build_transform()

    # =====================================================================
    # DATASET LOADING (MODIFIED FOR CIFAR-100-C LEGACY HOLDOUT)
    # =====================================================================

    if dataset_name == "cifar100c":

        cifar100c_dir = os.path.join(data_root, "CIFAR-100-C")
        severities = cfg.get("severities_train", [1, 2, 3, 4])
        corruptions = cfg.get("corruptions_train", ALL_CORRUPTIONS)

        ds = CIFAR100CSubset(
            cifar100c_dir,
            corruptions,
            severities,
            tfm
        )

        labels = [ds.labels_all[p] for (_, p) in ds.items]
        num_classes = 100

    else:

        ds, num_classes = load_train_dataset(
            dataset_name,
            data_root,
            tfm
        )

        labels = get_labels(ds, dataset_name)

    # Backbone
    model, feature_dim, weights_tag = build_backbone(backbone_name, device)

    alpha_tag = format_alpha(alpha)

    out_dir = os.path.join(
        out_root,
        dataset_name.upper(),
        f"{backbone_name}-{weights_tag}_TRAIN_A{alpha_tag}_X{feature_dim}"
    )
    os.makedirs(out_dir, exist_ok=True)

    splits = dirichlet_split(
        labels,
        num_classes,
        num_clients,
        alpha,
        seed
    )

    meta = MetaInfo(
        paper="The Gaussian-Head OFL Family: One-Shot Federated Learning from Client Global Statistics",
        arxiv="https://arxiv.org/abs/2602.01186",
        dataset_name=dataset_name.upper(),
        dataset_split="train",
        num_classes=num_classes,
        num_clients=num_clients,
        dirichlet_alpha=alpha,
        seed=seed,
        backbone=backbone_name,
        weights_tag=weights_tag,
        feature_dim=feature_dim,
        preprocessing="Resize(224) + ToTensor() + ImageNet Normalize",
        image_size=224,
        normalization_mean=IMAGENET_MEAN,
        normalization_std=IMAGENET_STD,
        stats="A_per_class_x, SUMSQ_per_class_x, N_per_class, B_global_x, S_per_class_x",
        notes="X-space only. float64 accumulation.",
    )

    print("\n" + "=" * 80)
    print("GH-OFL CLIENT (X-space)")
    print("=" * 80)
    print(f"Dataset     : {dataset_name}")
    print(f"Backbone    : {backbone_name} ({weights_tag})")
    print(f"Feature dim : {feature_dim}")
    print(f"Clients     : {num_clients}")
    print(f"Alpha       : {alpha}")
    print("=" * 80 + "\n")

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

        t0 = time.time()

        stats = aggregate_client_stats_x(
            model=model,
            loader=loader,
            num_classes=num_classes,
            feature_dim=feature_dim,
            device=device,
            dataset_name=dataset_name,
        )

        payload = {
            "meta": asdict(meta),
            "client_id": cid,
            "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
            **stats,
        }

        out_path = os.path.join(out_dir, f"client_{cid:02d}.pt")
        torch.save(payload, out_path)

        n_samples = int(stats["N_per_class"].sum().item())
        dt = (time.time() - t0) / 60.0

        print(f"[OK] client {cid:02d} | samples={n_samples} | {dt:.2f} min")

    print("\n[DONE]")
    print(f"Output dir: {os.path.abspath(out_dir)}\n")


if __name__ == "__main__":
    main()