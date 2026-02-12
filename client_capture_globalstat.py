#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client-side (RP + var per-classe in z) per CIFAR-10 e CIFAR-100.

Per dataset in {CIFAR10, CIFAR100}:
  - Split non-IID via Dirichlet(alpha)
  - Estrae feature STRONG con ResNet18 (ImageNet) -> x ∈ R^d (d=512)
  - Proietta con RP: z = x @ R  (R ∈ R^{d×k}, seed pubblico)
  - Accumula per client:
      A_per_class_z   [C,k] float64   (somma z per classe)
      N_per_class     [C]   int64
      B_global_z      [k,k] float64   (somma z^T z su tutto il client)
      SUMSQ_per_class [C,k] float64   (somma z∘z per classe)  → var per-classe in server
  - Salva in: ./client_stats_RP/<DATASET>/resnet18-IMAGENET1K_V1/client_XX.pt
"""

import os, time, random, warnings, math
from dataclasses import asdict, dataclass
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# ----------------------------
# Config
# ----------------------------
DATA_ROOT       = "./data"
OUTPUT_ROOT     = "./client_stats_RP"
NUM_CLIENTS     = 10
DIRICHLET_ALPHA = 0.5
BATCH_SIZE      = 512
NUM_WORKERS     = 2
SEED            = 42
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY      = torch.cuda.is_available()
# Random Projection
RP_SEED         = 12345
K_RP            = 256  # dimensione z

DATASETS = ["CIFAR10", "CIFAR100"]

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights' are deprecated")
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated since 0.13")
warnings.filterwarnings("ignore", message="Anti-alias option is always applied for PIL Image input.")

# ----------------------------
# Meta dataclass
# ----------------------------
@dataclass
class MetaInfo:
    dataset_name: str
    num_clients: int
    dirichlet_alpha: float
    feature_dim: int            # d (ResNet18 penultimo layer)
    rp_dim: int                 # k (RP)
    rp_seed: int
    backbone: str
    weights_tag: str
    img_size: int
    normalization_mean: Tuple[float, float, float]
    normalization_std: Tuple[float, float, float]
    class_names: Tuple[str, ...]
    has_sumsq_per_class: bool   # abbiamo SUMSQ_per_class per var per-classe

# ----------------------------
# Seeds
# ----------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------
# Dataset helpers
# ----------------------------
def get_train_dataset(name: str, tfm):
    if name == "CIFAR10":
        return datasets.CIFAR10(root=DATA_ROOT, train=True, transform=tfm, download=True)
    if name == "CIFAR100":
        return datasets.CIFAR100(root=DATA_ROOT, train=True, transform=tfm, download=True)
    raise ValueError(f"Dataset non supportato: {name}")

def get_labels_and_classes(name: str):
    base_tfm = transforms.ToTensor()
    if name == "CIFAR10":
        ds = datasets.CIFAR10(root=DATA_ROOT, train=True, transform=base_tfm, download=True)
        labels = [ds[i][1] for i in range(len(ds))]
        return labels, tuple(ds.classes)
    if name == "CIFAR100":
        ds = datasets.CIFAR100(root=DATA_ROOT, train=True, transform=base_tfm, download=True)
        labels = [ds[i][1] for i in range(len(ds))]
        return labels, tuple(ds.classes)
    raise ValueError(f"Dataset non supportato: {name}")

# ----------------------------
# Backbone (ResNet18 STRONG)
# ----------------------------
def build_resnet18_strong():
    img_size = 224
    try:
        from torchvision.models import ResNet18_Weights
        w = ResNet18_Weights.IMAGENET1K_V1
        m = models.resnet18(weights=w)
        mean = tuple(float(x) for x in w.meta["mean"])
        std  = tuple(float(x) for x in w.meta["std"])
    except Exception:
        m = models.resnet18(pretrained=True)
        mean, std = (0.485,0.456,0.406), (0.229,0.224,0.225)
    m.fc = nn.Identity()
    m.eval().to(DEVICE)

    tfm = transforms.Compose([
        transforms.Resize(img_size, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return m, 512, tfm, img_size, mean, std

# ----------------------------
# Random Projection
# ----------------------------
def build_rp_matrix(d: int, k: int, seed: int) -> torch.Tensor:
    # Gaussian RP con varianza 1/d (He-like) → mantiene scala
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.randn(d, k, generator=g) / math.sqrt(d)
    return R  # [d,k], cpu float32

# ----------------------------
# Split Dirichlet
# ----------------------------
def dirichlet_split(labels: List[int], num_classes: int, num_clients: int, alpha: float, seed: int):
    set_seed(seed)
    labels_t = torch.tensor(labels, dtype=torch.long)
    per_client = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx = torch.where(labels_t == c)[0]
        idx = idx[torch.randperm(len(idx))]
        dist = torch.distributions.Dirichlet(torch.full((num_clients,), float(alpha))).sample()
        counts = (dist * len(idx)).round().to(torch.long).tolist()

        diff = len(idx) - sum(counts)
        for k in range(abs(diff)):
            counts[k % num_clients] += 1 if diff > 0 else -1
        counts = [max(0, int(x)) for x in counts]
        while sum(counts) > len(idx):
            for k in range(num_clients):
                if sum(counts) <= len(idx): break
                if counts[k] > 0: counts[k] -= 1
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

# ----------------------------
# Accumulo A/N/B/SUMSQ in z
# ----------------------------
@torch.no_grad()
def aggregate_stats_z(loader: DataLoader, num_classes: int, R: torch.Tensor):
    k = R.shape[1]
    A = torch.zeros(num_classes, k, dtype=torch.float64, device=DEVICE)
    SUMSQ = torch.zeros(num_classes, k, dtype=torch.float64, device=DEVICE)
    N = torch.zeros(num_classes, dtype=torch.long, device=DEVICE)
    B = torch.zeros(k, k, dtype=torch.float64, device=DEVICE)
    total = 0

    for imgs, y in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        x = loader.feature_extractor(imgs)        # [B,512] on DEVICE
        z = x @ R.to(DEVICE)                      # [B,k]
        B += z.t().mm(z).to(torch.float64)
        total += z.size(0)

        # per-classe
        for cls in y.unique():
            cls = int(cls.item())
            mask = (y == cls)
            if mask.any():
                zc = z[mask].to(torch.float64)    # [m,k]
                A[cls] += zc.sum(dim=0)
                SUMSQ[cls] += (zc * zc).sum(dim=0)
                N[cls] += int(mask.sum().item())

    return {
        "A_per_class_z": A.detach().cpu(),
        "SUMSQ_per_class_z": SUMSQ.detach().cpu(),
        "N_per_class": N.detach().cpu(),
        "B_global_z": B.detach().cpu(),
        "num_samples": int(total),
    }

# ----------------------------
# Main
# ----------------------------
def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    set_seed(SEED)

    model, d, tfm, img_size, mean, std = build_resnet18_strong()
    R = build_rp_matrix(d, K_RP, RP_SEED)  # cpu

    for dset_name in DATASETS:
        labels, class_names = get_labels_and_classes(dset_name)
        C = len(class_names)
        splits = dirichlet_split(labels, num_classes=C, num_clients=NUM_CLIENTS,
                                 alpha=DIRICHLET_ALPHA, seed=SEED)

        trainset = get_train_dataset(dset_name, tfm)
        out_dir = os.path.join(OUTPUT_ROOT, dset_name, "resnet18-IMAGENET1K_V1")
        os.makedirs(out_dir, exist_ok=True)

        meta = MetaInfo(
            dataset_name=dset_name,
            num_clients=NUM_CLIENTS,
            dirichlet_alpha=DIRICHLET_ALPHA,
            feature_dim=d,
            rp_dim=K_RP,
            rp_seed=RP_SEED,
            backbone="resnet18",
            weights_tag="IMAGENET1K_V1",
            img_size=img_size,
            normalization_mean=tuple(float(x) for x in mean),
            normalization_std=tuple(float(x) for x in std),
            class_names=class_names,
            has_sumsq_per_class=True,
        )

        def _attach_fe(x): return model(x)

        for i, idxs in enumerate(splits):
            subset = Subset(trainset, idxs)
            loader = DataLoader(
                subset, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=False
            )
            loader.feature_extractor = _attach_fe  # type: ignore

            stats = aggregate_stats_z(loader, num_classes=C, R=R)
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
            out_path = os.path.join(out_dir, f"client_{i:02d}.pt")
            torch.save(payload, out_path)
            counts = payload["N_per_class"].tolist()
            print(f"[OK] {dset_name}/resnet18-IMAGENET1K_V1 | client {i:02d} -> {out_path} | "
                  f"samples={payload['num_samples']} | "
                  f"SZ_per_class=yes | class_counts[sum]={sum(counts)}")

    print("\n[DONE] Statistiche RP client salvate in:", os.path.abspath(OUTPUT_ROOT))

if __name__ == "__main__":
    main()






