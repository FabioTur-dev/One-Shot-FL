#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Client-side — CIFAR-10, ResNet18 (ImageNet), Random Projection (k=256).
Genera statistiche aggregate A/N/B + SUMSQ in spazio RP (z = x @ R) per due split:
- alpha=0.5 -> ./client_stats_RP/CIFAR10/resnet18-IMAGENET1K_V1
- alpha=0.1 -> ./client_stats_RP_a0p1/CIFAR10/resnet18-IMAGENET1K_V1_RP256_A0p1
"""

import os, math, random, time, warnings, platform
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

# ----------------------------
# Config
# ----------------------------
DATA_ROOT = "./data"

OUT_ALPHA05 = "./client_stats_RP/CIFAR10/resnet18-IMAGENET1K_V1"
OUT_ALPHA01 = "./client_stats_RP_a0p1/CIFAR10/resnet18-IMAGENET1K_V1_RP256_A0p1"

NUM_CLIENTS = 10
ALPHAS = [0.5, 0.1]  # generiamo entrambi gli split
BATCH_SIZE = 512
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEV = torch.device(DEVICE)
IS_WIN = platform.system().lower().startswith("win")
PIN_MEMORY = torch.cuda.is_available() and (not IS_WIN)
NUM_WORKERS = 2 if IS_WIN else 4

# Random Projection
RP_SEED = 12345
K_RP = 256  # dim z

warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum or None for 'weights' are deprecated")
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", message="Anti-alias option is always applied for PIL Image input.")

# ----------------------------
# Meta dataclass
# ----------------------------
@dataclass
class MetaInfo:
    dataset_name: str
    num_clients: int
    dirichlet_alpha: float
    feature_dim: int
    rp_dim: int
    rp_seed: int
    backbone: str
    source_type: str
    weights_tag: str
    img_size: int
    normalization_mean: Tuple[float, float, float]
    normalization_std: Tuple[float, float, float]
    class_names: Tuple[str, ...]
    has_sumsq_per_class: bool

# ----------------------------
# Utils
# ----------------------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_resnet18_strong():
    try:
        from torchvision.models import ResNet18_Weights
        w = ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=w)
        mean = tuple(float(x) for x in w.meta["mean"])
        std = tuple(float(x) for x in w.meta["std"])
    except Exception:
        model = models.resnet18(pretrained=True)
        mean, std = (0.485,0.456,0.406), (0.229,0.224,0.225)
    model.fc = nn.Identity()
    model.eval().to(DEV)
    tfm = transforms.Compose([
        transforms.Resize(224, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return model, 512, tfm, 224, mean, std

def build_rp_matrix(d: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.randn(d, k, generator=g) / math.sqrt(d)  # [d,k] CPU float32
    return R

def get_cifar10_train(tfm):
    ds = datasets.CIFAR10(root=DATA_ROOT, train=True, transform=tfm, download=True)
    return ds

def get_labels(ds) -> List[int]:
    return [ds[i][1] for i in range(len(ds))]

def dirichlet_split(labels: List[int], num_classes: int, num_clients: int, alpha: float, seed: int):
    set_seed(seed)
    labels_t = torch.tensor(labels, dtype=torch.long)
    per_client = [[] for _ in range(num_clients)]
    for c in range(num_classes):
        idx = torch.where(labels_t == c)[0]
        idx = idx[torch.randperm(len(idx))]
        dist = torch.distributions.Dirichlet(torch.full((num_clients,), float(alpha))).sample()
        counts = (dist * len(idx)).round().to(torch.long).tolist()

        # correggi somma
        diff = len(idx) - sum(counts)
        for k in range(abs(diff)):
            counts[k % num_clients] += 1 if diff > 0 else -1

        # bounding
        counts = [max(0, int(x)) for x in counts]
        while sum(counts) > len(idx):
            for k in range(num_clients):
                if sum(counts) <= len(idx): break
                if counts[k] > 0: counts[k] -= 1
        while sum(counts) < len(idx):
            counts[sum(counts) % num_clients] += 1

        # assegna
        s = 0
        for i in range(num_clients):
            e = s + counts[i]
            if e > s:
                per_client[i].extend(idx[s:e].tolist())
            s = e

    # shuffle per client
    for i in range(num_clients):
        rng = random.Random(seed + i)
        rng.shuffle(per_client[i])

    return per_client

@torch.no_grad()
def aggregate_stats_z(loader: DataLoader, R: torch.Tensor, C: int):
    k = R.shape[1]

    A = torch.zeros(C, k, dtype=torch.float64, device=DEV)
    SUMSQ = torch.zeros(C, k, dtype=torch.float64, device=DEV)
    N = torch.zeros(C, dtype=torch.long, device=DEV)
    B = torch.zeros(k, k, dtype=torch.float64, device=DEV)

    # tieni R in float32 sul device
    R_dev32 = R.to(DEV, dtype=torch.float32)

    for imgs, y in loader:
        imgs = imgs.to(DEV, non_blocking=True)
        y = y.to(DEV, non_blocking=True)

        # niente autocast qui (o comunque riportiamo a float32)
        x = loader.feature_extractor(imgs)  # [B,512]
        x = x.to(torch.float32)  # <— forza FP32

        # z in FP32 sul device
        z = x @ R_dev32  # [B,k] float32

        B += (z.t().mm(z)).to(torch.float64)

        # per-classe
        for cls in y.unique():
            cls_i = int(cls.item())
            mask = (y == cls)
            if mask.any():
                zc = z[mask].to(torch.float64)  # [m,k] in float64 per accumulo
                A[cls_i] += zc.sum(dim=0)
                SUMSQ[cls_i] += (zc * zc).sum(dim=0)
                N[cls_i] += int(mask.sum().item())

    return {
        "A_per_class_z": A.cpu(),
        "SUMSQ_per_class_z": SUMSQ.cpu(),
        "N_per_class": N.cpu(),
        "B_global_z": B.cpu(),
    }

def main():
    set_seed(SEED)

    model, d, tfm, img_size, mean, std = build_resnet18_strong()
    class_names = tuple(datasets.CIFAR10(root=DATA_ROOT, train=True, download=True).classes)
    C = len(class_names)

    R = build_rp_matrix(d, K_RP, RP_SEED)  # condivisa per tutti i client/alpha

    # definisci output dirs
    out_dirs = {
        0.5: OUT_ALPHA05,
        0.1: OUT_ALPHA01,
    }

    # dataset completo (stessa normalizzazione del FE)
    trainset = get_cifar10_train(tfm)
    labels = get_labels(trainset)

    # meta base
    meta_base = dict(
        dataset_name="CIFAR10",
        num_clients=NUM_CLIENTS,
        feature_dim=d,
        rp_dim=K_RP,
        rp_seed=RP_SEED,
        backbone="resnet18",
        source_type="torchvision",
        weights_tag="IMAGENET1K_V1",
        img_size=img_size,
        normalization_mean=tuple(float(x) for x in mean),
        normalization_std=tuple(float(x) for x in std),
        class_names=class_names,
        has_sumsq_per_class=True,
    )

    def _attach_fe(x):
        return model(x)

    for alpha in ALPHAS:
        out_dir = out_dirs[alpha]
        os.makedirs(out_dir, exist_ok=True)

        # split Dirichlet
        splits = dirichlet_split(labels, C, NUM_CLIENTS, alpha=alpha, seed=SEED)

        print(f"\n[INFO] Genero client CIFAR-10 | alpha={alpha} | dir={out_dir}")
        meta = MetaInfo(**{**meta_base, "dirichlet_alpha": float(alpha)})

        # per client
        for cid, idxs in enumerate(splits):
            subset = Subset(trainset, idxs)
            loader = DataLoader(
                subset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                drop_last=False,
            )
            loader.feature_extractor = _attach_fe  # type: ignore

            stats = aggregate_stats_z(loader, R, C)

            payload = {
                "meta": asdict(meta),
                "client_id": cid,
                "A_per_class_z": stats["A_per_class_z"],
                "SUMSQ_per_class_z": stats["SUMSQ_per_class_z"],
                "N_per_class": stats["N_per_class"],
                "B_global_z": stats["B_global_z"],
                "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
            }

            out_path = os.path.join(out_dir, f"client_{cid:02d}.pt")
            torch.save(payload, out_path)

            print(
                f"[OK] client {cid:02d} -> {out_path} | "
                f"samples={int(stats['N_per_class'].sum().item())}"
            )

    print("\n[DONE] Statistiche salvate in:")
    for alpha in ALPHAS:
        print(" -", os.path.abspath(out_dirs[alpha]))

if __name__ == "__main__":
    main()
