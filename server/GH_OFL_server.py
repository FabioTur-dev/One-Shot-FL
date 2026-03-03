#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=====================================================================================
GH-OFL | SERVER | X-SPACE (NO RP) | CAMERA-READY (ICLR 2026)
=====================================================================================

UPDATED (minimal changes) to match your "better" implementation behavior:
  - FisherMix: Linear head (no cosine margin)
  - Proto-Hyper: base = NB_f (spherical in Fisher), student = base + residual(z_f)
  - KD teacher: teacher_blend * LDA_f + (1-teacher_blend) * NB_f
  - Loss: alpha_kd * KD + (1-alpha_kd) * CE   (same as your last code)

Everything else kept identical: aggregation, x-space NBdiag/LDA/QDAfull, Fisher subspace,
Gaussian synthesis, streaming evaluation.

=====================================================================================
"""

from __future__ import annotations

import os
import glob
import math
import time
import random
import argparse
import platform
from typing import Dict, List, Tuple, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import build_backbone


# -----------------------------
# Constants
# -----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

MIN_EPS = 1e-6


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# YAML / naming
# -----------------------------
def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def format_alpha(alpha: float) -> str:
    s = f"{alpha:.6f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


# -----------------------------
# Preprocessing / Backbone
# -----------------------------
def _resize_224():
    try:
        return transforms.Resize(224, antialias=True)
    except TypeError:
        return transforms.Resize(224)


def build_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((224, 224)),  # <-- resize diretto
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# -----------------------------
# CIFAR-100-C (robustness)
# -----------------------------
import tarfile
import urllib.request
import shutil

# Root of project (independent of where you launch the script)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")

CIFARC_DIR = os.path.join(DATA_DIR, "CIFAR-100-C")

ALL_CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
    "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
    "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter",
    "speckle_noise", "zoom_blur"
]


def ensure_cifar100c_present():
    if os.path.isdir(CIFARC_DIR):
        return

    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"
    tmp_tar = os.path.join("../data", "CIFAR-100-C.tar")

    os.makedirs("../data", exist_ok=True)
    print("[INFO] CIFAR-100-C non trovato: scarico (~1.3GB)...")
    urllib.request.urlretrieve(url, tmp_tar)

    print("[INFO] Estrazione...")
    with tarfile.open(tmp_tar, "r") as tf:
        tf.extractall("./data")

    os.remove(tmp_tar)

    if not os.path.isdir(CIFARC_DIR):
        for d in os.listdir("../data"):
            if d.lower().startswith("cifar-100-c"):
                shutil.move(os.path.join("../data", d), CIFARC_DIR)
                break


class CIFAR100C_Test(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, severities=(5,), corruptions=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.severities = list(severities)
        self.corruptions = list(corruptions) if corruptions is not None else ALL_CORRUPTIONS

        self.labels_all = np.load(os.path.join(root, "labels.npy"))
        self.items = []

        for corr in self.corruptions:
            _ = np.load(os.path.join(root, f"{corr}.npy"), mmap_mode="r")
            for s in self.severities:
                lo = (s - 1) * 10000
                hi = s * 10000
                for p in range(lo, hi):
                    self.items.append((corr, p))

        self._cache = {}

    def __len__(self):
        return len(self.items)

    def _get_arr(self, c):
        if c not in self._cache:
            self._cache[c] = np.load(os.path.join(self.root, f"{c}.npy"), mmap_mode="r")
        return self._cache[c]

    def __getitem__(self, idx):
        from PIL import Image
        c, p = self.items[idx]
        arr = self._get_arr(c)

        img_np = np.array(arr[p])
        img_pil = Image.fromarray(img_np)

        if self.transform is not None:
            img = self.transform(img_pil)
        else:
            img = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        y = int(self.labels_all[p])
        return img, y


def load_test_dataset(dataset_name: str, data_root: str, tfm, cfg: Dict):
    dataset_name = dataset_name.lower()

    if dataset_name == "svhn":
        return datasets.SVHN(data_root, split="test", transform=tfm, download=True), 10

    if dataset_name == "cifar10":
        return datasets.CIFAR10(data_root, train=False, transform=tfm, download=True), 10

    if dataset_name == "cifar100":
        return datasets.CIFAR100(data_root, train=False, transform=tfm, download=True), 100

    if dataset_name == "flowers102":
        from torchvision.datasets import Flowers102
        ds = Flowers102(root=data_root, split="test", transform=tfm, download=True)
        return ds, 102

    if dataset_name == "aircraft":
        from torchvision.datasets import FGVCAircraft
        ds = FGVCAircraft(root=data_root, split="test", transform=tfm, download=False)
        return ds, 100

    if dataset_name == "cifar100c":
        ensure_cifar100c_present()
        severities = tuple(cfg.get("corruption_severities", [5]))
        corruptions = cfg.get("corruptions", None)
        ds = CIFAR100C_Test(root=CIFARC_DIR, transform=tfm, severities=severities, corruptions=corruptions)
        return ds, 100

    raise ValueError(f"Unknown dataset: {dataset_name}")


def normalize_labels(y: torch.Tensor, dataset_name: str, device: torch.device) -> torch.Tensor:
    if dataset_name.lower() == "svhn":
        return (y.to(device, non_blocking=True) % 10).to(torch.long)
    return y.to(device, non_blocking=True).to(torch.long)


# -----------------------------
# Numerics helpers
# -----------------------------
def symmetrize(S: torch.Tensor) -> torch.Tensor:
    return 0.5 * (S + S.t())


def shrink_cov(S: torch.Tensor, alpha: float) -> torch.Tensor:
    S = symmetrize(S)
    d = S.shape[0]
    tr = torch.trace(S).clamp_min(0.0)
    return (1.0 - alpha) * S + alpha * (tr / max(1, d)) * torch.eye(d, dtype=S.dtype, device=S.device)


def invsqrt_psd(S: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    S = symmetrize(S)
    evals, evecs = torch.linalg.eigh(S)
    evals = evals.clamp_min(eps)
    return (evecs * evals.rsqrt().unsqueeze(0)) @ evecs.t()


def top1_correct(logits: torch.Tensor, y: torch.Tensor) -> int:
    return int((logits.argmax(1) == y).sum().item())


# -----------------------------
# Load & aggregate client stats
# -----------------------------
def list_client_files(stats_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(stats_dir, "client_*.pt")))


def safe_load_pt(path: str) -> Optional[Dict]:
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[WARN] Could not load {path}: {e}")
        return None


def aggregate_clients_x(paths: List[str], num_classes: int) -> Dict:
    """
    Supports:
    NEW:
      A_per_class_x, SUMSQ_per_class_x, N_per_class, B_global_x, S_per_class_x (optional)
    LEGACY:
      A_per_class, N_per_class, B_global, S_per_class (optional), SUMSQ_per_class (optional)
    """
    A_sum = None
    Q_sum = None
    N_sum = None
    B_sum = None
    S_sum = None
    meta0 = None

    valid = 0
    for p in paths:
        dct = safe_load_pt(p)
        if dct is None or not isinstance(dct, dict):
            continue

        if meta0 is None:
            meta0 = dct.get("meta", {})

        is_new = ("A_per_class_x" in dct) and ("B_global_x" in dct)

        if is_new:
            A = dct["A_per_class_x"]
            B = dct["B_global_x"]
            N = dct["N_per_class"]
            Q = dct.get("SUMSQ_per_class_x", None)
            S = dct.get("S_per_class_x", None)
        else:
            A = dct["A_per_class"]
            B = dct["B_global"]
            N = dct["N_per_class"]
            S = dct.get("S_per_class", None)
            Q = dct.get("SUMSQ_per_class", None)

        A = A.to(torch.float64)
        B = B.to(torch.float64)
        N = N.to(torch.long)
        S = S.to(torch.float64) if S is not None else None

        if Q is None and S is not None:
            Q = torch.diagonal(S, dim1=1, dim2=2).contiguous()
        elif Q is not None:
            Q = Q.to(torch.float64)

        if A_sum is None:
            A_sum = torch.zeros_like(A)
            N_sum = torch.zeros_like(N)
            B_sum = torch.zeros_like(B)
            if Q is not None:
                Q_sum = torch.zeros_like(Q)
            if S is not None:
                S_sum = torch.zeros_like(S)

        A_sum += A
        N_sum += N
        B_sum += B

        if Q is not None:
            if Q_sum is None:
                Q_sum = torch.zeros_like(Q)
            Q_sum += Q

        if S is not None:
            if S_sum is None:
                S_sum = torch.zeros_like(S)
            S_sum += S

        valid += 1

    if A_sum is None or valid == 0:
        raise RuntimeError("No valid client stats found.")

    C, d = A_sum.shape
    assert C == num_classes, f"Stats num_classes mismatch: stats C={C}, config C={num_classes}"
    N_total = int(N_sum.sum().item())

    mu = A_sum / N_sum.clamp_min(1).unsqueeze(1)  # [C,d]
    pri = (N_sum.to(torch.float64) / float(max(1, N_total))).clamp_min(1e-12)
    logp = torch.log(pri)

    SW = B_sum.clone()
    for c in range(C):
        Nc = int(N_sum[c].item())
        if Nc > 0:
            mc = mu[c].unsqueeze(1)
            SW -= float(Nc) * (mc @ mc.t())
    denom = float(max(1, N_total - C))
    Sigma_pool = symmetrize(SW / denom)

    if Q_sum is not None:
        Ez2 = Q_sum / N_sum.clamp_min(1).unsqueeze(1)
        Var_diag = (Ez2 - mu * mu).clamp_min(1e-6)
    else:
        print("[WARN] SUMSQ missing and S missing: falling back to pooled diagonal for NBdiag.")
        pooled_diag = torch.diagonal(Sigma_pool, dim1=0, dim2=1).unsqueeze(0).repeat(C, 1)
        Var_diag = pooled_diag.clamp_min(1e-6)

    Sigma_cls = None
    has_S = (S_sum is not None)
    if has_S:
        Sigma_cls = torch.zeros(C, d, d, dtype=torch.float64)
        for c in range(C):
            Nc = int(N_sum[c].item())
            mc = mu[c].unsqueeze(1)
            if Nc > 1:
                Sigma_cls[c] = (S_sum[c] - float(Nc) * (mc @ mc.t())) / float(Nc - 1)
            else:
                Sigma_cls[c] = Sigma_pool.clone()
        Sigma_cls = 0.5 * (Sigma_cls + Sigma_cls.transpose(1, 2))

    return dict(
        meta=meta0,
        num_files=valid,
        C=C, d=d, N_total=N_total,
        mu=mu, pri=pri, logp=logp,
        Sigma_pool=Sigma_pool,
        Var_diag=Var_diag,
        Sigma_cls=Sigma_cls,
        has_S=has_S,
    )


# -----------------------------
# Closed-form heads (x-space)
# -----------------------------
def logits_nbdiag_cpu(x64: torch.Tensor, mu64: torch.Tensor, var_diag64: torch.Tensor, logp64: torch.Tensor) -> torch.Tensor:
    dif = x64.unsqueeze(1) - mu64.unsqueeze(0)  # [B,C,d]
    quad = 0.5 * (dif * dif / var_diag64.unsqueeze(0)).sum(dim=2)   # [B,C]
    logdet = 0.5 * torch.log(var_diag64).sum(dim=1).unsqueeze(0)    # [1,C]
    return logp64.unsqueeze(0) - quad - logdet


def logits_lda_cpu(x64: torch.Tensor, mu64: torch.Tensor, Sig64: torch.Tensor, logp64: torch.Tensor, shrink: float) -> torch.Tensor:
    S = shrink_cov(Sig64, shrink)
    S = symmetrize(S) + MIN_EPS * torch.eye(S.shape[0], dtype=torch.float64, device=S.device)
    W = torch.linalg.solve(S, mu64.t())  # [d,C]
    b = (-0.5 * (mu64 * W.t()).sum(dim=1) + logp64).unsqueeze(0)
    return x64 @ W + b


class QDAFullFastGPU:
    def __init__(self, mu64_cpu: torch.Tensor, Sigma_cls64_cpu: torch.Tensor, logp64_cpu: torch.Tensor, shrink: float, device: torch.device):
        self.device = device
        self.shrink = float(shrink)

        C, d = mu64_cpu.shape
        self.C = int(C)
        self.d = int(d)

        self.mu = mu64_cpu.to(device, dtype=torch.float32)
        self.logp = logp64_cpu.to(device, dtype=torch.float32)
        Sig = Sigma_cls64_cpu.to(device, dtype=torch.float32)

        I = torch.eye(d, device=device, dtype=torch.float32)

        self.L_list: List[torch.Tensor] = []
        self.logdet_half = torch.empty(C, device=device, dtype=torch.float32)

        for c in range(C):
            Sc = shrink_cov(Sig[c], self.shrink)
            Sc = symmetrize(Sc) + (MIN_EPS * I)
            L = torch.linalg.cholesky(Sc)
            self.L_list.append(L)
            self.logdet_half[c] = torch.log(torch.diag(L)).sum()

    @torch.no_grad()
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        out = torch.empty(B, self.C, device=self.device, dtype=torch.float32)

        for c in range(self.C):
            dif = x - self.mu[c].unsqueeze(0)
            sol = torch.cholesky_solve(dif.unsqueeze(2), self.L_list[c]).squeeze(2)
            quad = 0.5 * (dif * sol).sum(dim=1)
            out[:, c] = self.logp[c] - quad - self.logdet_half[c]

        return out


# -----------------------------
# Fisher subspace + synthesis
# -----------------------------
def fisher_subspace(
    mu: torch.Tensor, pri: torch.Tensor, Sigma_pool: torch.Tensor,
    lda_shrink: float, energy: float, max_k: int
) -> Tuple[torch.Tensor, int, torch.Tensor]:

    mu = mu.to(torch.float64)
    pri = pri.to(torch.float64)
    C, d = mu.shape

    mbar = (pri.unsqueeze(1) * mu).sum(dim=0, keepdim=True)

    SB = torch.zeros(d, d, dtype=torch.float64)
    for c in range(C):
        dm = (mu[c:c+1] - mbar)
        SB += float(pri[c].item()) * (dm.t() @ dm)
    SB = symmetrize(SB)

    SW = shrink_cov(Sigma_pool.to(torch.float64), lda_shrink)
    SW = symmetrize(SW) + MIN_EPS * torch.eye(d, dtype=torch.float64)

    W_invsqrt = invsqrt_psd(SW, eps=1e-12)
    A = symmetrize(W_invsqrt @ SB @ W_invsqrt)

    lam, U = torch.linalg.eigh(A)
    lam = lam.clamp_min(0)
    idx = torch.argsort(lam, descending=True)
    lam = lam[idx]
    U = U[:, idx]

    if lam.numel() == 0 or float(lam.sum().item()) <= 0:
        kf = min(2, d)
        V = torch.eye(d, dtype=torch.float64)[:, :kf]
        return V.contiguous(), kf, lam

    csum = torch.cumsum(lam, dim=0)
    tot = float(csum[-1].item())
    kf = int((csum / max(tot, 1e-12) >= energy).nonzero(as_tuple=True)[0][0].item() + 1)
    kf = max(2, min(kf, max_k, d))

    V = (W_invsqrt @ U[:, :kf]).contiguous()
    return V, kf, lam


def synthesize_fisher(
    mu: torch.Tensor,
    Sigma_pool: torch.Tensor,
    Sigma_cls: Optional[torch.Tensor],
    V: torch.Tensor,
    ns_per_class: int,
    syn_shrink: float,
    base_tau: float,
    tau_clip_rng: Tuple[float, float],
    seed: int,
    use_class_cov: bool,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    Minimal upgrade: allow class-cov synthesis in Fisher (like your last code),
    otherwise fall back to pooled-cov synthesis.
    """
    set_seed(seed)

    mu64 = mu.to(torch.float64)
    V64 = V.to(torch.float64)

    C = mu64.shape[0]
    kf = V64.shape[1]

    tau = max(tau_clip_rng[0], min(tau_clip_rng[1], float(base_tau)))
    taus = [float(tau) for _ in range(C)]

    mu_f = mu64 @ V64  # [C,kf]

    Zs, Ys = [], []

    if use_class_cov and (Sigma_cls is not None):
        for c in range(C):
            Sc = shrink_cov(Sigma_cls[c].to(torch.float64), syn_shrink)
            Sf = V64.t() @ symmetrize(Sc) @ V64
            Sf = symmetrize(Sf) + MIN_EPS * torch.eye(kf, dtype=torch.float64)
            Sf = (tau ** 2) * Sf + MIN_EPS * torch.eye(kf, dtype=torch.float64)
            L = torch.linalg.cholesky(Sf)

            eps = (torch.randn(ns_per_class, kf, dtype=torch.float64) @ L.t())
            z = mu_f[c].unsqueeze(0) + eps
            y = torch.full((ns_per_class,), c, dtype=torch.long)
            Zs.append(z.to(torch.float32))
            Ys.append(y)
    else:
        Sp = shrink_cov(Sigma_pool.to(torch.float64), syn_shrink)
        Sf = V64.t() @ symmetrize(Sp) @ V64
        Sf = symmetrize(Sf) + MIN_EPS * torch.eye(kf, dtype=torch.float64)
        Sf = (tau ** 2) * Sf + MIN_EPS * torch.eye(kf, dtype=torch.float64)
        L = torch.linalg.cholesky(Sf)

        for c in range(C):
            eps = (torch.randn(ns_per_class, kf, dtype=torch.float64) @ L.t())
            z = mu_f[c].unsqueeze(0) + eps
            y = torch.full((ns_per_class,), c, dtype=torch.long)
            Zs.append(z.to(torch.float32))
            Ys.append(y)

    return torch.cat(Zs, dim=0), torch.cat(Ys, dim=0), taus


# -----------------------------
# Fisher-space Gaussians (LDA_f base)
# -----------------------------
def build_gaussians_fisher(
    mu: torch.Tensor, pri: torch.Tensor,
    Sigma_pool: torch.Tensor,
    V: torch.Tensor,
    lda_shrink: float,
    device: torch.device
) -> Dict:
    mu64 = mu.to(torch.float64)
    pri64 = pri.to(torch.float64)
    V64 = V.to(torch.float64)

    C = mu64.shape[0]
    kf = V64.shape[1]

    mu_f = (mu64 @ V64).to(device, dtype=torch.float64)  # [C,kf]
    logp = torch.log(pri64.clamp_min(1e-12)).to(device, dtype=torch.float64)

    Sig_f_pool = V64.t() @ shrink_cov(Sigma_pool.to(torch.float64), lda_shrink) @ V64
    Sig_f_pool = symmetrize(Sig_f_pool) + MIN_EPS * torch.eye(kf, dtype=torch.float64)
    Sig_f_pool = Sig_f_pool.to(device, dtype=torch.float64)

    # NB spherical variance in Fisher from pooled covariance
    sigma2_nb = float(torch.trace(Sig_f_pool).item() / float(max(1, kf)))
    sigma2_nb = max(sigma2_nb, 1e-12)

    return dict(
        mu_f=mu_f,              # float64
        Sig_f_pool=Sig_f_pool,  # float64
        logp=logp,              # float64
        sigma2_nb=sigma2_nb,    # float
        kf=kf,
    )


@torch.no_grad()
def lda_scores_f(zf: torch.Tensor, g: Dict) -> torch.Tensor:
    z64 = zf.to(torch.float64)
    mu_f = g["mu_f"]
    Sig  = g["Sig_f_pool"]
    logp = g["logp"]

    S = symmetrize(Sig) + MIN_EPS * torch.eye(Sig.shape[0], dtype=torch.float64, device=Sig.device)
    W = torch.linalg.solve(S, mu_f.t())
    b = -0.5 * (mu_f * W.t()).sum(dim=1) + logp
    out = z64 @ W + b.unsqueeze(0)
    return out.to(torch.float32)


@torch.no_grad()
def nb_scores_f(zf: torch.Tensor, g: Dict) -> torch.Tensor:
    """
    Spherical NB in Fisher: sigma2 from trace(Sig_f_pool)/kf
    """
    z = zf.to(torch.float32)
    mu = g["mu_f"].to(torch.float32)      # [C,kf]
    logp = g["logp"].to(torch.float32)    # [C]
    sigma2 = float(g["sigma2_nb"])
    kf = int(g["kf"])

    mu2 = (mu * mu).sum(dim=1)                 # [C]
    z2 = (z * z).sum(dim=1)                    # [B]
    zmu = z @ mu.t()                           # [B,C]
    quad = 0.5 * (z2.unsqueeze(1) - 2.0*zmu + mu2.unsqueeze(0)) / sigma2
    logdet = 0.5 * kf * math.log(sigma2 + 1e-12)
    return logp.unsqueeze(0) - quad - logdet


# -----------------------------
# Heads: Linear FisherMix + Proto-Hyper
# -----------------------------
class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.fc(z)


class ProtoHyperResidual(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, rank: int):
        super().__init__()
        r = int(min(rank, in_dim))
        self.U = nn.Linear(in_dim, r, bias=False)
        self.V = nn.Linear(r, num_classes, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.V(self.U(z))


def kd_mix_loss(student: torch.Tensor, teacher: torch.Tensor, y: torch.Tensor, alpha_kd: float, T: float) -> torch.Tensor:
    """
    Matches your last code:
      loss = alpha_kd * (T^2)*KL + (1-alpha_kd)*CE
    """
    ce = F.cross_entropy(student, y)
    if alpha_kd <= 0:
        return ce
    logp_s = F.log_softmax(student / T, dim=1)
    p_t = F.softmax(teacher / T, dim=1).detach()
    kl = F.kl_div(logp_s, p_t, reduction="batchmean")
    return alpha_kd * (T * T) * kl + (1.0 - alpha_kd) * ce


def train_fishermix(
    Zs_cpu: torch.Tensor, Ys_cpu: torch.Tensor,
    kf: int, C: int,
    device: torch.device,
    epochs: int, batch_size: int,
    lr: float, wd: float,
    # kept for backward compatibility (unused)
    scale: float, margin: float,
    is_win: bool
) -> LinearHead:
    fm = LinearHead(kf, C).to(device)
    opt = torch.optim.AdamW(fm.parameters(), lr=lr, weight_decay=wd)

    ds = torch.utils.data.TensorDataset(Zs_cpu, Ys_cpu)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=0 if is_win else 2,
        pin_memory=torch.cuda.is_available() and (not is_win)
    )

    for ep in range(1, epochs + 1):
        fm.train()
        tot = 0.0
        seen = 0
        for z, y in dl:
            z = z.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = fm(z)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = z.size(0)
            tot += loss.item() * bs
            seen += bs

        if ep % 10 == 0 or ep == 1 or ep == epochs:
            print(f"[FisherMix] ep {ep:03d}/{epochs} | loss={tot/max(1,seen):.4f}")

    return fm


def train_protohyper(
    Zs_cpu: torch.Tensor, Ys_cpu: torch.Tensor,
    kf: int, C: int,
    g_f: Dict,
    device: torch.device,
    epochs: int, batch_size: int,
    lr: float, wd: float,
    rank: int,
    kd_alpha: float, kd_T: float,
    teacher_blend: float,
    is_win: bool
) -> ProtoHyperResidual:
    """
    Proto-Hyper (your working variant):
      base    = NB_f(z)
      teacher = teacher_blend * LDA_f(z) + (1-teacher_blend) * NB_f(z)
      student = base + residual(z)
      loss    = alpha*KD + (1-alpha)*CE
    """
    ph = ProtoHyperResidual(kf, C, rank=min(rank, kf)).to(device)
    opt = torch.optim.AdamW(ph.parameters(), lr=lr, weight_decay=wd)

    ds = torch.utils.data.TensorDataset(Zs_cpu, Ys_cpu)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=0 if is_win else 2,
        pin_memory=torch.cuda.is_available() and (not is_win)
    )

    w = float(teacher_blend)

    for ep in range(1, epochs + 1):
        ph.train()
        tot = 0.0
        seen = 0
        for z, y in dl:
            z = z.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                base = nb_scores_f(z, g_f)         # [B,C]
                lda  = lda_scores_f(z, g_f)        # [B,C]
                teacher = w * lda + (1.0 - w) * base

            student = base + ph(z)
            loss = kd_mix_loss(student, teacher, y, alpha_kd=kd_alpha, T=kd_T)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = z.size(0)
            tot += loss.item() * bs
            seen += bs

        if ep % 10 == 0 or ep == 1 or ep == epochs:
            print(f"[Proto-Hyper] ep {ep:03d}/{epochs} | loss={tot/max(1,seen):.4f} | teacher={w:.2f}*LDA + {1-w:.2f}*NB")

    return ph


# -----------------------------
# Streaming evaluation
# -----------------------------
@torch.no_grad()
def evaluate_streaming(
    dataset_name: str,
    loader: DataLoader,
    fe: nn.Module,
    device: torch.device,
    mu64: torch.Tensor,
    var_diag64: torch.Tensor,
    Sig_pool64: torch.Tensor,
    logp64: torch.Tensor,
    qda_x: Optional[QDAFullFastGPU],
    V64_cpu: torch.Tensor,
    g_f: Dict,
    fm: Optional[LinearHead],
    ph: Optional[ProtoHyperResidual],
) -> Dict[str, Optional[float]]:

    have_qda_x = (qda_x is not None)
    have_fm = (fm is not None)
    have_ph = (ph is not None)

    tot = 0
    corr_nb = 0
    corr_lda = 0
    corr_qda = 0 if have_qda_x else None
    corr_fm = 0 if have_fm else None
    corr_ph = 0 if have_ph else None

    V64 = V64_cpu.to(torch.float64)

    for imgs, y in loader:
        imgs = imgs.to(device, non_blocking=True)
        y_dev = normalize_labels(y, dataset_name, device)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            x32 = fe(imgs).to(torch.float32)

        B = x32.size(0)
        tot += B

        x64_cpu = x32.detach().cpu().to(torch.float64)
        y_cpu = y_dev.detach().cpu()

        nb_logits = logits_nbdiag_cpu(x64_cpu, mu64, var_diag64, logp64)
        lda_logits = logits_lda_cpu(x64_cpu, mu64, Sig_pool64, logp64, shrink=0.0)

        corr_nb += top1_correct(nb_logits, y_cpu)
        corr_lda += top1_correct(lda_logits, y_cpu)

        if have_qda_x:
            q_logits = qda_x.logits(x32)
            corr_qda += top1_correct(q_logits, y_dev)

        if have_fm or have_ph:
            zf = (x64_cpu @ V64).to(torch.float32).to(device, non_blocking=True)

        if have_fm:
            fm_logits = fm(zf)
            corr_fm += top1_correct(fm_logits, y_dev)

        if have_ph:
            base = nb_scores_f(zf, g_f)
            ph_logits = base + ph(zf)
            corr_ph += top1_correct(ph_logits, y_dev)

    def pct(x: int) -> float:
        return 100.0 * float(x) / float(max(1, tot))

    return dict(
        GH_NBdiag=pct(corr_nb),
        GH_LDA=pct(corr_lda),
        GH_QDAfull=None if corr_qda is None else pct(corr_qda),
        FisherMix=None if corr_fm is None else pct(corr_fm),
        ProtoHyper=None if corr_ph is None else pct(corr_ph),
        total=int(tot),
    )


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="GH-OFL Server | X-space full (no RP)")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    dataset_name: str = str(cfg["dataset"]).lower()
    alpha: float = float(cfg["dirichlet_alpha"])

    backbone_cfg: Optional[str] = cfg.get("backbone", None)
    if backbone_cfg is not None:
        backbone_cfg = str(backbone_cfg).lower()

    seed: int = int(cfg.get("seed", 123))
    data_root: str = str(cfg.get("data_root", "./data"))
    stats_root: str = str(cfg.get("stats_root", "./client_stats_X"))

    test_batch: int = int(cfg.get("test_batch", 256))

    lda_shrink: float = float(cfg.get("lda_shrink", 0.05))
    qda_shrink: float = float(cfg.get("qda_shrink", 0.05))

    fisher_energy: float = float(cfg.get("fisher_energy", 0.99))
    fisher_max_k: int = int(cfg.get("fisher_max_dim", 128))

    syn_ns_per_class: int = int(cfg.get("syn_ns_per_class", 10000))
    syn_shrink: float = float(cfg.get("syn_shrink", 0.0))
    syn_tau_base: float = float(cfg.get("syn_tau_base", 0.75))
    syn_tau_clip: Tuple[float, float] = tuple(cfg.get("syn_tau_clip", (0.60, 1.40)))  # type: ignore
    syn_use_class_cov: bool = bool(cfg.get("syn_use_class_cov", True))

    fm_epochs: int = int(cfg.get("fm_epochs", 80))
    fm_batch: int = int(cfg.get("fm_batch", 2048))
    fm_lr: float = float(cfg.get("fm_lr", 8e-4))
    fm_wd: float = float(cfg.get("fm_wd", 3e-4))

    # kept for backwards compatibility (unused)
    fm_scale: float = float(cfg.get("fm_scale", 30.0))
    fm_margin: float = float(cfg.get("fm_margin", 0.05))

    ph_epochs: int = int(cfg.get("ph_epochs", 20))
    ph_batch: int = int(cfg.get("ph_batch", 1024))
    ph_lr: float = float(cfg.get("ph_lr", 1e-3))
    ph_wd: float = float(cfg.get("ph_wd", 5e-4))
    ph_rank: int = int(cfg.get("ph_rank", 128))
    kd_alpha: float = float(cfg.get("kd_alpha", 0.85))
    kd_T: float = float(cfg.get("kd_T", 4.0))

    # IMPORTANT: now interpreted as LDA weight in teacher (0.7 -> 0.7*LDA +0.3*NB)
    teacher_blend: float = float(cfg.get("teacher_blend", 0.7))

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_win = platform.system().lower().startswith("win")
    pin_memory = torch.cuda.is_available() and (not is_win)
    num_workers = int(cfg.get("num_workers", 2 if is_win else 4))

    tfm = build_transform()
    testset, num_classes = load_test_dataset(dataset_name, data_root, tfm, cfg)

    # -----------------------------
    # Stats directory selection
    # -----------------------------
    alpha_tag = format_alpha(alpha)

    if dataset_name == "cifar100c":

        base_dir = os.path.join(stats_root, "CIFAR100C")

        if not os.path.isdir(base_dir):
            raise RuntimeError(f"CIFAR100C stats folder not found: {base_dir}")

        cand = sorted(glob.glob(os.path.join(base_dir, f"*_TRAIN_A{alpha_tag}_X*")))

        if not cand:
            raise RuntimeError(
                f"No CIFAR100C stats folder found under {base_dir} matching *_TRAIN_A{alpha_tag}_X*"
            )

        if backbone_cfg is not None:
            cand = [
                d for d in cand
                if os.path.basename(d).lower().startswith(backbone_cfg + "-")
            ]
            if not cand:
                raise RuntimeError(f"No CIFAR100C stats folder for backbone={backbone_cfg}")

        stats_dir = cand[0]
        paths = list_client_files(stats_dir)

        if len(paths) == 0:
            raise RuntimeError(f"No client_*.pt found in: {stats_dir}")

    else:
        base_dir = os.path.join(stats_root, dataset_name.upper())

        if not os.path.isdir(base_dir):
            raise RuntimeError(f"Dataset stats folder not found: {base_dir}")

        cand = sorted(glob.glob(os.path.join(base_dir, f"*_TRAIN_A{alpha_tag}_X*")))

        if not cand:
            raise RuntimeError(
                f"No stats folder found under {base_dir} matching *_TRAIN_A{alpha_tag}_X*"
            )

        if backbone_cfg is not None:
            cand = [
                d for d in cand
                if os.path.basename(d).lower().startswith(backbone_cfg + "-")
            ]
            if not cand:
                raise RuntimeError(f"No stats folder for backbone={backbone_cfg}")

        stats_dir = cand[0]
        paths = list_client_files(stats_dir)

        if len(paths) == 0:
            raise RuntimeError(f"No client_*.pt found in: {stats_dir}")

    # -----------------------------
    # Meta consistency
    # -----------------------------
    first = safe_load_pt(paths[0])
    if first is None:
        raise RuntimeError(f"Could not load first client file for meta check: {paths[0]}")
    meta = first.get("meta", {}) if isinstance(first, dict) else {}
    meta_backbone = str(meta.get("backbone", "")).lower()
    meta_weights = str(meta.get("weights_tag", ""))
    meta_dim = int(meta.get("feature_dim", -1))

    if backbone_cfg is not None and meta_backbone != backbone_cfg:
        raise RuntimeError(
            "Backbone mismatch between YAML and client meta.\n"
            f"  YAML backbone : {backbone_cfg}\n"
            f"  META backbone : {meta_backbone}\n"
            f"  stats_dir     : {stats_dir}\n"
        )

    backbone_name = backbone_cfg if backbone_cfg is not None else meta_backbone
    if backbone_name is None or backbone_name.strip() == "":
        raise RuntimeError("Could not infer backbone from meta and YAML did not specify it.")

    if meta_dim <= 0:
        raise RuntimeError("Client meta does not contain a valid feature_dim.")

    fe, feature_dim, weights_tag = build_backbone(backbone_name, device)

    if int(feature_dim) != int(meta_dim):
        raise RuntimeError(
            "Feature dimension mismatch between server backbone and client meta.\n"
            f"  backbone_name : {backbone_name}\n"
            f"  server dim    : {feature_dim}\n"
            f"  meta dim      : {meta_dim}\n"
            f"  stats_dir     : {stats_dir}\n"
        )

    if meta_weights.strip() != "" and str(weights_tag) != str(meta_weights):
        raise RuntimeError(
            "Weights tag mismatch between server backbone and client meta.\n"
            f"  backbone_name : {backbone_name}\n"
            f"  server weights: {weights_tag}\n"
            f"  meta weights  : {meta_weights}\n"
            f"  stats_dir     : {stats_dir}\n"
        )

    print("\n" + "=" * 90)
    print("GH-OFL | SERVER | X-SPACE (NO RP) — CAMERA-READY FULL")
    print("=" * 90)
    print(f"[INFO] dataset       : {dataset_name.upper()} (test)")
    print(f"[INFO] dirichlet α   : {alpha}   (tag={alpha_tag})")
    print(f"[INFO] seed          : {seed}")
    print(f"[INFO] device        : {device.type}")
    print(f"[INFO] data_root     : {os.path.abspath(data_root)}")
    print(f"[INFO] stats_dir     : {os.path.abspath(stats_dir)}")
    print(f"[INFO] client files  : {len(paths)}")
    print(f"[INFO] shrink        : LDA={lda_shrink} | QDA={qda_shrink}")
    print(f"[INFO] fisher        : energy={fisher_energy} | max_k={fisher_max_k}")
    print(f"[INFO] synth         : ns/class={syn_ns_per_class} | shrink={syn_shrink} | tau={syn_tau_base} clip={syn_tau_clip} | class_cov={syn_use_class_cov}")
    print(f"[INFO] train FM      : epochs={fm_epochs} batch={fm_batch} lr={fm_lr} wd={fm_wd} (LINEAR)")
    print(f"[INFO] train PH      : epochs={ph_epochs} batch={ph_batch} lr={ph_lr} wd={ph_wd} rank={ph_rank} KD={kd_alpha} T={kd_T} teacher={teacher_blend:.2f}*LDA+{1-teacher_blend:.2f}*NB")
    print("-" * 90)
    print(f"[INFO] backbone(meta): {meta_backbone} ({meta_weights}) | d={meta_dim}")
    if backbone_cfg is not None:
        print(f"[INFO] backbone(cfg): {backbone_cfg} (enforced)")
    print(f"[INFO] backbone(srv): {backbone_name} ({weights_tag}) | d={feature_dim}")
    print("=" * 90 + "\n")

    t0 = time.time()

    # -----------------------------
    # Aggregate client stats
    # -----------------------------
    agg = aggregate_clients_x(paths, num_classes=num_classes)
    mu = agg["mu"].to(torch.float64)
    pri = agg["pri"].to(torch.float64)
    logp = agg["logp"].to(torch.float64)
    Sig_pool = agg["Sigma_pool"].to(torch.float64)
    Var_diag = agg["Var_diag"].to(torch.float64)
    Sig_cls = agg["Sigma_cls"].to(torch.float64) if agg["Sigma_cls"] is not None else None

    Sig_pool_shrunk = shrink_cov(Sig_pool, lda_shrink)
    Sig_pool_shrunk = symmetrize(Sig_pool_shrunk) + MIN_EPS * torch.eye(Sig_pool_shrunk.shape[0], dtype=torch.float64)

    qda_x = None
    if Sig_cls is not None:
        print("[INFO] Precomputing QDAfull(x) Cholesky on GPU...")
        qda_x = QDAFullFastGPU(mu, Sig_cls, logp, shrink=qda_shrink, device=device)
    else:
        print("[WARN] S_per_class missing: GH-QDAfull(x) disabled.")

    # -----------------------------
    # Fisher subspace
    # -----------------------------
    print("[INFO] Building Fisher subspace...")
    V, kf, lam = fisher_subspace(mu, pri, Sig_pool, lda_shrink=lda_shrink, energy=fisher_energy, max_k=fisher_max_k)
    V64_cpu = V.to(torch.float64).cpu()

    if lam.numel() > 0:
        top = [float(x) for x in lam[:min(10, lam.numel())].cpu()]
        print(f"[DBG] Fisher kf={kf} | top eigvals: {top}")
    else:
        print(f"[DBG] Fisher kf={kf} | no eigvals (degenerate SB/SW)")

    # -----------------------------
    # Synthesis in Fisher
    # -----------------------------
    print("[INFO] Synthesizing Fisher-space Gaussian data...")
    Zs_cpu, Ys_cpu, taus = synthesize_fisher(
        mu=mu,
        Sigma_pool=Sig_pool,
        Sigma_cls=Sig_cls,
        V=V,
        ns_per_class=syn_ns_per_class,
        syn_shrink=syn_shrink,
        base_tau=syn_tau_base,
        tau_clip_rng=(float(syn_tau_clip[0]), float(syn_tau_clip[1])),
        seed=seed,
        use_class_cov=syn_use_class_cov,
    )
    print(f"[INFO] Synth total={Zs_cpu.shape[0]} | dim={Zs_cpu.shape[1]} | tau={taus[0]:.3f}")

    # -----------------------------
    # Fisher-space Gaussians (LDA_f + NB_f helpers)
    # -----------------------------
    print("[INFO] Building Fisher-space Gaussians (LDA_f + NB_f)...")
    g_f = build_gaussians_fisher(
        mu=mu, pri=pri,
        Sigma_pool=Sig_pool,
        V=V,
        lda_shrink=lda_shrink,
        device=device
    )

    # -----------------------------
    # Train FisherMix (LINEAR)
    # -----------------------------
    print("[INFO] Training FisherMix (LINEAR)...")
    fm = train_fishermix(
        Zs_cpu=Zs_cpu, Ys_cpu=Ys_cpu,
        kf=kf, C=num_classes,
        device=device,
        epochs=fm_epochs, batch_size=fm_batch,
        lr=fm_lr, wd=fm_wd,
        scale=fm_scale, margin=fm_margin,   # unused
        is_win=is_win
    )
    fm.eval()

    # -----------------------------
    # Train Proto-Hyper (NB base + KD teacher)
    # -----------------------------
    print("[INFO] Training Proto-Hyper (NB base + KD teacher)...")
    ph = train_protohyper(
        Zs_cpu=Zs_cpu, Ys_cpu=Ys_cpu,
        kf=kf, C=num_classes,
        g_f=g_f,
        device=device,
        epochs=ph_epochs, batch_size=ph_batch,
        lr=ph_lr, wd=ph_wd,
        rank=ph_rank,
        kd_alpha=kd_alpha, kd_T=kd_T,
        teacher_blend=teacher_blend,
        is_win=is_win
    )
    ph.eval()

    # -----------------------------
    # Test loader + streaming eval
    # -----------------------------
    test_loader = DataLoader(
        testset,
        batch_size=test_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    print("[INFO] Evaluating on test (streaming)...")
    t_eval = time.time()
    res = evaluate_streaming(
        dataset_name=dataset_name,
        loader=test_loader,
        fe=fe,
        device=device,
        mu64=mu.cpu(),
        var_diag64=Var_diag.cpu(),
        Sig_pool64=Sig_pool_shrunk.cpu(),
        logp64=logp.cpu(),
        qda_x=qda_x,
        V64_cpu=V64_cpu,
        g_f=g_f,
        fm=fm,
        ph=ph
    )
    dt_eval = time.time() - t_eval

    dt = time.time() - t0
    print("\n" + "=" * 90)
    print("FINAL SUMMARY (X-SPACE, FULL PIPELINE)")
    print("=" * 90)
    print(f"[INFO] dataset     : {dataset_name.upper()} | alpha={alpha} | clients={len(paths)}")
    print(f"[INFO] stats_dir   : {os.path.abspath(stats_dir)}")
    print(f"[INFO] backbone    : {backbone_name} ({weights_tag}) | d={feature_dim}")
    print(f"[INFO] eval_time   : {dt_eval/60.0:.2f} min | total_time={dt/60.0:.2f} min")
    print("-" * 90)
    print(f"GH-NBdiag   : {res['GH_NBdiag']:6.2f}%")
    print(f"GH-LDA      : {res['GH_LDA']:6.2f}%  (shrink={lda_shrink})")
    if res["GH_QDAfull"] is None:
        print("GH-QDAfull  :   n/a  (S_per_class missing)")
    else:
        print(f"GH-QDAfull  : {res['GH_QDAfull']:6.2f}%  (shrink={qda_shrink})")
    print("-" * 90)
    print(f"FisherMix   : {res['FisherMix']:6.2f}%  (linear head, kf={kf})")
    print(f"Proto-Hyper : {res['ProtoHyper']:6.2f}%  (NB base + residual, kf={kf}, rank={min(ph_rank,kf)}, KD={kd_alpha}, T={kd_T}, teacher={teacher_blend:.2f}*LDA+{1-teacher_blend:.2f}*NB)")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()