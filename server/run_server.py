#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=====================================================================================
GH-OFL | SERVER | X-SPACE (NO RP) | CAMERA-READY (ICLR 2026)
=====================================================================================

Paper:
  "The Gaussian-Head OFL Family: One-Shot Federated Learning from Client Global Statistics"
  arXiv:2602.01186

What this script does (SERVER-SIDE, TEST EVAL):
  1) Loads client statistics saved by client/run_client.py (X-space, no RP).
  2) Aggregates per-client statistics into global moments (no raw data).
  3) Evaluates closed-form Gaussian heads in x-space:
       - GH-NBdiag   : diagonal Naive Gaussian from SUMSQ moments
       - GH-LDA      : pooled covariance from B_global_x and means (shrinkage)
       - GH-QDAfull  : full per-class covariance from S_per_class_x (shrinkage), FAST GPU
  4) Builds Fisher subspace from whitened SB/SW (LDA shrinkage).
  5) Generates data-free synthetic samples in Fisher space (Gaussian synthesis).
  6) Trains Fisher-space heads on synthetic data:
       - FisherMix   : cosine classifier (ArcFace-like), trained on synth
       - Proto-Hyper : low-rank residual over standardized LDA_f logits, trained with KD:
            base    = LDA_f logits
            teacher = blend * QDA_f + (1-blend) * LDA_f  (all in Fisher space; cheap because kf << 512)
            student = standardize(base) + residual(z_f)
            loss    = KD(KL) + CE

Inference for Proto-Hyper:
  - Uses the STUDENT:  standardize(LDA_f(z_f)) + residual(z_f)
  - QDA_f / teacher is used ONLY during training as distillation signal (not required at inference).

Design notes:
  - X-space only (no RP), equivalent to your full "SVHN server X-space" pipeline.
  - Numerics: aggregation in float64 CPU; QDAfull in x implemented as fast GPU with per-class Cholesky.
  - Evaluation is streaming (no need to keep full test set features in RAM).
  - Default hyperparameters match your full SVHN code; YAML can override.

Expected input layout (repo-relative):
  ./client_stats_X/{DATASET}/resnet18-IMAGENET1K_V1_TRAIN_A{alpha}_X512/client_XX.pt

Config:
  Use --config configs/{dataset}.yaml
  Required YAML keys:
    dataset: "svhn" | "cifar10" | "cifar100"
    dirichlet_alpha: float
  Optional YAML keys (defaults are paper-coherent):
    seed, data_root, stats_root, batch sizes, shrinkages, fisher params, synthesis, training, KD, etc.

Example:
  python server/run_server.py --config configs/svhn.yaml

IMPORTANT REPRODUCIBILITY NOTE:
  If your repository code has evolved since the paper experiments, exact numbers may differ slightly
  (e.g., due to library versions, minor numeric changes, or refactors), while the methodology remains
  consistent with the paper.

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
from torchvision import datasets, transforms, models


# -----------------------------
# Constants
# -----------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

FEATURE_DIM = 512
MIN_EPS = 1e-6


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior helps reproducibility but may reduce throughput.
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
        _resize_224(),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_resnet18_imagenet(device: torch.device) -> Tuple[nn.Module, str]:
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


# -----------------------------
# Dataset
# -----------------------------
def load_test_dataset(dataset_name: str, data_root: str, tfm) -> Tuple[torch.utils.data.Dataset, int]:
    dataset_name = dataset_name.lower()

    if dataset_name == "svhn":
        ds = datasets.SVHN(root=data_root, split="test", transform=tfm, download=True)
        return ds, 10

    if dataset_name == "cifar10":
        ds = datasets.CIFAR10(root=data_root, train=False, transform=tfm, download=True)
        return ds, 10

    if dataset_name == "cifar100":
        ds = datasets.CIFAR100(root=data_root, train=False, transform=tfm, download=True)
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


def standardize_logits(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = logits.mean(dim=1, keepdim=True)
    s = logits.std(dim=1, keepdim=True, unbiased=False).clamp_min(eps)
    return (logits - m) / s


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
    Aggregates client payloads (CPU float64):
      A_per_class_x, SUMSQ_per_class_x, N_per_class, B_global_x, S_per_class_x

    Returns:
      mu [C,d], pri [C], logp [C], Sigma_pool [d,d], Var_diag [C,d], Sigma_cls [C,d,d] (optional)
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
        if dct is None:
            continue

        if meta0 is None:
            meta0 = dct.get("meta", {})

        A = dct["A_per_class_x"].to(torch.float64)        # [C,d]
        Q = dct["SUMSQ_per_class_x"].to(torch.float64)    # [C,d]
        N = dct["N_per_class"].to(torch.long)             # [C]
        B = dct["B_global_x"].to(torch.float64)           # [d,d]
        S = dct.get("S_per_class_x", None)                # [C,d,d] optional

        if A_sum is None:
            A_sum = torch.zeros_like(A)
            Q_sum = torch.zeros_like(Q)
            N_sum = torch.zeros_like(N)
            B_sum = torch.zeros_like(B)
            if S is not None:
                S_sum = torch.zeros_like(S.to(torch.float64))

        A_sum += A
        Q_sum += Q
        N_sum += N
        B_sum += B

        if S is not None:
            if S_sum is None:
                S_sum = torch.zeros_like(S.to(torch.float64))
            S_sum += S.to(torch.float64)

        valid += 1

    if A_sum is None or valid == 0:
        raise RuntimeError("No valid client stats found.")

    C, d = A_sum.shape
    assert C == num_classes, f"Stats num_classes mismatch: stats C={C}, config C={num_classes}"
    N_total = int(N_sum.sum().item())

    mu = A_sum / N_sum.clamp_min(1).unsqueeze(1)  # [C,d]
    pri = (N_sum.to(torch.float64) / float(max(1, N_total))).clamp_min(1e-12)
    logp = torch.log(pri)

    # Pooled within-class covariance from:
    #   SW = B - sum_c N_c * mu_c mu_c^T, then / (N_total - C)
    SW = B_sum.clone()
    for c in range(C):
        Nc = int(N_sum[c].item())
        if Nc > 0:
            mc = mu[c].unsqueeze(1)
            SW -= float(Nc) * (mc @ mc.t())
    denom = float(max(1, N_total - C))
    Sigma_pool = symmetrize(SW / denom)

    # NB diagonal variances:
    #   Var_c = E[x^2] - (E[x])^2  from SUMSQ and A
    Ez2 = Q_sum / N_sum.clamp_min(1).unsqueeze(1)
    Var_diag = (Ez2 - mu * mu).clamp_min(1e-6)

    # Full per-class covariance if S exists
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
    # x64: [B,d], mu64: [C,d], var_diag64: [C,d]
    dif = x64.unsqueeze(1) - mu64.unsqueeze(0)  # [B,C,d]
    quad = 0.5 * (dif * dif / var_diag64.unsqueeze(0)).sum(dim=2)   # [B,C]
    logdet = 0.5 * torch.log(var_diag64).sum(dim=1).unsqueeze(0)    # [1,C]
    return logp64.unsqueeze(0) - quad - logdet


def logits_lda_cpu(x64: torch.Tensor, mu64: torch.Tensor, Sig64: torch.Tensor, logp64: torch.Tensor, shrink: float) -> torch.Tensor:
    S = shrink_cov(Sig64, shrink)
    S = symmetrize(S) + MIN_EPS * torch.eye(S.shape[0], dtype=torch.float64)
    W = torch.linalg.solve(S, mu64.t())  # [d,C]
    b = (-0.5 * (mu64 * W.t()).sum(dim=1) + logp64).unsqueeze(0)
    return x64 @ W + b


class QDAFullFastGPU:
    """
    FAST QDA_full in x-space (d=512):
      - per-class cholesky precompute on GPU float32
      - per-batch cholesky_solve
    """
    def __init__(self, mu64_cpu: torch.Tensor, Sigma_cls64_cpu: torch.Tensor, logp64_cpu: torch.Tensor, shrink: float, device: torch.device):
        self.device = device
        self.shrink = float(shrink)

        C, d = mu64_cpu.shape
        self.C = int(C)
        self.d = int(d)

        self.mu = mu64_cpu.to(device, dtype=torch.float32)      # [C,d]
        self.logp = logp64_cpu.to(device, dtype=torch.float32)  # [C]
        Sig = Sigma_cls64_cpu.to(device, dtype=torch.float32)   # [C,d,d]

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
        """
        x: [B,d] float32 on device
        returns logits [B,C] float32
        """
        B = x.shape[0]
        out = torch.empty(B, self.C, device=self.device, dtype=torch.float32)

        for c in range(self.C):
            dif = x - self.mu[c].unsqueeze(0)  # [B,d]
            sol = torch.cholesky_solve(dif.unsqueeze(2), self.L_list[c]).squeeze(2)  # [B,d]
            quad = 0.5 * (dif * sol).sum(dim=1)  # [B]
            out[:, c] = self.logp[c] - quad - self.logdet_half[c]

        return out


# -----------------------------
# Fisher subspace + synthesis
# -----------------------------
def fisher_subspace(
    mu: torch.Tensor, pri: torch.Tensor, Sigma_pool: torch.Tensor,
    lda_shrink: float, energy: float, max_k: int
) -> Tuple[torch.Tensor, int, torch.Tensor]:
    """
    Stable Fisher subspace:
      SB from class means + priors
      SW = shrink(Sigma_pool, lda_shrink)
      whiten SB with SW^{-1/2}, eigendecompose, take top-k for given energy.
    Returns V [d,kf] (float64 CPU), kf, eigenvalues.
    """
    mu = mu.to(torch.float64)
    pri = pri.to(torch.float64)
    C, d = mu.shape

    mbar = (pri.unsqueeze(1) * mu).sum(dim=0, keepdim=True)  # [1,d]

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

    V = (W_invsqrt @ U[:, :kf]).contiguous()  # [d,kf]
    return V, kf, lam


def synthesize_fisher(
    mu: torch.Tensor,
    Sigma_pool: torch.Tensor,
    V: torch.Tensor,
    ns_per_class: int,
    syn_shrink: float,
    base_tau: float,
    tau_clip_rng: Tuple[float, float],
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    """
    Data-free Gaussian synthesis in Fisher space (pooled cov), matching your SVHN full code:
      - pooled cov in Fisher (shrunk with syn_shrink)
      - constant tau per class (clipped)
    Returns:
      Zs_cpu float32 [C*ns, kf], Ys_cpu long [C*ns], taus list
    """
    set_seed(seed)

    mu64 = mu.to(torch.float64)
    V64 = V.to(torch.float64)

    C = mu64.shape[0]
    kf = V64.shape[1]

    Sp = shrink_cov(Sigma_pool.to(torch.float64), syn_shrink)
    Sf = V64.t() @ symmetrize(Sp) @ V64
    Sf = symmetrize(Sf) + MIN_EPS * torch.eye(kf, dtype=torch.float64)

    tau = max(tau_clip_rng[0], min(tau_clip_rng[1], float(base_tau)))
    taus = [float(tau) for _ in range(C)]

    mu_f = mu64 @ V64  # [C,kf]
    L = torch.linalg.cholesky(Sf + MIN_EPS * torch.eye(kf, dtype=torch.float64))

    Zs, Ys = [], []
    for c in range(C):
        eps = (torch.randn(ns_per_class, kf, dtype=torch.float64) @ L.t()) * float(taus[c])
        z = mu_f[c].unsqueeze(0) + eps
        y = torch.full((ns_per_class,), c, dtype=torch.long)
        Zs.append(z.to(torch.float32))
        Ys.append(y)
    return torch.cat(Zs, dim=0), torch.cat(Ys, dim=0), taus


# -----------------------------
# Fisher-space Gaussians (LDA_f + optional QDA_f)
# -----------------------------
def build_gaussians_fisher(
    mu: torch.Tensor, pri: torch.Tensor,
    Sigma_pool: torch.Tensor, Sigma_cls: Optional[torch.Tensor],
    V: torch.Tensor,
    lda_shrink: float, qda_shrink: float,
    device: torch.device
) -> Dict:
    """
    Build Fisher-space Gaussian parameters.
    Uses float64 on GPU for stable solves (kf <= 128).

    NOTE (performance fix, same math):
      We ALSO precompute QDA_f Cholesky + logdet ONCE here (if Sigma_cls exists),
      so Proto-Hyper KD does not recompute them per batch/epoch.
    """
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

    Sig_f_cls = None
    qda_Ls: Optional[List[torch.Tensor]] = None
    qda_logdet_half: Optional[torch.Tensor] = None

    if Sigma_cls is not None:
        Sig_f_cls = torch.zeros(C, kf, kf, dtype=torch.float64)
        for c in range(C):
            Sc = shrink_cov(Sigma_cls[c].to(torch.float64), qda_shrink)
            Sf = V64.t() @ symmetrize(Sc) @ V64
            Sig_f_cls[c] = symmetrize(Sf) + MIN_EPS * torch.eye(kf, dtype=torch.float64)
        Sig_f_cls = Sig_f_cls.to(device, dtype=torch.float64)

        # ---- PRECOMPUTE QDA_f Cholesky + logdet ONCE (same math as before) ----
        I = torch.eye(kf, dtype=torch.float64, device=device)
        qda_Ls = []
        qda_logdet_half = torch.empty(C, dtype=torch.float64, device=device)
        for c in range(C):
            Sc = symmetrize(Sig_f_cls[c]) + MIN_EPS * I
            L = torch.linalg.cholesky(Sc)
            qda_Ls.append(L)
            qda_logdet_half[c] = torch.log(torch.diag(L)).sum()

    return dict(
        mu_f=mu_f,
        Sig_f_pool=Sig_f_pool,
        Sig_f_cls=Sig_f_cls,
        logp=logp,
        # cached QDA_f factors (optional)
        qda_Ls=qda_Ls,
        qda_logdet_half=qda_logdet_half,
    )


@torch.no_grad()
def lda_scores_f(zf: torch.Tensor, g: Dict) -> torch.Tensor:
    # zf: [B,kf] float32/float64 on device
    z64 = zf.to(torch.float64)
    mu_f = g["mu_f"]        # [C,kf] float64
    Sig  = g["Sig_f_pool"]  # [kf,kf] float64
    logp = g["logp"]        # [C] float64

    S = symmetrize(Sig) + MIN_EPS * torch.eye(Sig.shape[0], dtype=torch.float64, device=Sig.device)
    W = torch.linalg.solve(S, mu_f.t())  # [kf,C]
    b = -0.5 * (mu_f * W.t()).sum(dim=1) + logp
    out = z64 @ W + b.unsqueeze(0)
    return out.to(torch.float32)


@torch.no_grad()
def qda_scores_f(zf: torch.Tensor, g: Dict, chunk: int = 8192) -> torch.Tensor:
    assert g["Sig_f_cls"] is not None
    z64 = zf.to(torch.float64)

    mu_f = g["mu_f"]        # [C,kf] float64
    SigC = g["Sig_f_cls"]   # [C,kf,kf] float64
    logp = g["logp"]        # [C] float64

    C, kf, _ = SigC.shape

    # ---- USE CACHED CHOLESKY IF PRESENT (same math, no recompute) ----
    Ls = g.get("qda_Ls", None)
    logdet_half = g.get("qda_logdet_half", None)

    if (Ls is None) or (logdet_half is None):
        # Fallback (should not happen now): compute once per call
        I = torch.eye(kf, dtype=torch.float64, device=SigC.device)
        Ls = []
        logdet_half = torch.empty(C, dtype=torch.float64, device=SigC.device)
        for c in range(C):
            Sc = symmetrize(SigC[c]) + MIN_EPS * I
            L = torch.linalg.cholesky(Sc)
            Ls.append(L)
            logdet_half[c] = torch.log(torch.diag(L)).sum()

    N = z64.shape[0]
    out = torch.empty(N, C, dtype=torch.float64, device=SigC.device)

    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        zc = z64[s:e]
        for c in range(C):
            dif = zc - mu_f[c].unsqueeze(0)  # [B,kf]
            sol = torch.cholesky_solve(dif.unsqueeze(2), Ls[c]).squeeze(2)
            quad = 0.5 * (dif * sol).sum(dim=1)
            out[s:e, c] = logp[c] - quad - logdet_half[c]

    return out.to(torch.float32)


# -----------------------------
# Heads: FisherMix + Proto-Hyper
# -----------------------------
class CosineMarginHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, scale: float, margin: float):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, in_dim) * 0.01)
        self.scale = float(scale)
        self.margin = float(margin)

    def forward(self, z: torch.Tensor, y: Optional[torch.Tensor] = None) -> torch.Tensor:
        z = F.normalize(z, dim=1)
        W = F.normalize(self.W, dim=1)
        cos = z @ W.t()
        if (y is None) or (self.margin <= 0):
            return self.scale * cos
        onehot = torch.zeros_like(cos)
        onehot.scatter_(1, y.view(-1, 1), 1.0)
        return self.scale * (cos - onehot * self.margin)


class ProtoHyperResidual(nn.Module):
    """Low-rank residual adapter over Fisher features (paper style)."""
    def __init__(self, in_dim: int, num_classes: int, rank: int):
        super().__init__()
        r = int(min(rank, in_dim))
        self.U = nn.Linear(in_dim, r, bias=False)
        self.V = nn.Linear(r, num_classes, bias=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.V(self.U(z))


def kd_ce_loss(student: torch.Tensor, teacher: torch.Tensor, y: torch.Tensor, alpha_kd: float, T: float) -> torch.Tensor:
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
    scale: float, margin: float,
    is_win: bool
) -> CosineMarginHead:
    fm = CosineMarginHead(kf, C, scale=scale, margin=margin).to(device)
    opt = torch.optim.AdamW(fm.parameters(), lr=lr, weight_decay=wd)

    ds = torch.utils.data.TensorDataset(Zs_cpu, Ys_cpu)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=0 if is_win else 2, pin_memory=torch.cuda.is_available() and (not is_win)
    )

    for ep in range(1, epochs + 1):
        fm.train()
        tot = 0.0
        seen = 0
        for z, y in dl:
            z = z.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = fm(z, y)
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
    ph = ProtoHyperResidual(kf, C, rank=min(rank, kf)).to(device)
    opt = torch.optim.AdamW(ph.parameters(), lr=lr, weight_decay=wd)

    ds = torch.utils.data.TensorDataset(Zs_cpu, Ys_cpu)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, drop_last=False,
        num_workers=0 if is_win else 2, pin_memory=torch.cuda.is_available() and (not is_win)
    )

    has_qda_f = (g_f["Sig_f_cls"] is not None) and (teacher_blend > 0.0)

    for ep in range(1, epochs + 1):
        ph.train()
        tot = 0.0
        seen = 0
        for z, y in dl:
            z = z.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                base = lda_scores_f(z, g_f)                 # [B,C]
                base_s = standardize_logits(base)

                if has_qda_f:
                    q = qda_scores_f(z, g_f)                # [B,C]  (NOW cached Cholesky => no explosion)
                    teacher = teacher_blend * q + (1.0 - teacher_blend) * base
                else:
                    teacher = base

                teacher_s = standardize_logits(teacher)

            student = base_s + ph(z)
            loss = kd_ce_loss(student, teacher_s, y, alpha_kd=kd_alpha, T=kd_T)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = z.size(0)
            tot += loss.item() * bs
            seen += bs

        if ep % 10 == 0 or ep == 1 or ep == epochs:
            tag = "LDA+QDA" if has_qda_f else "LDA-only"
            print(f"[Proto-Hyper] ep {ep:03d}/{epochs} | loss={tot/max(1,seen):.4f} | teacher={tag}")

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
    # Aggregated stats (CPU float64)
    mu64: torch.Tensor,
    var_diag64: torch.Tensor,
    Sig_pool64: torch.Tensor,
    logp64: torch.Tensor,
    # QDAfull predictor (optional)
    qda_x: Optional[QDAFullFastGPU],
    # Fisher objects
    V64_cpu: torch.Tensor,
    g_f: Dict,
    fm: Optional[CosineMarginHead],
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

    for imgs, y in loader:
        imgs = imgs.to(device, non_blocking=True)
        y_dev = normalize_labels(y, dataset_name, device)

        # Feature extraction (x-space)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            x32 = fe(imgs).to(torch.float32)  # [B,512]

        B = x32.size(0)
        tot += B

        # Closed-form NBdiag + LDA on CPU float64 (paper-coherent)
        x64_cpu = x32.detach().cpu().to(torch.float64)
        y_cpu = y_dev.detach().cpu()

        nb_logits = logits_nbdiag_cpu(x64_cpu, mu64, var_diag64, logp64)
        lda_logits = logits_lda_cpu(x64_cpu, mu64, Sig_pool64, logp64, shrink=0.0)  # Sig_pool64 already shrunk outside if desired
        corr_nb += top1_correct(nb_logits, y_cpu)
        corr_lda += top1_correct(lda_logits, y_cpu)

        # QDAfull in x-space (GPU)
        if have_qda_x:
            q_logits = qda_x.logits(x32)  # [B,C] device
            corr_qda += top1_correct(q_logits, y_dev)

        # Fisher projection
        if have_fm or have_ph:
            V64 = V64_cpu.to(torch.float64)  # CPU float64
            zf = (x64_cpu @ V64).to(torch.float32).to(device, non_blocking=True)  # [B,kf]

        # FisherMix
        if have_fm:
            fm_logits = fm(zf, y=None)
            corr_fm += top1_correct(fm_logits, y_dev)

        # Proto-Hyper inference (student)
        if have_ph:
            base = lda_scores_f(zf, g_f)
            base_s = standardize_logits(base)
            ph_logits = base_s + ph(zf)
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
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config (e.g., configs/svhn.yaml)")
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    dataset_name: str = str(cfg["dataset"]).lower()
    alpha: float = float(cfg["dirichlet_alpha"])

    seed: int = int(cfg.get("seed", 123))
    data_root: str = str(cfg.get("data_root", "./data"))
    stats_root: str = str(cfg.get("stats_root", "./client_stats_X"))

    # Evaluation batch
    test_batch: int = int(cfg.get("test_batch", 256))

    # Shrinkage
    lda_shrink: float = float(cfg.get("lda_shrink", 0.05))
    qda_shrink: float = float(cfg.get("qda_shrink", 0.05))

    # Fisher
    fisher_energy: float = float(cfg.get("fisher_energy", 0.99))
    fisher_max_k: int = int(cfg.get("fisher_max_dim", 128))

    # Synthesis
    syn_ns_per_class: int = int(cfg.get("syn_ns_per_class", 20000))
    syn_shrink: float = float(cfg.get("syn_shrink", 0.0))
    syn_tau_base: float = float(cfg.get("syn_tau_base", 0.75))
    syn_tau_clip: Tuple[float, float] = tuple(cfg.get("syn_tau_clip", (0.60, 1.40)))  # type: ignore

    # FisherMix training
    fm_epochs: int = int(cfg.get("fm_epochs", 60))
    fm_batch: int = int(cfg.get("fm_batch", 1024))
    fm_lr: float = float(cfg.get("fm_lr", 1e-3))
    fm_wd: float = float(cfg.get("fm_wd", 5e-4))
    fm_scale: float = float(cfg.get("fm_scale", 20.0))
    fm_margin: float = float(cfg.get("fm_margin", 0.0))

    # ProtoHyper training
    ph_epochs: int = int(cfg.get("ph_epochs", 60))
    ph_batch: int = int(cfg.get("ph_batch", 1024))
    ph_lr: float = float(cfg.get("ph_lr", 1e-3))
    ph_wd: float = float(cfg.get("ph_wd", 5e-4))
    ph_rank: int = int(cfg.get("ph_rank", 128))
    kd_alpha: float = float(cfg.get("kd_alpha", 0.85))
    kd_T: float = float(cfg.get("kd_T", 4.0))
    teacher_blend: float = float(cfg.get("teacher_blend", 0.50))

    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_win = platform.system().lower().startswith("win")
    pin_memory = torch.cuda.is_available() and (not is_win)
    num_workers = int(cfg.get("num_workers", 2 if is_win else 4))

    tfm = build_transform()
    testset, num_classes = load_test_dataset(dataset_name, data_root, tfm)

    # Stats directory matches client naming
    alpha_tag = format_alpha(alpha)
    # We assume stats were generated with resnet18-IMAGENET1K_V1 (or fallback) tag in the folder name.
    # To keep it robust, we scan dataset folder and pick the first matching train/alpha folder if needed.
    stats_glob = os.path.join(stats_root, dataset_name.upper(), f"resnet18-*_TRAIN_A{alpha_tag}_X512")
    cand = sorted(glob.glob(stats_glob))
    if not cand:
        raise RuntimeError(f"No stats folder found matching: {stats_glob}")
    stats_dir = cand[0]

    paths = list_client_files(stats_dir)
    if not paths:
        raise RuntimeError(f"No client_*.pt found in: {stats_dir}")

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
    print(f"[INFO] synth         : ns/class={syn_ns_per_class} | shrink={syn_shrink} | tau={syn_tau_base} clip={syn_tau_clip}")
    print(f"[INFO] train FM      : epochs={fm_epochs} batch={fm_batch} lr={fm_lr} wd={fm_wd} s={fm_scale} m={fm_margin}")
    print(f"[INFO] train PH      : epochs={ph_epochs} batch={ph_batch} lr={ph_lr} wd={ph_wd} rank={ph_rank} KD={kd_alpha} T={kd_T} blend={teacher_blend}")
    print("=" * 90 + "\n")

    t0 = time.time()

    # Aggregate client stats
    agg = aggregate_clients_x(paths, num_classes=num_classes)
    mu = agg["mu"].to(torch.float64)                 # [C,d]
    pri = agg["pri"].to(torch.float64)               # [C]
    logp = agg["logp"].to(torch.float64)             # [C]
    Sig_pool = agg["Sigma_pool"].to(torch.float64)   # [d,d]
    Var_diag = agg["Var_diag"].to(torch.float64)     # [C,d]
    Sig_cls = agg["Sigma_cls"].to(torch.float64) if agg["Sigma_cls"] is not None else None

    # Apply LDA shrink now (so lda_scores_x can use shrink=0.0 during streaming)
    Sig_pool_shrunk = shrink_cov(Sig_pool, lda_shrink)
    Sig_pool_shrunk = symmetrize(Sig_pool_shrunk) + MIN_EPS * torch.eye(Sig_pool_shrunk.shape[0], dtype=torch.float64)

    # Build backbone (server-side FE)
    fe, weights_tag = build_resnet18_imagenet(device)

    # QDAfull x-space predictor (optional, but paper-full expects it if S exists)
    qda_x = None
    if Sig_cls is not None:
        print("[INFO] Precomputing QDAfull(x) Cholesky on GPU...")
        qda_x = QDAFullFastGPU(mu, Sig_cls, logp, shrink=qda_shrink, device=device)
    else:
        print("[WARN] S_per_class_x missing: GH-QDAfull(x) disabled; QDA_f teacher may be disabled too.")

    # Fisher subspace
    print("[INFO] Building Fisher subspace...")
    V, kf, lam = fisher_subspace(mu, pri, Sig_pool, lda_shrink=lda_shrink, energy=fisher_energy, max_k=fisher_max_k)
    V64_cpu = V.to(torch.float64).cpu()

    if lam.numel() > 0:
        top = [float(x) for x in lam[:min(10, lam.numel())].cpu()]
        print(f"[DBG] Fisher kf={kf} | top eigvals: {top}")
    else:
        print(f"[DBG] Fisher kf={kf} | no eigvals (degenerate SB/SW)")

    # Synthesis in Fisher
    print("[INFO] Synthesizing Fisher-space Gaussian data...")
    Zs_cpu, Ys_cpu, taus = synthesize_fisher(
        mu=mu,
        Sigma_pool=Sig_pool,
        V=V,
        ns_per_class=syn_ns_per_class,
        syn_shrink=syn_shrink,
        base_tau=syn_tau_base,
        tau_clip_rng=(float(syn_tau_clip[0]), float(syn_tau_clip[1])),
        seed=seed
    )
    print(f"[INFO] Synth total={Zs_cpu.shape[0]} | dim={Zs_cpu.shape[1]} | tau={taus[0]:.3f}")

    # Build Fisher-space Gaussians (base/teacher)
    print("[INFO] Building Fisher-space Gaussians (LDA_f + optional QDA_f)...")
    g_f = build_gaussians_fisher(
        mu=mu, pri=pri,
        Sigma_pool=Sig_pool,
        Sigma_cls=Sig_cls,   # if None => QDA_f disabled
        V=V,
        lda_shrink=lda_shrink,
        qda_shrink=qda_shrink,
        device=device
    )

    # Train FisherMix
    print("[INFO] Training FisherMix...")
    fm = train_fishermix(
        Zs_cpu=Zs_cpu, Ys_cpu=Ys_cpu,
        kf=kf, C=num_classes,
        device=device,
        epochs=fm_epochs, batch_size=fm_batch,
        lr=fm_lr, wd=fm_wd,
        scale=fm_scale, margin=fm_margin,
        is_win=is_win
    )
    fm.eval()

    # Train Proto-Hyper
    print("[INFO] Training Proto-Hyper...")
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

    # Test loader
    test_loader = DataLoader(
        testset,
        batch_size=test_batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    # Evaluate streaming
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

    # Print summary
    dt = time.time() - t0
    print("\n" + "=" * 90)
    print("FINAL SUMMARY (X-SPACE, FULL PIPELINE)")
    print("=" * 90)
    print(f"[INFO] dataset     : {dataset_name.upper()} | alpha={alpha} | clients={len(paths)}")
    print(f"[INFO] stats_dir   : {os.path.abspath(stats_dir)}")
    print(f"[INFO] backbone    : resnet18 ({weights_tag})")
    print(f"[INFO] eval_time   : {dt_eval/60.0:.2f} min | total_time={dt/60.0:.2f} min")
    print("-" * 90)
    print(f"GH-NBdiag   : {res['GH_NBdiag']:6.2f}%")
    print(f"GH-LDA      : {res['GH_LDA']:6.2f}%  (shrink={lda_shrink})")
    if res["GH_QDAfull"] is None:
        print("GH-QDAfull  :   n/a  (S_per_class_x missing)")
    else:
        print(f"GH-QDAfull  : {res['GH_QDAfull']:6.2f}%  (shrink={qda_shrink})")
    print("-" * 90)
    if res["FisherMix"] is None:
        print("FisherMix   :   n/a")
    else:
        print(f"FisherMix   : {res['FisherMix']:6.2f}%  (kf={kf}, s={fm_scale}, m={fm_margin})")
    if res["ProtoHyper"] is None:
        print("Proto-Hyper :   n/a")
    else:
        print(f"Proto-Hyper : {res['ProtoHyper']:6.2f}%  (kf={kf}, rank={min(ph_rank,kf)}, KD={kd_alpha}, T={kd_T}, blend={teacher_blend})")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()