#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Server-side STRONG — holdout pulito su CIFAR-100-C.

Usa client stats generati da CIFAR-100-C con severità {1,2,3,4} (o LOCO).
Valuta su:
- severità=5 di TUTTE le corruzioni (holdout) oppure
- severità=1..5 della corruzione lasciata fuori (LOCO), se nei meta è presente mode='loco:<corr>'.

Metodi: NB_paper, NB_diag, LDA, QDA_full, FisherMix, ProtoHyper (data-free in z).
"""

import os
import glob
import math
import warnings
import tarfile
import urllib.request
import shutil
import time
import platform
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# =======================
# Config
# =======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE_T = torch.device(DEVICE)
IS_WINDOWS = platform.system().lower().startswith("win")
PIN_MEMORY = torch.cuda.is_available() and (not IS_WINDOWS)
NUM_WORKERS = 2 if IS_WINDOWS else 4
PERSISTENT = not IS_WINDOWS
PREFETCH = 1 if IS_WINDOWS else 2

DATA_ROOT = "./data"
CIFARC_DIR = os.path.join(DATA_ROOT, "CIFAR-100-C")

# <<<<<< PUNTA ALL'OUTPUT DEL CLIENT HOLDOUT >>>>>>
STATS_ROOT = "./client_stats/CIFAR100C_HOLDOUT/resnet18-IMAGENET1K_V1"

TEST_BATCH = 512 if IS_WINDOWS else 1024
MIN_EPS = 1e-6
LDA_SHRINK = 0.05
QDA_SHRINK = 0.07

# Hyper sintetico / z
SYN_BATCH = 8192 if not IS_WINDOWS else 4096
FM_EPOCHS = 20
PH_EPOCHS = 20
LR = 1e-3
WD_FM = 2e-4
WD_PH = 5e-4

HYP = dict(
    FISHER_ENERGY=0.995,
    FISHER_MAX_K=128,
    SYN_NS_PER_CLASS=3000,  # 300k/epoca
    SYN_TAU=0.85,
    SYN_SHRINK=0.07,
    PH_RANK=128,
    USE_CLASS_COV=True,
    KD_T=2.0,
    KD_ALPHA=0.5,  # distillazione bilanciata
)

ALL_CORRUPTIONS = [
    "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost",
    "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression",
    "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter",
    "speckle_noise", "zoom_blur"
]

# =======================
# Misc
# =======================
warnings.filterwarnings("ignore", message="You are using torch.load with weights_only=False")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum or None for 'weights' are deprecated")
warnings.filterwarnings("ignore", message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", message="Anti-alias option is always applied for PIL Image input.")

try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

# =======================
# CIFAR-100-C Dataset (test)
# =======================
def ensure_cifar100c_present():
    if os.path.isdir(CIFARC_DIR):
        return
    url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"
    tmp_tar = os.path.join(DATA_ROOT, "CIFAR-100-C.tar")
    os.makedirs(DATA_ROOT, exist_ok=True)
    print("[INFO] CIFAR-100-C non trovato: scarico (≈1.3GB)...")
    urllib.request.urlretrieve(url, tmp_tar)  # nosec
    print("[INFO] Estrazione...")
    with tarfile.open(tmp_tar, "r") as tf:
        tf.extractall(DATA_ROOT)
    os.remove(tmp_tar)
    if not os.path.isdir(CIFARC_DIR):
        for d in os.listdir(DATA_ROOT):
            if d.lower().startswith("cifar-100-c"):
                shutil.move(os.path.join(DATA_ROOT, d), CIFARC_DIR)
                break


class CIFAR100C_Test(Dataset):
    def __init__(self, root: str, tfm, severities=(5,), corruptions=None):
        super().__init__()
        self.root = root
        self.tfm = tfm
        self.severities = list(severities)
        self.corruptions = list(corruptions) if corruptions is not None else ALL_CORRUPTIONS
        self.labels_all = np.load(os.path.join(root, "labels.npy"))
        self.items = []
        for corr in self.corruptions:
            arr = np.load(os.path.join(root, f"{corr}.npy"), mmap_mode="r")
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

    def __getitem__(self, i):
        c, p = self.items[i]
        arr = self._get_arr(c)
        img_np = arr[p].copy()
        img = torch.from_numpy(img_np).permute(2, 0, 1).contiguous().to(torch.float32) / 255.0
        if self.tfm is not None:
            img = self.tfm(img)
        y = int(self.labels_all[p])
        return img, y

# =======================
# I/O helpers (client stats)
# =======================
def safe_load_pt(p: str) -> Dict:
    try:
        return torch.load(p, map_location="cpu")
    except Exception as e:
        print(f"[WARN] skip {p}: {e}")
        return {}


def list_client_files() -> List[str]:
    files = sorted(glob.glob(os.path.join(STATS_ROOT, "client_*.pt")))
    out = []
    for p in files:
        d = safe_load_pt(p)
        if not d:
            continue
        if all(k in d for k in ["A_per_class", "N_per_class", "B_global"]):
            out.append(p)
        else:
            print(f"[WARN] file ignorato (mancano A/N/B): {p}")
    return out

# =======================
# Backbone & TFMs (ResNet18 strong)
# =======================
def build_resnet18_strong():
    try:
        from torchvision.models import ResNet18_Weights
        w = ResNet18_Weights.IMAGENET1K_V1
        m = models.resnet18(weights=w)
        mean = tuple(float(x) for x in w.meta["mean"])
        std = tuple(float(x) for x in w.meta["std"])
    except Exception:
        m = models.resnet18(pretrained=True)
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    m.fc = nn.Identity()
    m.eval().to(DEVICE_T)
    tfm = transforms.Compose([
        transforms.Normalize(mean=mean, std=std),
    ])
    return m, tfm

# =======================
# Gaussian / discriminant utils
# =======================
def shrink_cov(S: torch.Tensor, alpha: float = 0.05):
    d = S.shape[-1]
    tr = float(torch.trace(S))
    I = torch.eye(d, dtype=S.dtype)
    return (1.0 - alpha) * S + alpha * (tr / d) * I


def build_nb_paper_from_ABN(A: torch.Tensor, N: torch.Tensor, B: torch.Tensor):
    A = A.to(torch.float64)
    N = N.to(torch.long)
    B = B.to(torch.float64)
    C, d = A.shape
    N_total = int(N.sum().item())
    mu = A / N.clamp_min(1).unsqueeze(1)
    pri = (N / float(N_total)).to(torch.float64)
    logp = torch.log(pri.clamp_min(1e-12))
    trB = torch.trace(B).item()
    mu_sq = (mu.pow(2).sum(dim=1) * N.to(torch.float64)).sum().item()
    W = max(0.0, trB - mu_sq)
    sigma2 = W / (max(1, (N_total - C)) * float(d))
    sigma2 = float(max(sigma2, 1e-12))
    return mu, logp, sigma2


def logpdf_gauss_spherical(x, mu, sigma2, log_prior):
    x = x.to(torch.float64)
    mu = mu.to(torch.float64)
    log_prior = log_prior.to(torch.float64)
    n, d = x.shape
    inv2 = 0.5 / float(sigma2)
    x2 = (x * x).sum(dim=1)
    mu2 = (mu * mu).sum(dim=1)
    x_dot_mu = x @ mu.t()
    quad = inv2 * (x2.unsqueeze(1) - 2.0 * x_dot_mu + mu2.unsqueeze(0))
    logdet = 0.5 * d * math.log(float(sigma2) + 1e-12)
    return log_prior.unsqueeze(0) - quad - logdet


def logpdf_gauss_diag(x, mu, var_diag, log_prior):
    x = x.to(torch.float64)
    mu = mu.to(torch.float64)
    inv = (1.0 / var_diag.to(torch.float64))
    x2_inv = x.pow(2) @ inv
    mu_inv = mu * inv
    mu2_inv = (mu * mu_inv).sum(dim=1)
    x_dot_mu_inv = x @ mu_inv.t()
    quad = 0.5 * (x2_inv.unsqueeze(1) - 2.0 * x_dot_mu_inv + mu2_inv.unsqueeze(0))
    logdet = 0.5 * torch.log(var_diag.to(torch.float64)).sum()
    return log_prior.to(torch.float64).unsqueeze(0) - quad - logdet


def lda_scores(x, mu, Sigma_pool, log_prior, alpha=LDA_SHRINK):
    d = Sigma_pool.shape[0]
    S = shrink_cov(Sigma_pool, alpha=alpha) + MIN_EPS * torch.eye(d, dtype=Sigma_pool.dtype)
    W = torch.linalg.solve(S, mu.t())
    wx = x @ W
    b = -0.5 * (mu * W.t()).sum(dim=1) + log_prior
    return wx + b


def qda_precompute(Sigma_cls, alpha=QDA_SHRINK, eps=1e-6):
    C, d, _ = Sigma_cls.shape
    L_list = []
    logdet = []
    I = torch.eye(d, dtype=Sigma_cls.dtype)
    for j in range(C):
        Sj = shrink_cov(Sigma_cls[j], alpha=alpha) + eps * I
        L = torch.linalg.cholesky(Sj)
        L_list.append(L)
        logdet.append(torch.log(torch.diag(L)).sum().item())
    return L_list, torch.tensor(logdet, dtype=torch.float64)


@torch.no_grad()
def qda_predict_batch(Xb, mu, L_list, logdet, log_prior):
    Xb = Xb.to(torch.float64)
    mu = mu.to(torch.float64)
    log_prior = log_prior.to(torch.float64)
    m = Xb.shape[0]
    C = mu.shape[0]
    scores = torch.empty(m, C, dtype=torch.float64)
    for j, L in enumerate(L_list):
        dif = Xb - mu[j].unsqueeze(0)
        Yt = torch.cholesky_solve(dif.t(), L)
        y = Yt.t()
        quad = 0.5 * (dif * y).sum(dim=1)
        scores[:, j] = (log_prior[j] - quad - logdet[j])
    return scores.argmax(dim=1)

# =======================
# Fisher subspace & synth in z
# =======================
def fisher_subspace_by_energy(mu, priors, Sigma_pool, alpha=0.05, energy=0.99, max_k=128):
    C, d = mu.shape
    mbar = (priors.unsqueeze(1) * mu).sum(dim=0, keepdim=True).to(torch.float64)
    Sb = torch.zeros(d, d, dtype=torch.float64)
    for j in range(C):
        dm = (mu[j:j+1].to(torch.float64) - mbar)
        Sb += float(priors[j].item()) * (dm.t() @ dm)
    Sw = Sigma_pool.to(torch.float64)
    tr = torch.trace(Sw).item()
    Sw = (1.0 - alpha) * Sw + alpha * (tr / d) * torch.eye(d, dtype=torch.float64)
    M = torch.linalg.solve(Sw, Sb)
    evals, evecs = torch.linalg.eig(M)
    evals = evals.real.clamp_min(0)
    evecs = evecs.real
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]
    if evals.numel() == 0 or float(evals.sum().item()) <= 0:
        k = min(2, d)
        V = evecs[:, :k] if evecs.numel() > 0 else torch.eye(d, k, dtype=torch.float64)
        return V.contiguous(), k
    csum = torch.cumsum(evals, dim=0)
    tot = float(csum[-1].item())
    k = int((csum / max(tot, 1e-12) >= energy).nonzero(as_tuple=True)[0][0].item() + 1)
    k = max(2, min(k, max_k, d))
    V = evecs[:, :k].contiguous()
    return V, k


def prepare_gaussian_z_params(mu, Sigma_pool, Sigma_cls, priors, V,
                              use_class_cov=True, shrink_alpha=0.05, tau=0.85):
    C, d = mu.shape
    k = V.shape[1]
    V64 = V.to(torch.float64)
    mu_z = (mu.to(torch.float64) @ V64)
    if use_class_cov and Sigma_cls is not None:
        L_list = []
        logdet = []
        for j in range(C):
            Sj = Sigma_cls[j].to(torch.float64)
            tr = torch.trace(Sj).item()
            Sj = (1.0 - shrink_alpha) * Sj + shrink_alpha * (tr / d) * torch.eye(d, dtype=torch.float64)
            Sz = V64.t() @ Sj @ V64
            Sz = (tau ** 2) * Sz + 1e-6 * torch.eye(k, dtype=torch.float64)
            Lj = torch.linalg.cholesky(Sz)
            L_list.append(Lj)
            logdet.append(torch.log(torch.diag(Lj)).sum().item())
    else:
        Sw = Sigma_pool.to(torch.float64)
        tr = torch.trace(Sw).item()
        Sw = (1.0 - shrink_alpha) * Sw + shrink_alpha * (tr / d) * torch.eye(d, dtype=torch.float64)
        Sz = V64.t() @ Sw @ V64
        Sz = (tau ** 2) * Sz + 1e-6 * torch.eye(k, dtype=torch.float64)
        Lp = torch.linalg.cholesky(Sz)
        L_list = [Lp for _ in range(C)]
        logdet = [torch.log(torch.diag(Lp)).sum().item() for _ in range(C)]

    # pooled in z per LDA_z
    Sz_pool = V64.t() @ Sigma_pool.to(torch.float64) @ V64
    Sz_pool = shrink_cov(Sz_pool, alpha=shrink_alpha) + 1e-6 * torch.eye(k, dtype=torch.float64)
    return mu_z.contiguous(), L_list, torch.tensor(logdet, dtype=torch.float64), Sz_pool


class SynthGaussianZ(Dataset):
    def __init__(self, mu_z: torch.Tensor, L_list: List[torch.Tensor], ns_per_class: int, seed: int = 123):
        super().__init__()
        self.mu_z = [m.clone().to(torch.float32) for m in mu_z]
        self.L = [L.clone().to(torch.float32) for L in L_list]
        self.C = len(self.mu_z)
        self.k = self.mu_z[0].numel()
        self.ns_per_class = int(ns_per_class)
        self.total = self.C * self.ns_per_class
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        j = idx % self.C
        eps = torch.randn(self.k, generator=self.gen)
        z = self.mu_z[j] + (eps @ self.L[j].t())
        y = j
        return z, y

# =======================
# Heads (semplici e stabili)
# =======================
class LogisticHead(nn.Module):
    def __init__(self, in_dim, C):
        super().__init__()
        self.W = nn.Linear(in_dim, C, bias=True)

    def forward(self, z):
        return self.W(z)


class ProtoHyperResidual(nn.Module):
    def __init__(self, in_dim, C, rank=128):
        super().__init__()
        self.U = nn.Linear(in_dim, rank, bias=False)
        self.V = nn.Linear(rank, C, bias=True)

    def forward(self, z):
        return self.V(self.U(z))

# =======================
# Evaluate (streaming, STRONG only)
# =======================
@torch.no_grad()
def evaluate_all(
    fe, tfm, agg, fm_model, ph_model, nbz_logits_fn, V64: torch.Tensor,
    L_full: Optional[List[torch.Tensor]], logdet_full: Optional[torch.Tensor],
    severities: List[int], corruptions: List[str]
) -> Dict[str, float]:
    testset = CIFAR100C_Test(CIFARC_DIR, tfm=tfm, severities=tuple(severities), corruptions=corruptions)
    dl = DataLoader(
        testset, batch_size=TEST_BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT, prefetch_factor=PREFETCH
    )

    mu, pri, Sigma_pool, Sigma_cls = agg["mu"], agg["priors"], agg["Sigma_pool"], agg["Sigma_cls"]
    var_diag = torch.diag(Sigma_pool).clamp_min(1e-6)
    logpri = torch.log(pri.clamp_min(1e-12))
    mu_nb, logp_nb, s2_nb = build_nb_paper_from_ABN(agg["A"], agg["N"], agg["B"])

    have_qda = (Sigma_cls is not None) and (L_full is not None) and (logdet_full is not None)

    tot = 0
    corr_nb = 0
    corr_nbd = 0
    corr_lda = 0
    corr_qda = 0 if have_qda else None
    corr_fm = 0
    corr_ph = 0

    start = time.time()
    for imgs, y in dl:
        imgs = imgs.to(DEVICE_T, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            imgs = F.interpolate(imgs, size=224, mode="bilinear", align_corners=False)
            X = fe(imgs)

        X64 = X.detach().cpu().to(torch.float64)
        y = y.to("cpu")
        bsz = X64.size(0)

        # NB_paper (sferico)
        pred_nbp = logpdf_gauss_spherical(X64, mu_nb, s2_nb, logp_nb).argmax(1)
        corr_nb += int((pred_nbp == y).sum())

        # NB_diag (diag globale)
        pred_nbd = logpdf_gauss_diag(X64, mu, var_diag, logpri).argmax(1)
        corr_nbd += int((pred_nbd == y).sum())

        # LDA
        pred_lda = lda_scores(X64, mu, Sigma_pool, logpri, alpha=LDA_SHRINK).argmax(1)
        corr_lda += int((pred_lda == y).sum())

        # QDA_full (se disponibile)
        if have_qda:
            pred_qda = qda_predict_batch(X64, mu, L_full, logdet_full, logpri)
            corr_qda += int((pred_qda == y).sum())

        # Proiezione in Fisher e teste trainabili
        Z = (X64 @ V64).to(torch.float32).to(DEVICE_T)

        # FisherMix
        with torch.no_grad():
            fm_logits = fm_model(Z)
        pred_fm = fm_logits.argmax(1).cpu()
        corr_fm += int((pred_fm == y).sum())

        # ProtoHyper = base (NB in z) + residuo
        base = nbz_logits_fn(Z)
        ph_logits = base + ph_model(Z)
        pred_ph = ph_logits.argmax(1).cpu()
        corr_ph += int((pred_ph == y).sum())

        tot += bsz

    print(f"[INFO] Eval completata su {tot} esempi in {(time.time() - start) / 60:.1f} min.")

    res = dict(
        NB_paper=100.0 * corr_nb / float(tot),
        NB_diag=100.0 * corr_nbd / float(tot),
        LDA=100.0 * corr_lda / float(tot),
        QDA_full=None if corr_qda is None else 100.0 * corr_qda / float(tot),
        FisherMix=100.0 * corr_fm / float(tot),
        ProtoHyper=100.0 * corr_ph / float(tot),
    )
    return res


def print_summary(title: str, res: Dict[str, float]):
    print(f"\n================ {title} ================")
    for k in ["NB_paper", "NB_diag", "LDA", "QDA_full", "FisherMix", "ProtoHyper"]:
        v = res.get(k, None)
        print(f"{k:11s}: {'n/a' if v is None else f'{v:6.2f}%'}")
    print("=========================================")

# =======================
# z-logits helper (DEVICE)
# =======================
def nb_logits_in_z_dev_fn(mu_z_64: torch.Tensor, sigma2_z: float, log_prior_64: torch.Tensor):
    mu = mu_z_64.to(device=DEVICE_T, dtype=torch.float32)
    logp = log_prior_64.to(device=DEVICE_T, dtype=torch.float32)
    k = mu.shape[1]
    logdet = 0.5 * k * math.log(float(sigma2_z) + 1e-12)
    inv2 = 0.5 / float(sigma2_z)
    mu2 = (mu * mu).sum(dim=1)
    mu_t = mu.t().contiguous()

    def _fn(Z32: torch.Tensor) -> torch.Tensor:
        Z = Z32.to(device=DEVICE_T, dtype=torch.float32)
        x2 = (Z * Z).sum(dim=1)
        xmu = Z @ mu_t
        quad = inv2 * (x2.unsqueeze(1) - 2.0 * xmu + mu2.unsqueeze(0))
        return (logp.unsqueeze(0) - quad - logdet)

    return _fn

# =======================
# Main
# =======================
def main():
    # Verifica/Download dataset
    ensure_cifar100c_present()

    # Carica client stats
    paths = list_client_files()
    if not paths:
        print(f"[ERR] Nessun file client in {STATS_ROOT}")
        return

    # aggrega
    agg = aggregate_from_clients(paths)
    C, d, N_total = agg["C"], agg["d"], agg["N_total"]
    meta = agg["meta"]
    mode = meta.get("mode", "holdout_s5")
    train_sev = meta.get("severities_train", (1, 2, 3, 4))
    train_corrs = list(meta.get("corruptions_included", ALL_CORRUPTIONS))

    print(f"[INFO] trovati {len(paths)} file client in {STATS_ROOT}")
    print(f"[INFO] C={C} classi, d={d} dim, N_total={N_total} | train severities={train_sev} | mode={mode}")

    # backbone + tfm
    fe, tfm = build_resnet18_strong()

    # Proiezione di Fisher + params in z
    V, k = fisher_subspace_by_energy(
        agg["mu"], agg["priors"], agg["Sigma_pool"],
        alpha=LDA_SHRINK, energy=HYP["FISHER_ENERGY"], max_k=HYP["FISHER_MAX_K"]
    )
    V64 = V.to(torch.float64)
    print(f"[INFO] Fisher subspace: k={k}")

    mu_z, L_list, logdet_z, Sz_pool = prepare_gaussian_z_params(
        agg["mu"], agg["Sigma_pool"], agg["Sigma_cls"], agg["priors"], V,
        use_class_cov=HYP["USE_CLASS_COV"], shrink_alpha=HYP["SYN_SHRINK"], tau=HYP["SYN_TAU"]
    )
    kz = V64.shape[1]
    sigma2_z = float(torch.trace(V64.t() @ agg["Sigma_pool"].to(torch.float64) @ V64).item() / float(kz))
    logpri = torch.log(agg["priors"].clamp_min(1e-12))

    # z logits helpers
    nbz_logits_fn = nb_logits_in_z_dev_fn(mu_z, sigma2_z, logpri)

    # logistic (FisherMix)
    fm = LogisticHead(kz, C).to(DEVICE_T)
    opt_fm = torch.optim.AdamW(fm.parameters(), lr=LR, weight_decay=WD_FM)

    # residual (ProtoHyper)
    ph = ProtoHyperResidual(kz, C, rank=min(HYP["PH_RANK"], kz)).to(DEVICE_T)
    opt_ph = torch.optim.AdamW(ph.parameters(), lr=LR, weight_decay=WD_PH)

    ce = nn.CrossEntropyLoss()
    kl = nn.KLDivLoss(reduction="batchmean")
    T = float(HYP["KD_T"])
    alpha = float(HYP["KD_ALPHA"])

    # teacher ensemble in z (stabile: 0.7*LDAz + 0.3*NBz)
    # costruiamo logits LDA_z via soluzione chiusa
    mu_z64 = mu_z.to(torch.float64)
    Sz = Sz_pool.to(torch.float64)
    W = torch.linalg.solve(Sz, mu_z64.t())
    b = (-0.5 * (mu_z64 * (W.t())).sum(dim=1) + logpri).to(torch.float32)
    W32 = W.to(torch.float32).to(DEVICE_T)
    b = b.to(DEVICE_T)

    def ldaz(Z32: torch.Tensor) -> torch.Tensor:
        Z = Z32.to(DEVICE_T)
        return Z @ W32 + b

    # dataset sintetico in z
    syn_ds = SynthGaussianZ(mu_z, L_list, ns_per_class=HYP["SYN_NS_PER_CLASS"], seed=123)
    syn_dl = DataLoader(syn_ds, batch_size=SYN_BATCH, shuffle=False, drop_last=False, num_workers=0, pin_memory=False)

    # ---- Train FisherMix ----
    steps = math.ceil(len(syn_ds) / max(1, SYN_BATCH))
    for ep in range(1, FM_EPOCHS + 1):
        fm.train()
        tot = 0.0
        seen = 0
        it = iter(syn_dl)
        for _ in range(steps):
            z, yb = next(it)
            z = z.to(DEVICE_T, non_blocking=True)
            yb = yb.to(DEVICE_T, non_blocking=True)
            opt_fm.zero_grad(set_to_none=True)
            logits = fm(z)
            loss = ce(logits, yb)
            loss.backward()
            opt_fm.step()
            bs = z.size(0)
            tot += loss.item() * bs
            seen += bs
        if ep % 5 == 0:
            print(f"[FM] ep {ep:02d}/{FM_EPOCHS} | loss={tot / seen:.4f}")

    # ---- Train ProtoHyper (residuo) con KD (0.7*LDAz + 0.3*NBz) ----
    for ep in range(1, PH_EPOCHS + 1):
        ph.train()
        tot = 0.0
        seen = 0
        it = iter(syn_dl)
        for _ in range(steps):
            z, yb = next(it)
            z = z.to(DEVICE_T, non_blocking=True)
            yb = yb.to(DEVICE_T, non_blocking=True)
            with torch.no_grad():
                base = nbz_logits_fn(z)
                ldaL = ldaz(z)
                teacher = 0.7 * ldaL + 0.3 * base
            logits = base + ph(z)
            loss_ce = ce(logits, yb)
            loss_kd = (T * T) * F.kl_div((logits / T).log_softmax(dim=1), (teacher / T).softmax(dim=1), reduction="batchmean")
            loss = alpha * loss_kd + (1.0 - alpha) * loss_ce
            opt_ph.zero_grad(set_to_none=True)
            loss.backward()
            opt_ph.step()
            bs = z.size(0)
            tot += loss.item() * bs
            seen += bs
        if ep % 5 == 0:
            print(f"[PH] ep {ep:02d}/{PH_EPOCHS} | loss={tot / seen:.4f}")

    # QDA_full in x (precompute)
    L_full = logdet_full = None
    if agg["Sigma_cls"] is not None:
        L_full, logdet_full = qda_precompute(agg["Sigma_cls"], alpha=QDA_SHRINK)

    # Target di valutazione:
    if isinstance(mode, str) and mode.startswith("loco:"):
        # LOCO: test su TUTTE le severità (1..5) della corruzione lasciata fuori
        left_out = mode.split(":", 1)[1]
        assert left_out in ALL_CORRUPTIONS
        test_corrs = [left_out]
        test_sev = [1, 2, 3, 4, 5]
        title = f"FINAL SUMMARY | CIFAR-100-C (LOCO: {left_out}, severities 1–5) | STRONG"
    else:
        # Holdout severità 5 su tutte le corruzioni
        test_corrs = ALL_CORRUPTIONS
        test_sev = [5]
        title = "FINAL SUMMARY | CIFAR-100-C (ALL corruptions, severity 5) | STRONG"

    # Eval
    res = evaluate_all(
        fe, tfm, agg,
        fm_model=fm, ph_model=ph, nbz_logits_fn=nbz_logits_fn,
        V64=V64, L_full=L_full, logdet_full=logdet_full,
        severities=test_sev, corruptions=test_corrs
    )
    print_summary(title, res)

# ---- aggregate_from_clients (riuso) ----
def aggregate_from_clients(paths: List[str]):
    meta0 = None
    A_sum = N_sum = B_sum = S_sum = None

    for p in paths:
        d = safe_load_pt(p)
        if not d:
            continue
        meta = d["meta"]
        if meta0 is None:
            meta0 = meta

        A = d["A_per_class"].to(torch.float64)
        N = d["N_per_class"].to(torch.long)
        B = d["B_global"].to(torch.float64)
        S = d.get("S_per_class", None)

        if A_sum is None:
            A_sum = torch.zeros_like(A, dtype=torch.float64)
            N_sum = torch.zeros_like(N, dtype=torch.long)
            B_sum = torch.zeros_like(B, dtype=torch.float64)
            S_sum = None if S is None else torch.zeros_like(S, dtype=torch.float64)

        A_sum += A
        N_sum += N
        B_sum += B

        if S is not None:
            if S_sum is None:
                S_sum = torch.zeros_like(S, dtype=torch.float64)
            S_sum += S.to(torch.float64)

    assert meta0 is not None, "Nessun file valido"

    C, d = A_sum.shape
    N_total = int(N_sum.sum().item())
    mu = A_sum / N_sum.clamp_min(1).unsqueeze(1)
    priors = (N_sum / float(N_total)).to(torch.float64)

    if S_sum is not None:
        SW = torch.zeros(d, d, dtype=torch.float64)
        for j in range(C):
            Nj = int(N_sum[j].item())
            if Nj > 0:
                Aj = A_sum[j].unsqueeze(1)
                Sj = S_sum[j]
                SW += (Sj - (Aj @ Aj.t()) / float(Nj))
        Sigma_pool = SW / float(max(1, N_total - C))
    else:
        m_all = (A_sum.sum(dim=0) / N_total).unsqueeze(1)
        Sigma_pool = (B_sum - (m_all @ m_all.t()) * N_total) / max(1, N_total - 1)

    Sigma_cls = None
    if S_sum is not None:
        Sigma_cls = torch.zeros(C, d, d, dtype=torch.float64)
        for j in range(C):
            Nj = int(N_sum[j].item())
            if Nj > 1:
                Aj = A_sum[j].unsqueeze(1)
                Sj = S_sum[j]
                Sigma_cls[j] = (Sj - (Aj @ Aj.t()) / float(Nj)) / float(max(1, Nj - 1))
            else:
                Sigma_cls[j] = Sigma_pool.clone()

    return dict(
        meta=meta0, A=A_sum, N=N_sum, B=B_sum, mu=mu, priors=priors,
        Sigma_pool=Sigma_pool, Sigma_cls=Sigma_cls, C=C, d=d, N_total=N_total
    )


if __name__ == "__main__":
    main()









