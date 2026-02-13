#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Server-side (ResNet18 only): baseline paper (weak/strong) + NB_diag/LDA/QDA + FisherMix + ProtoHyper.
Legge i client da:
  ./client_stats/resnet18-IMAGENET1K_V1/client_*.pt

Stima la testa Naive Gaussian dal solo A/B/N (nessun training) e valuta su CIFAR-10 test.

Pipeline:
- WEAK : Resize(224) + Normalize(0.5,0.5,0.5) -> NB_paper ~63%
- STRONG: Resize(224) + Normalize(ImageNet)

Se S_per_class è presente nei file client:
- Abilita QDA_full (cov per-classe)
- Usa Σ_j anche per i sintetici nel sottospazio Fisher.

Stampa tutto a terminale (niente file).
"""

import os, glob, math, warnings
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models

# =======================
# Config & Hyperparams
# =======================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = "./data"
STATS_ROOT = "./client_stats/resnet18-IMAGENET1K_V1"

BATCH_SIZE = 256
NUM_WORKERS = 2
PIN_MEMORY = torch.cuda.is_available()

# QDA / LDA
MIN_EPS = 1e-6
LDA_SHRINK = 0.05
QDA_SHRINK = 0.05

# Advanced (FisherMix / Proto-Hyper)
FISHER_ENERGY = 0.99
FISHER_MAX_K = 128
SYN_NS_PER_CLASS = 20000  # sintetici per classe (alza per spremere accuracy)
SYN_SHRINK = 0.05         # shrink cov in z
SYN_TAU = 0.85            # temperatura (0.7-0.9)
USE_CLASS_COV = True      # usa Σ_j (se disponibile) per sintetici

FM_EPOCHS = 60
PH_EPOCHS = 60
LR = 1e-3
WD = 5e-4
PH_RANK = 128             # rank residuo proto-hyper (≤ k)

# =======================
# Warnings cleanup
# =======================
warnings.filterwarnings(
    "ignore",
    message="You are using torch.load with weights_only=False",
)
warnings.filterwarnings(
    "ignore",
    message="Anti-alias option is always applied for PIL Image input. Argument antialias is ignored.",
)
warnings.filterwarnings(
    "ignore",
    message="Arguments other than a weight enum or None for 'weights' are deprecated since 0.13",
)
warnings.filterwarnings(
    "ignore",
    message="The parameter 'pretrained' is deprecated since 0.13",
)

# =======================
# I/O helpers
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
        ok = all(k in d for k in ["A_per_class", "N_per_class", "B_global"])
        if not ok:
            print(f"[WARN] file ignorato (mancano A/N/B): {p}")
            continue
        out.append(p)
    return out

# =======================
# Backbones & TFMs
# =======================
def build_resnet18_weak():
    """Resize 224 + Normalize(0.5). Serve a replicare ~63% con NB_paper."""
    try:
        from torchvision.models import ResNet18_Weights
        weights = ResNet18_Weights.IMAGENET1K_V1
        m = models.resnet18(weights=weights)
    except Exception:
        m = models.resnet18(pretrained=True)
    m.fc = nn.Identity()
    m.eval().to(DEVICE)

    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return m, tfm, mean, std

def build_resnet18_strong():
    """Resize 224 + Normalize(ImageNet mean/std)."""
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
    m.eval().to(DEVICE)

    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return m, tfm, mean, std

@torch.no_grad()
def extract_features(loader, fe: nn.Module):
    X, Y = [], []
    for imgs, y in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        z = fe(imgs)
        X.append(z.cpu())
        Y.append(y.clone())
    return torch.cat(X, dim=0), torch.cat(Y, dim=0)

# =======================
# Aggregazione A/N/B(/S)
# =======================
def aggregate_from_clients(paths: List[str]):
    meta0 = None
    A_sum = None
    N_sum = None
    B_sum = None
    S_sum = None

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

    # Sigma_pool
    if S_sum is not None:
        SW = torch.zeros(d, d, dtype=torch.float64)
        for j in range(C):
            Nj = int(N_sum[j].item())
            if Nj > 0:
                Aj = A_sum[j].unsqueeze(1)
                Sj = S_sum[j]
                SW += (Sj - (Aj @ Aj.t()) / float(Nj))
        denom = max(1, N_total - C)
        Sigma_pool = SW / float(denom)
    else:
        # fallback totale
        m_all = (A_sum.sum(dim=0)/N_total).unsqueeze(1)  # [d,1]
        Sigma_pool = (B_sum - (m_all @ m_all.t()) * N_total) / max(1, N_total-1)

    # Sigma per-classe (se S disponibile)
    Sigma_cls = None
    if S_sum is not None:
        Sigma_cls = torch.zeros(C, d, d, dtype=torch.float64)
        for j in range(C):
            Nj = int(N_sum[j].item())
            if Nj > 1:
                Aj = A_sum[j].unsqueeze(1)
                Sj = S_sum[j]
                Sigma_cls[j] = (Sj - (Aj @ Aj.t())/float(Nj)) / float(max(1, Nj-1))
            else:
                Sigma_cls[j] = Sigma_pool.clone()

    return dict(
        meta=meta0,
        A=A_sum,
        N=N_sum,
        B=B_sum,
        mu=mu,
        priors=priors,
        Sigma_pool=Sigma_pool,
        Sigma_cls=Sigma_cls,
        C=C,
        d=d,
        N_total=N_total
    )

# =======================
# Gaussian scores
# =======================
def shrink_cov(S: torch.Tensor, alpha: float = 0.05):
    d = S.shape[-1]
    tr = float(torch.trace(S))
    I = torch.eye(d, dtype=S.dtype)
    return (1.0 - alpha) * S + alpha * (tr / d) * I

def logpdf_gauss_spherical(x: torch.Tensor, mu: torch.Tensor, sigma2: float, log_prior: torch.Tensor):
    dif = x.unsqueeze(1) - mu.unsqueeze(0)
    quad = 0.5 * dif.pow(2).sum(dim=2) / float(sigma2)
    logdet = 0.5 * mu.shape[1] * math.log(float(sigma2) + 1e-12)
    return log_prior.unsqueeze(0) - quad - logdet

def logpdf_gauss_diag(x: torch.Tensor, mu: torch.Tensor, var_diag: torch.Tensor, log_prior: torch.Tensor):
    inv = 1.0 / var_diag
    dif = x.unsqueeze(1) - mu.unsqueeze(0)
    quad = 0.5 * (dif * dif * inv).sum(dim=2)
    logdet = 0.5 * torch.log(var_diag).sum().item()
    return log_prior.unsqueeze(0) - quad - logdet

def lda_scores(x, mu, Sigma_pool, log_prior, alpha=LDA_SHRINK):
    S = shrink_cov(Sigma_pool, alpha=alpha) + MIN_EPS * torch.eye(Sigma_pool.shape[0], dtype=Sigma_pool.dtype)
    W = torch.linalg.solve(S, mu.t())
    wx = x @ W
    b = -0.5 * (mu * W.t()).sum(dim=1) + log_prior
    return wx + b

def qda_scores(x, mu, Sigma_cls, log_prior, alpha=QDA_SHRINK):
    C, d, _ = Sigma_cls.shape
    scores = []
    I = torch.eye(d, dtype=Sigma_cls.dtype)
    for j in range(C):
        Sj = shrink_cov(Sigma_cls[j], alpha=alpha) + MIN_EPS * I
        L = torch.linalg.cholesky(Sj)
        dif = (x - mu[j].unsqueeze(0))
        y = torch.cholesky_solve(dif.unsqueeze(2), L).squeeze(2)
        quad = 0.5 * (dif * y).sum(dim=1)
        logdet = torch.log(torch.diag(L)).sum().item()
        scores.append((log_prior[j] - quad - logdet).unsqueeze(1))
    return torch.cat(scores, dim=1)

# =======================
# NB head from A/B/N
# =======================
def build_nb_paper_from_ABN(A: torch.Tensor, N: torch.Tensor, B: torch.Tensor):
    A = A.to(torch.float64)
    N = N.to(torch.long)
    B = B.to(torch.float64)

    C, d = A.shape
    N_total = int(N.sum().item())

    mu = A / N.clamp_min(1).unsqueeze(1)
    priors = (N / float(N_total)).to(torch.float64)
    log_prior = torch.log(priors.clamp_min(1e-12))

    trB = torch.trace(B).item()
    mu_sq = (mu.pow(2).sum(dim=1) * N.to(torch.float64)).sum().item()
    W = max(0.0, trB - mu_sq)
    sigma2 = W / (max(1, (N_total - C)) * float(d))
    sigma2 = float(max(sigma2, 1e-12))

    return mu, log_prior, sigma2

# =======================
# Fisher subspace & Synth
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

def synth_gaussian_in_z(mu, Sigma_pool, Sigma_cls, priors, V,
                        ns_per_class=10000,
                        use_class_cov=True,
                        shrink_alpha=0.05,
                        tau=0.85,
                        seed=123):
    torch.manual_seed(seed)

    C, d = mu.shape
    k = V.shape[1]
    V64 = V.to(torch.float64)
    mu_z = (mu.to(torch.float64) @ V64)

    if use_class_cov and Sigma_cls is not None:
        S_list = []
        for j in range(C):
            Sj = Sigma_cls[j].to(torch.float64)
            tr = torch.trace(Sj).item()
            Sj = (1.0 - shrink_alpha) * Sj + shrink_alpha * (tr / d) * torch.eye(d, dtype=torch.float64)
            Sz = V64.t() @ Sj @ V64
            Sz = (tau ** 2) * Sz + 1e-6 * torch.eye(k, dtype=torch.float64)
            S_list.append(Sz)
    else:
        Sw = Sigma_pool.to(torch.float64)
        tr = torch.trace(Sw).item()
        Sw = (1.0 - shrink_alpha) * Sw + shrink_alpha * (tr / d) * torch.eye(d, dtype=torch.float64)
        Sz_pool = V64.t() @ Sw @ V64
        Sz_pool = (tau ** 2) * Sz_pool + 1e-6 * torch.eye(k, dtype=torch.float64)
        S_list = [Sz_pool for _ in range(C)]

    Zs, Ys = [], []
    for j in range(C):
        n = int(ns_per_class)
        Lz = torch.linalg.cholesky(S_list[j])
        z = mu_z[j].unsqueeze(0) + torch.randn(n, k, dtype=torch.float64) @ Lz.t()
        y = torch.full((n,), j, dtype=torch.long)
        Zs.append(z.to(torch.float32))
        Ys.append(y)

    return torch.cat(Zs, dim=0), torch.cat(Ys, dim=0)

# =======================
# Heads for advanced
# =======================
class LogisticHead(nn.Module):
    def __init__(self, in_dim, C):
        super().__init__()
        self.W = nn.Linear(in_dim, C, bias=True)

    def forward(self, z):
        return self.W(z)

class ProtoHyperResidual(nn.Module):
    def __init__(self, in_dim, C, rank=64):
        super().__init__()
        self.U = nn.Linear(in_dim, rank, bias=False)
        self.V = nn.Linear(rank, C, bias=True)

    def forward(self, z):
        return self.V(self.U(z))

# =======================
# Eval pipelines
# =======================
def eval_weak_pipeline(paths: List[str], agg: Dict):
    print("\n[START] device={} | backbone=resnet18 | pipeline=weak".format(DEVICE))

    fe_w, tfm_w, _, _ = build_resnet18_weak()
    testset = datasets.CIFAR10(root=DATA_ROOT, train=False, transform=tfm_w, download=True)
    loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    X, y = extract_features(loader, fe_w)
    X64 = X.to(torch.float64)

    print("\n[RESULTS per-client — NB_paper (weak)]")
    print("{:<22} {:>8} {:>12} {:>8}".format("client_file","N_total","sigma2","Top-1"))
    print("-"*22 + " " + "-"*8 + " " + "-"*12 + " " + "-"*8)

    A_agg = torch.zeros_like(agg["A"], dtype=torch.float64)
    N_agg = torch.zeros_like(agg["N"], dtype=torch.long)
    B_agg = torch.zeros_like(agg["B"], dtype=torch.float64)

    for p in paths:
        dct = safe_load_pt(p)
        A = dct["A_per_class"]; N = dct["N_per_class"]; B = dct["B_global"]
        A_agg += A.to(torch.float64)
        N_agg += N.to(torch.long)
        B_agg += B.to(torch.float64)

        mu, logp, s2 = build_nb_paper_from_ABN(A, N, B)
        pred = logpdf_gauss_spherical(X64, mu, s2, logp).argmax(dim=1).cpu()
        acc = (pred == y).float().mean().item() * 100.0

        print("{:<22} {:>8d} {:>12.6e} {:>7.2f}%".format(
            os.path.basename(p), int(N.sum().item()), s2, acc))

    mu_w, logp_w, s2_w = build_nb_paper_from_ABN(A_agg, N_agg, B_agg)
    Sigma_pool_w = agg["Sigma_pool"]
    var_diag_w = torch.diag(Sigma_pool_w).clamp_min(1e-6)

    nbw = (logpdf_gauss_spherical(X64, mu_w, s2_w, logp_w).argmax(1).cpu() == y).float().mean().item()*100
    nbd = (logpdf_gauss_diag(X64, mu_w, var_diag_w, logp_w).argmax(1).cpu() == y).float().mean().item()*100
    lda = (lda_scores(X64, mu_w, Sigma_pool_w, logp_w, alpha=LDA_SHRINK).argmax(1).cpu() == y).float().mean().item()*100

    if agg["Sigma_cls"] is not None:
        qda = (qda_scores(X64, mu_w, agg["Sigma_cls"], logp_w, alpha=QDA_SHRINK).argmax(1).cpu() == y).float().mean().item()*100
    else:
        qda = float("nan")

    print("\n================ SUMMARY (ResNet18 / WEAK) ================")
    print("NB_paper : {:8.2f}%".format(nbw))
    print("NB_diag  : {:8.2f}%".format(nbd))
    print("LDA      : {:8.2f}%".format(lda))
    print("QDA_full : {:8}".format("{:.2f}%".format(qda) if qda==qda else "n/a"))
    print("===========================================================")

    return dict(NB_paper=nbw, NB_diag=nbd, LDA=lda,
                QDA_full=None if qda!=qda else qda)

def eval_strong_pipeline(agg: Dict):
    print("\n[START] device={} | backbone=resnet18 | pipeline=strong".format(DEVICE))

    fe_s, tfm_s, _, _ = build_resnet18_strong()
    testset = datasets.CIFAR10(root=DATA_ROOT, train=False, transform=tfm_s, download=True)
    loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY
    )

    X, y = extract_features(loader, fe_s)
    X64 = X.to(torch.float64)

    mu = agg["mu"]; pri = agg["priors"]
    Sigma_pool = agg["Sigma_pool"]; Sigma_cls = agg["Sigma_cls"]
    A, N, B = agg["A"], agg["N"], agg["B"]

    mu_nb, logp_nb, s2_nb = build_nb_paper_from_ABN(A, N, B)
    var_diag = torch.diag(Sigma_pool).clamp_min(1e-6)

    nbp = (logpdf_gauss_spherical(X64, mu_nb, s2_nb, logp_nb).argmax(1).cpu() == y).float().mean().item()*100
    nbd = (logpdf_gauss_diag(X64, mu, var_diag, torch.log(pri.clamp_min(1e-12))).argmax(1).cpu() == y).float().mean().item()*100
    lda = (lda_scores(X64, mu, Sigma_pool, torch.log(pri.clamp_min(1e-12)), alpha=LDA_SHRINK).argmax(1).cpu() == y).float().mean().item()*100

    if Sigma_cls is not None:
        qda = (qda_scores(X64, mu, Sigma_cls, torch.log(pri.clamp_min(1e-12)), alpha=QDA_SHRINK).argmax(1).cpu() == y).float().mean().item()*100
    else:
        qda = float("nan")

    V, k = fisher_subspace_by_energy(
        mu, pri, Sigma_pool,
        alpha=LDA_SHRINK,
        energy=FISHER_ENERGY,
        max_k=FISHER_MAX_K
    )

    WITH_CLASS = (Sigma_cls is not None) and USE_CLASS_COV

    Zs, Ys = synth_gaussian_in_z(
        mu, Sigma_pool, Sigma_cls, pri, V,
        ns_per_class=SYN_NS_PER_CLASS,
        use_class_cov=WITH_CLASS,
        shrink_alpha=SYN_SHRINK,
        tau=SYN_TAU,
        seed=123
    )

    fm = LogisticHead(k, mu.shape[0]).to(DEVICE)
    opt = torch.optim.AdamW(fm.parameters(), lr=LR, weight_decay=WD)
    ds = torch.utils.data.TensorDataset(Zs, Ys)
    dl = torch.utils.data.DataLoader(ds, batch_size=1024, shuffle=True)

    for ep in range(1, FM_EPOCHS+1):
        fm.train(); tot=0.0
        for z,yb in dl:
            z=z.to(DEVICE); yb=yb.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(fm(z), yb)
            loss.backward(); opt.step()
            tot += loss.item()*z.size(0)
        if ep % 10 == 0:
            print(f"[FM] ep {ep:02d} | loss={tot/len(ds):.4f}")

    V64 = V.to(torch.float64)
    mu_z_dev = (mu.to(torch.float64) @ V64).to(torch.float32).to(DEVICE)
    log_prior_dev = torch.log(pri.clamp_min(1e-12)).to(torch.float32).to(DEVICE)
    sigma2_z = float(torch.trace(V64.t() @ Sigma_pool.to(torch.float64) @ V64).item() / float(k))

    def nb_logits_in_z_dev(z32: torch.Tensor) -> torch.Tensor:
        z64 = z32.to(torch.float64).to(DEVICE)
        mu64 = mu_z_dev.to(torch.float64)
        logp64 = log_prior_dev.to(torch.float64)
        return logpdf_gauss_spherical(z64, mu64, sigma2_z, logp64).to(torch.float32)

    ph = ProtoHyperResidual(k, mu.shape[0], rank=min(PH_RANK, k)).to(DEVICE)
    opt2 = torch.optim.AdamW(ph.parameters(), lr=LR, weight_decay=WD)

    for ep in range(1, PH_EPOCHS+1):
        ph.train(); tot=0.0
        for z,yb in dl:
            z=z.to(DEVICE); yb=yb.to(DEVICE)
            with torch.no_grad():
                base = nb_logits_in_z_dev(z)
            logits = base + ph(z)
            loss = F.cross_entropy(logits, yb)
            opt2.zero_grad(set_to_none=True)
            loss.backward(); opt2.step()
            tot += loss.item()*z.size(0)
        if ep % 10 == 0:
            print(f"[PH] ep {ep:02d} | loss={tot/len(ds):.4f}")

    Zt = (X64 @ V64).to(torch.float32).to(DEVICE)

    with torch.no_grad():
        fm_acc = (fm(Zt).argmax(1).cpu() == y).float().mean().item()*100
        ph_acc = ((nb_logits_in_z_dev(Zt) + ph(Zt)).argmax(1).cpu() == y).float().mean().item()*100

    print("\n================ SUMMARY (ResNet18 / STRONG) ================")
    print("NB_paper  : {:8.2f}%".format(nbp))
    print("NB_diag   : {:8.2f}%".format(nbd))
    print("LDA       : {:8.2f}%".format(lda))
    print("QDA_full  : {:8}".format("{:.2f}%".format(qda) if qda==qda else "n/a"))
    print("FisherMix : {:8.2f}%".format(fm_acc))
    print("ProtoHyper: {:8.2f}%".format(ph_acc))
    print("==============================================================")

    return dict(
        NB_paper=nbp, NB_diag=nbd, LDA=lda,
        QDA_full=None if qda!=qda else qda,
        FisherMix=fm_acc, ProtoHyper=ph_acc
    )

# =======================
# Main
# =======================
def main():
    paths = list_client_files()
    if not paths:
        print(f"[ERR] Nessun file valido trovato in {STATS_ROOT}")
        return

    print(f"[INFO] gruppi resnet18 trovati: 1\n\n[GROUP] resnet18-IMAGENET1K_V1 | files={len(paths)}")

    _ = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True)
    _ = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True)

    agg = aggregate_from_clients(paths)

    _ = eval_weak_pipeline(paths, agg)
    _ = eval_strong_pipeline(agg)

if __name__ == "__main__":
    main()
