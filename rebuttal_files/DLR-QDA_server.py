#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GH-OFL — Server-side (SVHN) — VERSIONE A 100% CORRETTA
NB(z), NB_diag(z), LDA(z_f), FisherMix(z_f), ProtoHyper(z_f)
Coerente col client SVHN (RP=256) + pipeline ufficiale GH-OFL (NLP version).
"""

import os, glob, math, time, warnings, platform, random
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import SVHN
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights

# =====================================================================
# CONFIG
# =====================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEV = torch.device(DEVICE)
IS_WIN = platform.system().lower().startswith("win")

PIN_MEMORY = torch.cuda.is_available() and not IS_WIN
BATCH_TEST = 512

LDA_SHRINK = 0.05
SYN_SHRINK = 0.05

FM_EPOCHS = 20
PH_EPOCHS = 20
LR = 1e-3
WD_FM = 2e-4
WD_PH = 5e-4
MAX_NORM = 2.0
KD_T = 2.0
KD_ALPHA = 0.5
TEACHER_BLEND = (0.7, 0.3)  # (LDA(z_f), NB(z))

FISHER_ENERGY = 0.999
FISHER_MAX_K = 128

SYN_PER_CLASS = 4000                           # 40k synth → 4000 per 10 classi
ROOT = "./oneshot_bench/SVHN/0.5/clients/FedCGS"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

warnings.filterwarnings("ignore")


# =====================================================================
# UTILS
# =====================================================================
def set_seed(s=42):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def list_clients_in(folder: str) -> List[str]:
    flat = sorted(glob.glob(os.path.join(folder, "client_*.pt")))
    nested = sorted(glob.glob(os.path.join(folder, "client_*", "client.pt")))
    return flat + nested


def safe_load(p: str) -> Dict:
    try:
        return torch.load(p, map_location="cpu")
    except Exception:
        return {}


def shrink_cov(S: torch.Tensor, alpha: float) -> torch.Tensor:
    d = S.shape[-1]
    tr = float(torch.trace(S))
    return (1 - alpha) * S + alpha * (tr / d) * torch.eye(d, dtype=S.dtype)


def standardize_logits_per_row(L: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = L.mean(dim=1, keepdim=True)
    s = L.std(dim=1, keepdim=True).clamp_min(eps)
    return (L - m) / s


# =====================================================================
# AGGREGATION (z-space)
# =====================================================================
def aggregate_from_clients(files: List[str]) -> Dict:
    meta0 = None
    A_sum = S2_sum = B_sum = N_sum = None

    for p in files:
        d = safe_load(p)
        if not d:
            continue

        meta = d["meta"]
        if meta0 is None:
            meta0 = meta

        A = d["A_per_class_z"].to(torch.float64)
        S2 = d["SUMSQ_per_class_z"].to(torch.float64)
        N = d["N_per_class"].to(torch.long)
        B = d["B_global_z"].to(torch.float64)

        if A_sum is None:
            A_sum = torch.zeros_like(A)
            S2_sum = torch.zeros_like(S2)
            N_sum  = torch.zeros_like(N)
            B_sum  = torch.zeros_like(B)

        A_sum += A
        S2_sum += S2
        N_sum  += N
        B_sum  += B

    if A_sum is None:
        raise RuntimeError("Nessun client valido trovato.")

    C, k = A_sum.shape
    N_total = int(N_sum.sum().item())
    pri = (N_sum / float(N_total)).to(torch.float64)

    mu_z = A_sum / N_sum.clamp_min(1).unsqueeze(1)
    Ez2 = S2_sum / N_sum.clamp_min(1).unsqueeze(1)
    var_diag = (Ez2 - mu_z.pow(2)).clamp_min(1e-9)

    m_all = (A_sum.sum(dim=0) / float(N_total)).unsqueeze(1)
    Sigma_pool_z = (B_sum - (m_all @ m_all.t()) * float(N_total)) / max(1, N_total - 1)

    return dict(
        meta=meta0, C=C, k=k,
        mu_z=mu_z, priors=pri,
        var_diag=var_diag,
        Sigma_pool_z=Sigma_pool_z,
    )


# =====================================================================
# Fisher subspace
# =====================================================================
def fisher_subspace(mu_z: torch.Tensor,
                    pri: torch.Tensor,
                    Sigma_pool_z: torch.Tensor,
                    energy: float = FISHER_ENERGY,
                    max_k: int = FISHER_MAX_K):
    C, k = mu_z.shape

    mbar = (pri.unsqueeze(1) * mu_z).sum(dim=0, keepdim=True).to(torch.float64)

    Sb = torch.zeros(k, k, dtype=torch.float64)
    for c in range(C):
        dm = (mu_z[c:c+1].to(torch.float64) - mbar)
        Sb += float(pri[c].item()) * (dm.t() @ dm)

    Sw = shrink_cov(Sigma_pool_z.to(torch.float64), alpha=LDA_SHRINK)
    M = torch.linalg.solve(Sw, Sb)

    ev, evec = torch.linalg.eig(M)
    ev = ev.real.clamp_min(0)
    evec = evec.real

    idx = torch.argsort(ev, descending=True)
    ev = ev[idx]
    evec = evec[:, idx]

    csum = torch.cumsum(ev, dim=0)
    tot = float(csum[-1].item())
    r = int((csum / max(tot, 1e-12) >= energy).nonzero()[0].item() + 1)
    r = max(2, min(r, max_k, k))

    return evec[:, :r].contiguous(), r


# =====================================================================
# Logit functions — NB(z), NB_diag(z), LDA(z_f)
# =====================================================================
def nb_spherical_logits_z_fn(mu_z64: torch.Tensor,
                             Sigma_z64: torch.Tensor,
                             logpri64: torch.Tensor):
    k = mu_z64.shape[1]
    sigma2 = float(torch.trace(Sigma_z64) / k)

    mu = mu_z64.to(torch.float32).to(DEV)
    logp = logpri64.to(torch.float32).to(DEV)
    mu2 = (mu * mu).sum(dim=1)
    muT = mu.t().contiguous()

    inv2 = 0.5 / max(sigma2, 1e-12)
    logdet = 0.5 * k * math.log(max(sigma2, 1e-12))

    def fn(Z: torch.Tensor) -> torch.Tensor:
        Z = Z.to(torch.float32).to(DEV)
        x2 = (Z * Z).sum(dim=1)
        xmu = Z @ muT
        quad = inv2 * (x2.unsqueeze(1) - 2 * xmu + mu2.unsqueeze(0))
        return logp.unsqueeze(0) - quad - logdet

    return fn


def nb_diag_logits_z_fn(mu_z64: torch.Tensor,
                        var_diag64: torch.Tensor,
                        logpri64: torch.Tensor):
    mu = mu_z64.to(torch.float32).to(DEV)
    var = var_diag64.to(torch.float32).to(DEV).clamp_min(1e-9)
    logp = logpri64.to(torch.float32).to(DEV)

    def fn(Z: torch.Tensor) -> torch.Tensor:
        Z = Z.to(torch.float32).to(DEV)
        dif = Z.unsqueeze(1) - mu.unsqueeze(0)      # [B,C,k]
        quad = 0.5 * (dif.pow(2) / var.unsqueeze(0)).sum(dim=2)
        logdet = 0.5 * torch.log(var).sum(dim=1)
        return logp.unsqueeze(0) - quad - logdet.unsqueeze(0)

    return fn


def lda_logits_zf_fn(mu_zf64: torch.Tensor,
                     S_f64: torch.Tensor,
                     logpri64: torch.Tensor):
    Sshr = shrink_cov(S_f64, alpha=LDA_SHRINK) + 1e-6 * torch.eye(S_f64.shape[0])
    W64 = torch.linalg.solve(Sshr, mu_zf64.t())              # [kf, C]
    b64 = -0.5 * (mu_zf64 * W64.t()).sum(dim=1) + logpri64   # [C]

    W32 = W64.to(torch.float32).to(DEV)
    b32 = b64.to(torch.float32).to(DEV)

    def fn(Zf: torch.Tensor) -> torch.Tensor:
        return Zf.to(torch.float32).to(DEV) @ W32 + b32

    return fn, W64, b64


# =====================================================================
# Encoder SVHN (test)
# =====================================================================
class ResNet18_Features(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for p in m.parameters():
            p.requires_grad = False
        self.body = nn.Sequential(*list(m.children())[:-1])  # [B,512,1,1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x).flatten(1)                        # [B,512]


# =====================================================================
# Synthetic z_f dataset
# =====================================================================
class SynthZf(Dataset):
    def __init__(self,
                 mu_zf64: torch.Tensor,
                 V64: torch.Tensor,
                 var_diag: torch.Tensor,
                 ns_per_class: int,
                 seed: int = 123):
        super().__init__()
        self.mu = mu_zf64.to(torch.float64).cpu()
        self.C, self.kf = self.mu.shape
        self.V = V64.to(torch.float64).cpu()
        self.var_diag = var_diag.to(torch.float64).cpu()
        self.ns = ns_per_class
        self.total = self.C * self.ns

        self.gen = torch.Generator(device=DEV).manual_seed(seed)

        var_pool = self.var_diag.mean(dim=0)
        mean_pool = float(var_pool.mean().item())
        mean_cls = self.var_diag.mean(dim=1)

        self.L = []
        self.tau = []
        for c in range(self.C):
            D = torch.diag(self.var_diag[c])
            Sf = shrink_cov(self.V.t() @ D @ self.V, alpha=SYN_SHRINK)
            Sf = Sf + 1e-6 * torch.eye(self.kf)
            Lc = torch.linalg.cholesky(Sf)
            self.L.append(Lc.to(torch.float32).to(DEV))

            t = float((mean_cls[c] / mean_pool).sqrt().item())
            t = max(0.7, min(1.4, t))
            self.tau.append(torch.tensor(t, dtype=torch.float32, device=DEV))

        self.mu_f = self.mu.to(torch.float32).to(DEV)

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, idx: int):
        c = idx % self.C
        eps = torch.randn(self.kf, generator=self.gen, device=DEV)
        zf = self.mu_f[c] + self.tau[c] * (eps @ self.L[c].t())
        return zf, c


# =====================================================================
# RP reconstruction
# =====================================================================
def rp_from_meta(meta: Dict) -> torch.Tensor:
    d = int(meta["feature_dim"])
    k = int(meta["rp_dim"])
    seed = int(meta["rp_seed"])
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.randn(d, k, generator=g) / math.sqrt(d)
    return R.to(torch.float32).to(DEV)


# =====================================================================
# MAIN
# =====================================================================
def main():
    set_seed(42)
    print(f"[INFO] Device: {DEVICE}")

    files = list_clients_in(ROOT)
    print(f"[INFO] Found {len(files)} clients")

    # ------------------------------------------
    # Aggregazione
    # ------------------------------------------
    t0 = time.time()
    agg = aggregate_from_clients(files)
    t_agg = time.time() - t0

    C = agg["C"]
    k = agg["k"]

    mu_z = agg["mu_z"]
    var_diag = agg["var_diag"]
    Sigma_pool_z = agg["Sigma_pool_z"]
    pri = agg["priors"]
    logpri = torch.log(pri.clamp_min(1e-12))

    upload_kb = sum(os.path.getsize(f) for f in files) / 1024.0
    print(f"[OK] Aggregation done in {t_agg:.2f}s | upload={upload_kb:.1f} KB")

    # ------------------------------------------
    # NB(z), NB_diag(z)
    # ------------------------------------------
    nb_fn = nb_spherical_logits_z_fn(
        mu_z.to(torch.float64),
        Sigma_pool_z.to(torch.float64),
        logpri,
    )

    nb_diag_fn = nb_diag_logits_z_fn(
        mu_z.to(torch.float64),
        var_diag.to(torch.float64),
        logpri,
    )

    # ------------------------------------------
    # Fisher subspace (z_f)
    # ------------------------------------------
    t1 = time.time()
    V64, kf = fisher_subspace(mu_z, pri, Sigma_pool_z)
    t_fish = time.time() - t1

    print(f"[OK] Fisher: k_f={kf}, time={t_fish:.2f}s")

    mu_zf64 = (mu_z.to(torch.float64) @ V64)
    Sig_pool_zf64 = (V64.t() @ Sigma_pool_z.to(torch.float64) @ V64)

    lda_fn, W64_lda, b64_lda = lda_logits_zf_fn(mu_zf64, Sig_pool_zf64, logpri)

    # Whitening in z_f
    evals, evecs = torch.linalg.eigh(
        shrink_cov(Sig_pool_zf64, alpha=SYN_SHRINK) + 1e-6 * torch.eye(kf)
    )
    A_wh = (
        evecs
        @ torch.diag(evals.clamp_min(1e-9).rsqrt())
        @ evecs.t()
    ).to(torch.float32).to(DEV)

    # ------------------------------------------
    # Synthetic dataset in z_f
    # ------------------------------------------
    t2 = time.time()
    syn_ds = SynthZf(mu_zf64, V64, var_diag, SYN_PER_CLASS)
    syn_dl = DataLoader(syn_ds, batch_size=2048, shuffle=False)
    t_syn = time.time() - t2

    synth_mem = syn_ds.total * kf * 4 / (1024 * 1024)

    print(f"[OK] Synthetic data: {syn_ds.total} samples | time={t_syn:.2f}s")

    # ------------------------------------------
    # FisherMix & ProtoHyper
    # ------------------------------------------
    fm = nn.Linear(kf, C).to(DEV)
    ph = nn.Sequential(
        nn.Linear(kf, min(128, kf), bias=False),
        nn.Linear(min(128, kf), C, bias=True),
    ).to(DEV)

    opt_fm = torch.optim.AdamW(fm.parameters(), lr=LR, weight_decay=WD_FM)
    opt_ph = torch.optim.AdamW(ph.parameters(), lr=LR, weight_decay=WD_PH)
    sch_fm = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fm, T_max=FM_EPOCHS)
    sch_ph = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ph, T_max=PH_EPOCHS)

    # Warm-start da LDA(z_f)
    with torch.no_grad():
        fm.weight.copy_(W64_lda.to(torch.float32).t().to(DEV))
        fm.bias.copy_(b64_lda.to(torch.float32).to(DEV))

    ce = nn.CrossEntropyLoss(label_smoothing=0.10)

    # Train FM
    t3 = time.time()
    for ep in range(1, FM_EPOCHS + 1):
        fm.train()
        tot = seen = 0
        for zf, y in syn_dl:
            zf = zf.to(DEV)
            y = y.to(DEV)
            logits = fm(zf @ A_wh)

            opt_fm.zero_grad()
            loss = ce(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(fm.parameters(), MAX_NORM)
            opt_fm.step()

            bs = zf.size(0)
            tot += loss.item() * bs
            seen += bs
        sch_fm.step()
        if ep % 5 == 0:
            print(f"[FM] ep {ep:02d}/{FM_EPOCHS} | loss={tot/seen:.4f}")
    t_fm = time.time() - t3

    # -----------------------------
    # TRAIN PROTO-HYPER (KD su z_f, teacher LDA(z_f)+NB(z))
    # -----------------------------
    t4 = time.time()
    V32 = V64.to(torch.float32).to(DEV)  # fisso fuori dal loop

    for ep in range(1, PH_EPOCHS + 1):
        ph.train()
        tot = seen = 0
        for zf, y in syn_dl:
            zf = zf.to(DEV)
            y = y.to(DEV)

            # Ricostruisci Z nello spazio RP z (256D) da z_f
            Z = zf @ V32.t()  # [B,kf] @ [kf,256] → [B,256]

            # Teacher
            with torch.no_grad():
                tea = (
                    TEACHER_BLEND[0] * lda_fn(zf) +   # LDA(z_f)
                    TEACHER_BLEND[1] * nb_fn(Z)      # NB(z)
                )
                tea = standardize_logits_per_row(tea)

                base = standardize_logits_per_row(nb_fn(Z))   # NB(z) normalizzato

            # Student: base(z) + residuo(z_f)
            logits = base + ph(zf)

            loss_kd = F.kl_div(
                F.log_softmax(logits / KD_T, dim=1),
                F.softmax(tea / KD_T, dim=1),
                reduction="batchmean",
            ) * (KD_T * KD_T)

            loss_ce = F.cross_entropy(logits, y, label_smoothing=0.10)
            loss = KD_ALPHA * loss_kd + (1 - KD_ALPHA) * loss_ce

            opt_ph.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ph.parameters(), MAX_NORM)
            opt_ph.step()

            bs = zf.size(0)
            tot += loss.item() * bs
            seen += bs

        sch_ph.step()
        if ep % 5 == 0:
            print(f"[PH] ep {ep:02d}/{PH_EPOCHS} | loss=ok")
    t_ph = time.time() - t4

    # ------------------------------------------
    # TEST SET SVHN
    # ------------------------------------------
    t5 = time.time()

    encoder = ResNet18_Features().to(DEV).eval()
    R = rp_from_meta(agg["meta"])
    V32 = V64.to(torch.float32).to(DEV)

    tform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_ds = SVHN("../data", split="test", download=True, transform=tform)
    test_dl = DataLoader(test_ds, batch_size=BATCH_TEST, shuffle=False)

    tot = 0
    c_nb = c_nbdiag = c_lda = c_fm = c_ph = 0

    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(DEV)
            y = y.to(DEV)

            Z = encoder(x) @ R          # z-space [B,256]
            Zf = Z @ V32                # z_f-space [B,kf]

            # NB(z)
            pred_nb = nb_fn(Z).argmax(1)

            # NB_diag(z)
            pred_nbdiag = nb_diag_fn(Z).argmax(1)

            # LDA(z_f)
            pred_lda = lda_fn(Zf).argmax(1)

            # FM(z_f)
            pred_fm_ = fm(Zf @ A_wh).argmax(1)

            # PH(z_f) con base NB(z)
            base = standardize_logits_per_row(nb_fn(Z))   # NB in z-space
            pred_ph_ = (base + ph(Zf)).argmax(1)

            c_nb     += (pred_nb == y).sum().item()
            c_nbdiag += (pred_nbdiag == y).sum().item()
            c_lda    += (pred_lda == y).sum().item()
            c_fm     += (pred_fm_ == y).sum().item()
            c_ph     += (pred_ph_ == y).sum().item()
            tot      += y.numel()

    acc_nb     = 100 * c_nb     / tot
    acc_nbdiag = 100 * c_nbdiag / tot
    acc_lda    = 100 * c_lda    / tot
    acc_fm     = 100 * c_fm     / tot
    acc_ph     = 100 * c_ph     / tot

    t_test = time.time() - t5
    thr = tot / t_test

    # ------------------------------------------
    # SUMMARY
    # ------------------------------------------
    print("\n==================== FINAL SERVER SUMMARY (SVHN) ====================")
    print(f"Upload total:        {upload_kb:.1f} KB")
    print(f"Aggregation time:    {t_agg:.2f} s")
    print(f"Fisher time:         {t_fish:.2f} s")
    print(f"Synthesis time:      {t_syn:.2f} s")
    print(f"Synthesis memory:    {synth_mem:.2f} MB")
    print(f"FM train time:       {t_fm:.2f} s")
    print(f"PH train time:       {t_ph:.2f} s")
    print(f"Test infer time:     {t_test:.2f} s")
    print(f"Test throughput:     {thr:.1f} img/s")
    print(f"NB(z) accuracy:      {acc_nb:.2f}%")
    print(f"NB_diag(z):          {acc_nbdiag:.2f}%")
    print(f"LDA(z_f) accuracy:   {acc_lda:.2f}%")
    print(f"FisherMix accuracy:  {acc_fm:.2f}%")
    print(f"ProtoHyper accuracy: {acc_ph:.2f}%")
    print("=====================================================================")


if __name__ == "__main__":
    main()

