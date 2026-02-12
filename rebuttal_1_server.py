#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GH-OFL — Server-side (SVHN) — MODE B A2 with per-sample normalization (choice A)
Compatibile al 100% con il client MODE-B.

Caratteristiche principali:
 - Normalizzazione per-sample anche in test (come client) ✔
 - Fisher subspace 0.9995 energy, min 8, max 256
 - KD migliorata: teacher = 0.8*LDA + 0.2*NB, T=3.0
 - FisherMix warm-start da LDA
 - Whitening stabile
 - Metriche complete richieste dal revisore
"""

import os, glob, time, math, warnings, random
import numpy as np
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision.models import resnet18, ResNet18_Weights
from torchvision.datasets import SVHN
from torchvision import transforms

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEV = torch.device(DEVICE)

# ================================================================
# CONFIG A2
# ================================================================
LDA_SHRINK = 0.04
SYN_SHRINK = 0.04
FISHER_ENERGY = 0.9995
FISHER_MIN_K = 8
FISHER_MAX_K = 256

FM_EPOCHS = 22
PH_EPOCHS = 22
LR = 1.8e-3
WD_FM = 1.5e-4
WD_PH = 3.5e-4
MAX_NORM = 1.7

KD_T = 3.0
KD_ALPHA = 0.6
TEACH_BLEND = (0.8, 0.2)

SYN_PER_CLASS = 4000

# ================================================================
# UTILS
# ================================================================
def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def list_clients(folder):
    return sorted(glob.glob(os.path.join(folder, "client_*", "client.pt")))

def safe_load(path):
    try:
        return torch.load(path, map_location="cpu")
    except:
        return None

def shrink_cov(S, alpha):
    d = S.shape[-1]
    tr = float(torch.trace(S))
    return (1 - alpha) * S + alpha * (tr / d) * torch.eye(d, dtype=S.dtype)

# ================================================================
# AGGREGATION (MODE-B)
# ================================================================
def aggregate_clients(files):
    A_sum = S2_sum = B_sum = N_sum = None
    meta = None
    upload_kb = 0

    for p in files:
        d = safe_load(p)
        if d is None:
            continue

        upload_kb += os.path.getsize(p) / 1024
        if meta is None:
            meta = d["meta"]

        A = d["A_per_class_z"].to(torch.float64)
        S2 = d["SUMSQ_per_class_z"].to(torch.float64)
        N = d["N_per_class"].to(torch.long)
        B = d["B_global_z"].to(torch.float64)

        if A_sum is None:
            A_sum = torch.zeros_like(A)
            S2_sum = torch.zeros_like(S2)
            B_sum  = torch.zeros_like(B)
            N_sum  = torch.zeros_like(N)

        A_sum += A
        S2_sum += S2
        N_sum  += N
        B_sum  += B

    C, k = A_sum.shape
    N_tot = int(N_sum.sum())

    pri = (N_sum / float(N_tot)).to(torch.float64)
    mu = A_sum / N_sum.clamp_min(1).unsqueeze(1)
    Ez2 = S2_sum / N_sum.clamp_min(1).unsqueeze(1)
    var_diag = (Ez2 - mu.pow(2)).clamp_min(1e-9)

    global_mean = (A_sum.sum(0) / float(N_tot)).unsqueeze(1)
    Sigma_pool = (B_sum - (global_mean @ global_mean.t()) * float(N_tot)) / max(1, N_tot - 1)

    return dict(
        meta=meta,
        C=C,
        k=k,
        mu_z=mu,
        priors=pri,
        var_diag=var_diag,
        Sigma_pool_z=Sigma_pool,
        upload_kb=upload_kb
    )

# ================================================================
# FISHER SUBSPACE
# ================================================================
def fisher_subspace(mu, pri, Sigma):
    C, k = mu.shape
    mbar = (pri.unsqueeze(1) * mu).sum(0, keepdim=True)

    Sb = torch.zeros(k, k, dtype=torch.float64)
    for c in range(C):
        dm = (mu[c:c+1] - mbar)
        Sb += float(pri[c].item()) * (dm.t() @ dm)

    Sw = shrink_cov(Sigma, LDA_SHRINK)
    M = torch.linalg.solve(Sw, Sb)

    evals, evecs = torch.linalg.eigh(M)
    evals = evals.clamp_min(0)

    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]

    cum = torch.cumsum(evals, 0)
    tot = float(cum[-1].item())
    r = int((cum / tot >= FISHER_ENERGY).nonzero()[0].item()) + 1
    r = max(FISHER_MIN_K, min(FISHER_MAX_K, r, k))

    return evecs[:, :r].contiguous(), r

# ================================================================
# SYNTHETIC DATASET
# ================================================================
class SynthZF(Dataset):
    def __init__(self, mu_zf64, V, var_diag, ns):
        self.mu = mu_zf64.to(torch.float64)
        self.C, self.kf = self.mu.shape
        self.var = var_diag.to(torch.float64)
        self.V = V.to(torch.float64)
        self.ns = ns
        self.total = self.C * ns

        self.mu32 = self.mu.to(torch.float32).to(DEV)
        self.L = []

        for c in range(self.C):
            D = torch.diag(self.var[c])
            Sf = self.V.t() @ D @ self.V
            Sf = shrink_cov(Sf, SYN_SHRINK) + 1e-6 * torch.eye(self.kf, dtype=torch.float64)
            Lc = torch.linalg.cholesky(Sf)
            self.L.append(Lc.to(torch.float32).to(DEV))

    def __len__(self): return self.total

    def __getitem__(self, i):
        c = i % self.C
        eps = torch.randn(self.kf, device=DEV)
        zf = self.mu32[c] + eps @ self.L[c].t()
        return zf, c

# ================================================================
# NB / NB_diag / LDA LOGITS
# ================================================================
def nb_spherical_fn(mu_z64, Sigma_z64, logpri):
    k = mu_z64.shape[1]
    sigma2 = float(torch.trace(Sigma_z64) / k)

    mu = mu_z64.to(torch.float32).to(DEV)
    logp = logpri.to(torch.float32).to(DEV)
    mu2 = (mu*mu).sum(1)
    muT = mu.t()

    inv2 = 0.5 / max(sigma2, 1e-12)
    logdet = 0.5*k*math.log(max(sigma2, 1e-12))

    def fn(Z):
        Z = Z.to(torch.float32).to(DEV)
        x2 = (Z*Z).sum(1)
        xmu = Z @ muT
        quad = inv2*(x2.unsqueeze(1) - 2*xmu + mu2.unsqueeze(0))
        return logp.unsqueeze(0) - quad - logdet

    return fn

def nb_diag_fn(mu_z64, var_diag64, logpri):
    mu = mu_z64.to(torch.float32).to(DEV)
    var = var_diag64.to(torch.float32).to(DEV).clamp_min(1e-8)
    logp = logpri.to(torch.float32).to(DEV)

    def fn(Z):
        Z=Z.to(torch.float32).to(DEV)
        dif = Z.unsqueeze(1)-mu.unsqueeze(0)
        quad = 0.5*(dif.pow(2)/var.unsqueeze(0)).sum(2)
        logdet = 0.5*torch.log(var).sum(1)
        return logp.unsqueeze(0)-quad-logdet.unsqueeze(0)

    return fn

def lda_fn(mu64, Sf64, logpri):
    Sshr = shrink_cov(Sf64, LDA_SHRINK) + 1e-6*torch.eye(Sf64.shape[0])
    W64 = torch.linalg.solve(Sshr, mu64.t())
    b64 = -0.5*(mu64*W64.t()).sum(1) + logpri

    W32 = W64.to(torch.float32).to(DEV)
    b32 = b64.to(torch.float32).to(DEV)

    def fn(Z):
        return Z.to(torch.float32).to(DEV) @ W32 + b32

    return fn, W64, b64

# ================================================================
# FEATURE EXTRACTOR 32x32
# ================================================================
class ResNet18_Features(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        for p in m.parameters(): p.requires_grad=False
        self.body = nn.Sequential(*list(m.children())[:-1])
    def forward(self,x): return self.body(x).flatten(1)

SVHN_NORM = transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))

test_tf = transforms.Compose([
    transforms.ToTensor(),
    SVHN_NORM,
])

# ================================================================
# MODELS
# ================================================================
class FisherMix(nn.Module):
    def __init__(self,kf,C):
        super().__init__()
        self.fc = nn.Linear(kf,C)
    def forward(self,z): return self.fc(z)

class ProtoHyper(nn.Module):
    def __init__(self,kf,C,r=128):
        super().__init__()
        self.U = nn.Linear(kf,r,bias=False)
        self.V = nn.Linear(r,C,bias=True)
    def forward(self,z): return self.V(self.U(z))

# ================================================================
# MAIN
# ================================================================
def main():

    set_seed(42)
    ROOT="./oneshot_bench/SVHN/0.5/clients/FedCGS"

    print(f"[INFO] Device={DEVICE}")
    files=list_clients(ROOT)
    print(f"[INFO] Found {len(files)} clients")

    # =======================================================
    # 1) AGGREGATION
    # =======================================================
    t0=time.time()
    agg=aggregate_clients(files)
    t_agg=time.time()-t0

    print(f"[OK] Aggregation: {t_agg:.2f}s | upload={agg['upload_kb']:.1f}KB")

    C=agg["C"]
    k=agg["k"]
    mu_z=agg["mu_z"]
    pri=agg["priors"]
    var_diag=agg["var_diag"]
    Sigma=agg["Sigma_pool_z"]
    logpri=torch.log(pri.clamp_min(1e-12))

    nb_fn_z = nb_spherical_fn(mu_z, Sigma, logpri)
    nb_diag_z = nb_diag_fn(mu_z, var_diag, logpri)

    # =======================================================
    # 2) FISHER
    # =======================================================
    t0=time.time()
    V64,kf = fisher_subspace(mu_z,pri,Sigma)
    t_fish=time.time()-t0
    print(f"[OK] Fisher: k_f={kf}, t={t_fish:.2f}s")

    mu_zf64 = mu_z @ V64
    Sf64 = V64.t() @ Sigma @ V64

    lda_zf_fn,W64_lda,b64_lda = lda_fn(mu_zf64,Sf64,logpri)

    # whitening
    evals, evecs = torch.linalg.eigh(shrink_cov(Sf64,SYN_SHRINK)+1e-6*torch.eye(kf))
    Awh = (evecs @ torch.diag(evals.clamp_min(1e-9).rsqrt()) @ evecs.t()).to(torch.float32).to(DEV)

    # =======================================================
    # 3) SINTESI
    # =======================================================
    t0=time.time()
    syn_ds=SynthZF(mu_zf64,V64,var_diag,SYN_PER_CLASS)
    syn_dl=DataLoader(syn_ds,batch_size=2048,shuffle=True)
    t_syn=time.time()-t0
    syn_mem = syn_ds.total * kf * 4 / (1024*1024)

    print(f"[OK] Synthetic: {syn_ds.total} samples | t={t_syn:.2f}s")

    # =======================================================
    # 4) TRAIN FisherMix
    # =======================================================
    fm=FisherMix(kf,C).to(DEV)
    opt_fm=torch.optim.AdamW(fm.parameters(),lr=LR,weight_decay=WD_FM)
    sch_fm=torch.optim.lr_scheduler.CosineAnnealingLR(opt_fm,T_max=FM_EPOCHS)

    with torch.no_grad():
        fm.fc.weight.copy_(W64_lda.to(torch.float32).t().to(DEV))
        fm.fc.bias.copy_(b64_lda.to(torch.float32).to(DEV))

    t0=time.time()
    for ep in range(FM_EPOCHS):
        fm.train()
        for zf,y in syn_dl:
            zf=zf.to(DEV); y=y.to(DEV)
            zfw = zf @ Awh
            opt_fm.zero_grad()
            loss = F.cross_entropy(fm(zfw),y,label_smoothing=0.05)
            loss.backward()
            nn.utils.clip_grad_norm_(fm.parameters(),MAX_NORM)
            opt_fm.step()
        sch_fm.step()
    t_fm=time.time()-t0
    print(f"[OK] FisherMix train: {t_fm:.2f}s")

    # =======================================================
    # 5) TRAIN ProtoHyper A2
    # =======================================================
    ph=ProtoHyper(kf,C,r=min(160,kf)).to(DEV)
    opt_ph=torch.optim.AdamW(ph.parameters(),lr=LR,weight_decay=WD_PH)
    sch_ph=torch.optim.lr_scheduler.CosineAnnealingLR(opt_ph,T_max=PH_EPOCHS)

    t0=time.time()
    for ep in range(PH_EPOCHS):
        ph.train()
        for zf,y in syn_dl:
            zf=zf.to(DEV); y=y.to(DEV)

            Z_nb = zf @ V64.to(torch.float32).to(DEV).t()

            with torch.no_grad():
                tea = TEACH_BLEND[0]*lda_zf_fn(zf) + TEACH_BLEND[1]*nb_fn_z(Z_nb)
                tea = (tea-tea.mean(1,keepdim=True))/tea.std(1,keepdim=True).clamp_min(1e-6)
                base = nb_fn_z(Z_nb)
                base = (base-base.mean(1,keepdim=True))/base.std(1,keepdim=True).clamp_min(1e-6)

            logits = base + ph(zf)

            loss_kd = F.kl_div(
                F.log_softmax(logits/KD_T,1),
                F.softmax(tea/KD_T,1),
                reduction="batchmean"
            )*(KD_T*KD_T)

            loss_ce = F.cross_entropy(logits,y,label_smoothing=0.05)
            loss = KD_ALPHA*loss_kd + (1-KD_ALPHA)*loss_ce

            opt_ph.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ph.parameters(),MAX_NORM)
            opt_ph.step()

        sch_ph.step()
    t_ph=time.time()-t0
    print(f"[OK] ProtoHyper train: {t_ph:.2f}s")

    # =======================================================
    # 6) TEST (32×32 + per-sample normalization)
    # =======================================================
    FE=ResNet18_Features().to(DEV).eval()

    # ricostruisci RP EXATTAMENTE come lato client
    g = torch.Generator(device="cpu").manual_seed(agg["meta"]["rp_seed"])
    R = torch.randn(512,agg["k"],generator=g)/math.sqrt(512)
    R = R.to(torch.float32).to(DEV)

    V32 = V64.to(torch.float32).to(DEV)

    test = SVHN("./data", split="test", download=True, transform=test_tf)
    test_dl = DataLoader(test,batch_size=256,shuffle=False)

    t0=time.time()
    tot=0
    c_nb=c_nbd=c_lda=c_fm=c_ph=0

    with torch.no_grad():
        for x,y in test_dl:
            x=x.to(DEV); y=y.to(DEV)

            Z = FE(x) @ R

            # per-sample z-normalization (CHOICE A)
            mu = Z.mean(1,keepdim=True)
            std = Z.std(1,keepdim=True).clamp_min(1e-5)
            Z = (Z-mu)/std

            Zf = Z @ V32

            c_nb  += (nb_fn_z(Z).argmax(1)==y).sum().item()
            c_nbd += (nb_diag_fn(mu_z, var_diag, logpri)(Z).argmax(1)==y).sum().item()
            c_lda += (lda_zf_fn(Zf).argmax(1)==y).sum().item()
            c_fm  += (fm(Zf@Awh).argmax(1)==y).sum().item()

            base = nb_fn_z(Z)
            base=(base-base.mean(1,keepdim=True))/base.std(1,keepdim=True).clamp_min(1e-6)
            c_ph += ((base+ph(Zf)).argmax(1)==y).sum().item()

            tot += y.numel()

    t_test=time.time()-t0
    thr = tot/t_test

    print("\n==================== FINAL SERVER SUMMARY ====================")
    print(f"Upload total:        {agg['upload_kb']:.1f} KB")
    print(f"Aggregation time:    {t_agg:.2f} s")
    print(f"Fisher time:         {t_fish:.2f} s")
    print(f"Synthesis time:      {t_syn:.2f} s")
    print(f"Synthesis memory:    {syn_mem:.2f} MB")
    print(f"FM train time:       {t_fm:.2f} s")
    print(f"PH train time:       {t_ph:.2f} s")
    print(f"Test infer time:     {t_test:.2f} s")
    print(f"Test throughput:     {thr:.1f} img/s")
    print(f"NB(z) accuracy:      {100*c_nb/tot:.2f}%")
    print(f"NB_diag(z) acc:      {100*c_nbd/tot:.2f}%")
    print(f"LDA(z_f) accuracy:   {100*c_lda/tot:.2f}%")
    print(f"FisherMix accuracy:  {100*c_fm/tot:.2f}%")
    print(f"ProtoHyper accuracy: {100*c_ph/tot:.2f}%")
    print("=============================================================")

if __name__=="__main__":
    main()





