#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GH-OFL — AGNEWS Server side (end-to-end)

Usa le statistiche client-side in z (RP) da:
    ./ghofl_stats/AGNEWS/client.pt

Metodi:
 - NB_diag (in z)
 - LDA (in z)
 - QDA_full (in z) con covs diagonali stabili
 - FisherMix (in z)
 - ProtoHyper (in z)

Valuta su AGNEWS test + misura:
 - Tempo end-to-end (minuti)
 - Peak GPU memory (GB)
 - Head cost (ms/sample)
"""

import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEV = torch.device(DEVICE)

CLIENT_PATH = "../ghofl_stats/AGNEWS/client.pt"

# Parametri RP
FEAT_DIM = 768          # DistilBERT embedding size
K_RP = 256
RP_SEED = 12345

FM_EPOCHS = 15
PH_EPOCHS = 15
SYN_PER_CLASS = 3000
BATCH_SYN = 2048

# =========================================================
# Utils
# =========================================================
def shrink_cov(S: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    d = S.shape[-1]
    tr = float(torch.trace(S))
    I = torch.eye(d, dtype=S.dtype)
    return (1 - alpha) * S + alpha * (tr / d) * I

def build_rp_matrix(d: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.randn(d, k, generator=g) / math.sqrt(d)
    return R

# =========================================================
# Synthetic dataset in z
# =========================================================
class SynthZF(torch.utils.data.Dataset):
    def __init__(self, mu_z, L_list, ns_per_class):
        super().__init__()
        self.mu = [m.to(torch.float32) for m in mu_z]
        self.L = [L.to(torch.float32) for L in L_list]
        self.C = len(mu_z)
        self.k = mu_z[0].numel()
        self.ns = int(ns_per_class)
        self.total = self.C * self.ns
        self.gen = torch.Generator(device="cpu").manual_seed(123)

    def __len__(self): return self.total

    def __getitem__(self, idx):
        c = idx % self.C
        eps = torch.randn(self.k, generator=self.gen)
        z = self.mu[c] + eps @ self.L[c].t()
        return z, c

# =========================================================
# Heads
# =========================================================
class FisherMixHead(nn.Module):
    def __init__(self, k, C):
        super().__init__()
        self.fc = nn.Linear(k, C)

    def forward(self, z): return self.fc(z)

class ProtoHyper(nn.Module):
    def __init__(self, k, C, rank=128):
        super().__init__()
        self.U = nn.Linear(k, rank, bias=False)
        self.V = nn.Linear(rank, C)

    def forward(self, z): return self.V(self.U(z))

# =========================================================
# AGNEWS test loader
# =========================================================
def load_agnews_test(batch=64):
    ds = load_dataset("ag_news", split="test")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    encoder = DistilBertModel.from_pretrained("distilbert-base-uncased").to(DEV)
    encoder.eval()

    def collate(batch_items):
        texts = [x["text"] for x in batch_items]
        labels = torch.tensor([x["label"] for x in batch_items], dtype=torch.long)
        tok = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        tok = {k: v.to(DEV) for k, v in tok.items()}
        with torch.no_grad():
            emb = encoder(**tok).last_hidden_state[:, 0]  # [B,768]
        return emb, labels.to(DEV)

    return DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=collate)

# =========================================================
# HEAD COST measurement
# =========================================================
def measure_head_cost(head_fn, z_dim, device="cuda", warmup=15, runs=80, batch=512):
    z = torch.randn(batch, z_dim, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = head_fn(z)
        if device == "cuda": torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.time()
            _ = head_fn(z)
            if device == "cuda": torch.cuda.synchronize()
            t1 = time.time()
            times.append((t1 - t0) * 1000.0 / batch)

    return float(torch.tensor(times).mean())

# =========================================================
# MAIN
# =========================================================
def main():
    print(f"[INFO] Device: {DEVICE}")

    if not os.path.exists(CLIENT_PATH):
        raise FileNotFoundError(CLIENT_PATH)

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()

    # ---------------- Load stats ----------------
    stats = torch.load(CLIENT_PATH, map_location="cpu")

    A = stats["A_per_class_z"].to(torch.float64)
    SUMSQ = stats["SUMSQ_per_class_z"].to(torch.float64)
    N = stats["N_per_class"].to(torch.long)
    B = stats["B_global_z"].to(torch.float64)

    C, k = A.shape
    Ntot = int(N.sum().item())

    mu_z = A / N.clamp_min(1).unsqueeze(1)
    pri = (N / float(Ntot)).to(torch.float64)
    logpri = torch.log(pri.clamp_min(1e-12))

    Ez2 = SUMSQ / N.clamp_min(1).unsqueeze(1)
    var_diag = (Ez2 - mu_z.pow(2)).clamp_min(1e-9)

    m_all = (A.sum(0) / Ntot).unsqueeze(1)
    Sigma_pool_z = (B - Ntot * (m_all @ m_all.t())) / max(1, Ntot - 1)

    print(f"[INFO] Loaded stats: C={C}, k={k}, Ntot={Ntot}")

    # ---------------- QDA diag-friendly ----------------
    L_list = []
    logdets = []
    for c in range(C):
        S_c = torch.diag(var_diag[c])
        S_c = shrink_cov(S_c, 0.07) + 1e-6 * torch.eye(k, dtype=torch.float64)
        L = torch.linalg.cholesky(S_c)
        L_list.append(L)
        logdets.append(torch.log(torch.diag(L)).sum().item())
    logdets = torch.tensor(logdets, dtype=torch.float64)

    # ---------------- Synthetic dataset ----------------
    syn_ds = SynthZF(mu_z, L_list, SYN_PER_CLASS)
    syn_dl = DataLoader(syn_ds, batch_size=BATCH_SYN, shuffle=True, num_workers=0)

    # ---------------- FisherMix ----------------
    fm = FisherMixHead(k, C).to(DEV)
    opt_fm = torch.optim.AdamW(fm.parameters(), lr=1e-3, weight_decay=2e-4)
    ce = nn.CrossEntropyLoss()

    print("[TRAIN] FisherMix...")
    for ep in range(1, FM_EPOCHS + 1):
        fm.train()
        tot_loss = 0.0; seen = 0
        for z, y in syn_dl:
            z = z.to(DEV); y = y.to(DEV)
            logits = fm(z)
            loss = ce(logits, y)
            opt_fm.zero_grad(set_to_none=True); loss.backward(); opt_fm.step()
            tot_loss += loss.item() * z.size(0); seen += z.size(0)
        print(f" [FM] {ep:02d}/{FM_EPOCHS} | loss={tot_loss/seen:.4f}")

    # ---------------- ProtoHyper ----------------
    ph = ProtoHyper(k, C, rank=min(128, k)).to(DEV)
    opt_ph = torch.optim.AdamW(ph.parameters(), lr=1e-3, weight_decay=5e-4)

    def nb_diag_logits_z(z):
        dev = z.device
        z64 = z.to(torch.float64)
        mu = mu_z.to(dev); var = var_diag.to(dev); lp = logpri.to(dev)
        inv = 1.0 / var
        dif = z64.unsqueeze(1) - mu.unsqueeze(0)
        quad = 0.5 * (dif * dif * inv.unsqueeze(0)).sum(2)
        logdet = 0.5 * torch.log(var).sum(1)
        return (lp.unsqueeze(0) - quad - logdet.unsqueeze(0)).to(torch.float32)

    print("[TRAIN] ProtoHyper...")
    for ep in range(1, PH_EPOCHS + 1):
        ph.train()
        tot_loss = 0.0; seen = 0
        for z, y in syn_dl:
            z = z.to(DEV); y = y.to(DEV)
            with torch.no_grad(): base = nb_diag_logits_z(z)
            logits = base + ph(z)
            loss = ce(logits, y)
            opt_ph.zero_grad(); loss.backward(); opt_ph.step()
            tot_loss += loss.item() * z.size(0); seen += z.size(0)
        print(f" [PH] {ep:02d}/{PH_EPOCHS} | loss={tot_loss/seen:.4f}")

    # ---------------- LDA ----------------
    Sp = shrink_cov(Sigma_pool_z, 0.05) + 1e-6 * torch.eye(k, dtype=torch.float64)
    W64 = torch.linalg.solve(Sp, mu_z.t())
    b64 = (-0.5 * (mu_z * W64.t()).sum(1) + logpri)

    W32 = W64.to(torch.float32).to(DEV)
    b32 = b64.to(torch.float32).to(DEV)

    def lda_logits_z(z): return z @ W32 + b32

    # ---------------- QDA ----------------
    def qda_logits_z(z):
        dev = z.device
        z64 = z.to(torch.float64)
        B = z64.size(0)
        scores = torch.empty(B, C, dtype=torch.float64, device=dev)
        for c in range(C):
            dif = z64 - mu_z[c].to(dev)
            y = torch.cholesky_solve(dif.t(), L_list[c].to(dev)).t()
            quad = 0.5 * (dif * y).sum(1)
            scores[:, c] = logpri[c].to(dev) - quad - logdets[c].to(dev)
        return scores.to(torch.float32)

    # ---------------- Eval on AGNEWS ----------------
    print("[EVAL] AGNEWS test...")
    dl_test = load_agnews_test()

    R = build_rp_matrix(FEAT_DIM, K_RP, RP_SEED).to(torch.float32).to(DEV)

    tot = 0
    corr_nb = corr_lda = corr_qda = corr_fm = corr_ph = 0

    with torch.no_grad():
        for X, y in dl_test:
            X = X.to(DEV); y = y.to(DEV)
            z = X @ R

            nb_logits = nb_diag_logits_z(z)
            lda_logits = lda_logits_z(z)
            qda_logits = qda_logits_z(z)
            fm_logits = fm(z)
            ph_logits = nb_logits + ph(z)

            corr_nb += (nb_logits.argmax(1) == y).sum().item()
            corr_lda += (lda_logits.argmax(1) == y).sum().item()
            corr_qda += (qda_logits.argmax(1) == y).sum().item()
            corr_fm += (fm_logits.argmax(1) == y).sum().item()
            corr_ph += (ph_logits.argmax(1) == y).sum().item()
            tot += y.size(0)

    acc_nb = 100 * corr_nb / tot
    acc_lda = 100 * corr_lda / tot
    acc_qda = 100 * corr_qda / tot
    acc_fm = 100 * corr_fm / tot
    acc_ph = 100 * corr_ph / tot

    elapsed = (time.time() - t0) / 60.0
    peak = torch.cuda.max_memory_allocated()/(1024**3) if DEVICE=="cuda" else 0.0

    print("\n===== GH-OFL AGNEWS — FINAL RESULTS =====")
    print(f"NB_diag (z): {acc_nb:.2f}%")
    print(f"LDA     (z): {acc_lda:.2f}%")
    print(f"QDA_full(z): {acc_qda:.2f}%")
    print(f"FisherMix:   {acc_fm:.2f}%")
    print(f"ProtoHyper:  {acc_ph:.2f}%")
    print("----------------------------------------")
    print(f"End-to-end time: {elapsed:.2f} min")
    print(f"Peak GPU memory: {peak:.2f} GB")

    # ---------------- HEAD COST ----------------
    print("\n===== HEAD COST (ms/sample) =====")
    cost_nb = measure_head_cost(lambda z: nb_diag_logits_z(z), k, DEVICE)
    cost_lda = measure_head_cost(lambda z: lda_logits_z(z), k, DEVICE)
    cost_qda = measure_head_cost(lambda z: qda_logits_z(z), k, DEVICE)
    cost_fm = measure_head_cost(lambda z: fm(z), k, DEVICE)
    cost_ph = measure_head_cost(lambda z: nb_diag_logits_z(z)+ph(z), k, DEVICE)

    print(f"NB_diag:    {cost_nb:.4f} ms")
    print(f"LDA:        {cost_lda:.4f} ms")
    print(f"QDA_full:   {cost_qda:.4f} ms")
    print(f"FisherMix:  {cost_fm:.4f} ms")
    print(f"ProtoHyper: {cost_ph:.4f} ms")
    print("=================================\n")

if __name__ == "__main__":
    main()





