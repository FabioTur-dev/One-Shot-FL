#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GH_OFL_language_server.py

Server-side GH-OFL (NLP) con FisherMix + Proto-Hyper su DistilBERT RP:
- Dataset: SST2, AG_NEWS, DBPEDIA_14
- Client layout: ./oneshot_bench/{DATASET}/0.5/clients/FedCGS/client_XX/client.pt
- Metodi: NB_paper(z), NB_diag(z), LDA(z_f), FisherMix(z_f), ProtoHyper(z_f)
"""

import os, glob, math, time, warnings, random, platform
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

# =========================
# Config & device
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEV = torch.device(DEVICE)
IS_WIN = platform.system().lower().startswith("win")
PIN_MEMORY = torch.cuda.is_available() and (not IS_WIN)
NUM_WORKERS = 0 if IS_WIN else 2

BATCH_TEST = 512 if not IS_WIN else 256

LDA_SHRINK = 0.05
SYN_SHRINK = 0.05
QDA_SHRINK = 0.07
FISHER_ENERGY_DEFAULT = 0.999
FISHER_MAX_K_DEFAULT = 128

FM_EPOCHS = 20
PH_EPOCHS = 20
LR = 1e-3
WD_FM = 2e-4
WD_PH = 5e-4
MAX_NORM = 2.0
KD_T = 2.0
KD_ALPHA = 0.5
TEACHER_BLEND = (0.7, 0.3)  # (LDA_zf, NB_zf)

# Sintetici per dataset
SYN_NS_PER_CLASS_MAP = {
    "SST2": 8000,
    "AG_NEWS": 6000,
    "DBPEDIA_14": 3000,
}

# Label smoothing per dataset
LS_MAP = {
    "SST2": 0.05,
    "AG_NEWS": 0.10,
    "DBPEDIA_14": 0.10,
}

# Cartella dei client (coerente col client-side)
ROOT_TPL = "./oneshot_bench/{DATASET}/0.5/clients/FedCGS"

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", message="Arguments other than a weight enum or `None` for 'weights' are deprecated")
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


# =========================
# Utils
# =========================
def set_seed(s=42):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def list_clients_in(folder: str) -> List[str]:
    """
    Supporta:
    - layout piatto:   folder/client_00.pt
    - layout annidato: folder/client_00/client.pt  (quello del tuo client)
    """
    flat = sorted(glob.glob(os.path.join(folder, "client_*.pt")))
    nested = sorted(glob.glob(os.path.join(folder, "client_*", "client.pt")))
    return flat + nested


def safe_load(p: str) -> Dict:
    try:
        return torch.load(p, map_location="cpu")
    except Exception as e:
        print(f"[WARN] skip {p}: {e}")
        return {}


def shrink_cov(S: torch.Tensor, alpha: float = 0.05):
    d = S.shape[-1]
    tr = float(torch.trace(S))
    I = torch.eye(d, dtype=S.dtype)
    return (1.0 - alpha) * S + alpha * (tr / d) * I


def standardize_logits_per_row(logits: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = logits.mean(dim=1, keepdim=True)
    s = logits.std(dim=1, keepdim=True).clamp_min(eps)
    return (logits - m) / s


# =========================
# Dataset test (HuggingFace)
# =========================
def load_eval_dataset_and_encoder(ds_name: str, model_name: str, max_len: int = 128):
    ds = ds_name.upper()
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name).eval().to(DEV)

    if ds == "SST2":
        d = load_dataset("glue", "sst2")["validation"]
        texts, labels = d["sentence"], d["label"]
        C = 2
    elif ds == "AG_NEWS":
        d = load_dataset("ag_news")["test"]
        texts, labels = d["text"], d["label"]
        C = 4
    elif ds == "DBPEDIA_14":
        d = load_dataset("dbpedia_14")["test"]
        texts, labels = d["content"], d["label"]
        C = 14
    else:
        raise ValueError(f"Dataset non supportato: {ds_name}")

    class TextDS(Dataset):
        def __init__(self, texts, labels):
            self.texts = list(texts)
            self.labels = list(labels)
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            return self.texts[i], int(self.labels[i])

    def collate(batch):
        tx = [b[0] for b in batch]
        y = torch.tensor([b[1] for b in batch], dtype=torch.long)
        t = tok(
            tx, padding=True, truncation=True, max_length=max_len,
            return_tensors="pt",
        )
        return {k: v.to(DEV, non_blocking=True) for k, v in t.items()}, y

    dl = DataLoader(
        TextDS(texts, labels),
        batch_size=BATCH_TEST,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        collate_fn=collate,
    )
    d_hidden = enc.config.hidden_size

    @torch.no_grad()
    def fe_cls(batch_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = enc(**batch_inputs)
        return out.last_hidden_state[:, 0, :].to(torch.float32)

    return dl, fe_cls, d_hidden, C


# =========================
# Aggregazione stats (z)
# =========================
def aggregate_from_clients(paths: List[str]) -> Dict:
    meta0 = None
    A_sum = SUMSQ_sum = B_sum = None
    N_sum = None
    for p in paths:
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
            A_sum     = torch.zeros_like(A, dtype=torch.float64)
            SUMSQ_sum = torch.zeros_like(S2, dtype=torch.float64)
            N_sum     = torch.zeros_like(N, dtype=torch.long)
            B_sum     = torch.zeros_like(B, dtype=torch.float64)
        A_sum += A
        SUMSQ_sum += S2
        N_sum += N
        B_sum += B

    if A_sum is None:
        raise RuntimeError("Nessun file valido")

    C, k = A_sum.shape
    N_total = int(N_sum.sum().item())
    priors = (N_sum / float(N_total)).to(torch.float64)

    mu_z = A_sum / N_sum.clamp_min(1).unsqueeze(1)  # [C,k]
    Ez2  = SUMSQ_sum / N_sum.clamp_min(1).unsqueeze(1)
    var_diag = (Ez2 - mu_z.pow(2)).clamp_min(1e-9)

    m_all = (A_sum.sum(dim=0) / float(N_total)).unsqueeze(1)  # [k,1]
    Sigma_pool_z = (B_sum - (m_all @ m_all.t()) * float(N_total)) / max(1, N_total - 1)

    return dict(
        meta=meta0, C=C, k=k, N_total=N_total,
        priors=priors, mu_z=mu_z, var_diag=var_diag,
        Sigma_pool_z=Sigma_pool_z,
    )


# =========================
# Fisher subspace
# =========================
def fisher_subspace_by_energy(mu_z, priors, Sigma_pool_z, alpha=0.05, energy=0.999, max_k=128):
    C, k = mu_z.shape
    mbar = (priors.unsqueeze(1) * mu_z).sum(dim=0, keepdim=True).to(torch.float64)
    Sb = torch.zeros(k, k, dtype=torch.float64)
    for j in range(C):
        dm = (mu_z[j:j+1].to(torch.float64) - mbar)
        Sb += float(priors[j].item()) * (dm.t() @ dm)
    Sw = Sigma_pool_z.to(torch.float64)
    tr = float(torch.trace(Sw))
    Sw = (1.0 - alpha) * Sw + alpha * (tr / k) * torch.eye(k, dtype=torch.float64)
    M = torch.linalg.solve(Sw, Sb)
    evals, evecs = torch.linalg.eig(M)
    evals = evals.real.clamp_min(0)
    evecs = evecs.real
    idx = torch.argsort(evals, descending=True)
    evals = evals[idx]
    evecs = evecs[:, idx]
    if evals.numel() == 0 or float(evals.sum().item()) <= 0:
        r = min(2, k)
        V = evecs[:, :r] if evecs.numel() > 0 else torch.eye(k, r, dtype=torch.float64)
        return V.contiguous(), r
    csum = torch.cumsum(evals, dim=0)
    tot = float(csum[-1].item())
    r = int((csum / max(tot, 1e-12) >= energy).nonzero(as_tuple=True)[0][0].item() + 1)
    r = max(2, min(r, max_k, k))
    V = evecs[:, :r].contiguous()
    return V, r


# =========================
# Logit helpers
# =========================
def nb_spherical_logits_zf_fn(mu_zf64: torch.Tensor, Sigma_pool_zf64: torch.Tensor, logpri64: torch.Tensor):
    k = mu_zf64.shape[1]
    sigma2 = float(torch.trace(Sigma_pool_zf64).item() / float(k))
    mu = mu_zf64.to(torch.float32).to(DEV)
    logp = logpri64.to(torch.float32).to(DEV)
    mu2 = (mu * mu).sum(dim=1)
    mu_t = mu.t().contiguous()
    inv2 = 0.5 / float(max(sigma2, 1e-12))
    logdet = 0.5 * k * math.log(max(sigma2, 1e-12))

    def _fn(Zf32: torch.Tensor) -> torch.Tensor:
        Z = Zf32.to(DEV, dtype=torch.float32)
        x2 = (Z * Z).sum(dim=1)
        xmu = Z @ mu_t
        quad = inv2 * (x2.unsqueeze(1) - 2.0 * xmu + mu2.unsqueeze(0))
        return (logp.unsqueeze(0) - quad - logdet)
    return _fn


def lda_logits_zf_fn(mu_zf64: torch.Tensor, Sz_pool64: torch.Tensor, logpri64: torch.Tensor):
    Sz = shrink_cov(Sz_pool64, alpha=LDA_SHRINK) + 1e-6 * torch.eye(Sz_pool64.shape[0], dtype=torch.float64)
    W64 = torch.linalg.solve(Sz, mu_zf64.t())  # [kf,C]
    b64 = (-0.5 * (mu_zf64 * (W64.t())).sum(dim=1) + logpri64.to(torch.float64))
    W32 = W64.to(torch.float32).to(DEV)
    b32 = b64.to(torch.float32).to(DEV)

    def _fn(Zf32: torch.Tensor) -> torch.Tensor:
        Z = Zf32.to(torch.float32).to(DEV)
        return Z @ W32 + b32
    return _fn, W64, b64


# =========================
# Heads
# =========================
class LinearHead(nn.Module):
    def __init__(self, in_dim, C):
        super().__init__()
        self.fc = nn.Linear(in_dim, C, bias=True)
    def forward(self, z):
        return self.fc(z)


class ProtoHyperResidual(nn.Module):
    def __init__(self, in_dim, C, rank=128):
        super().__init__()
        self.U = nn.Linear(in_dim, rank, bias=False)
        self.V = nn.Linear(rank, C, bias=True)
    def forward(self, z):
        return self.V(self.U(z))


# =========================
# Dataset sintetico in z_f
# =========================
class SynthZF_ClassCondPerClass(Dataset):
    """
    z_f ~ N(mu_zf[c], tau_c[c]^2 * Sigma_c^f),
    con Sigma_c^f = V^T diag(var_c^z) V (shrinked)
    """
    def __init__(self, mu_zf64: torch.Tensor, V64: torch.Tensor, var_diag: torch.Tensor,
                 ns_per_class: int, shrink_alpha: float = SYN_SHRINK, seed: int = 123):
        super().__init__()
        self.mu = mu_zf64.to(torch.float64).cpu()
        self.C, self.kf = self.mu.shape
        self.V = V64.to(torch.float64).cpu()
        self.var_diag = var_diag.to(torch.float64).cpu()
        self.ns = int(ns_per_class)
        self.total = self.C * self.ns

        # GENERATOR SU DEV (CUDA) — FIX
        self.gen = torch.Generator(device=DEV).manual_seed(seed)

        self.L_list = []
        self.tau_c = []

        var_pool = self.var_diag.mean(dim=0).clamp_min(1e-9)
        mean_var_pool = float(var_pool.mean().item())
        mean_var_cls = self.var_diag.mean(dim=1).clamp_min(1e-9)

        for c in range(self.C):
            D = torch.diag(self.var_diag[c])
            Sf = self.V.t() @ D @ self.V
            Sf = shrink_cov(Sf, alpha=shrink_alpha) + 1e-6 * torch.eye(self.kf, dtype=torch.float64)
            Lc = torch.linalg.cholesky(Sf)
            self.L_list.append(Lc.to(torch.float32).to(DEV))
            tau = float((mean_var_cls[c] / mean_var_pool).sqrt().item())
            tau = max(0.7, min(1.4, tau))
            self.tau_c.append(torch.tensor(tau, dtype=torch.float32, device=DEV))

        self.mu_f = self.mu.to(torch.float32).to(DEV)

    def __len__(self): return self.total

    def __getitem__(self, idx):
        c = int(idx % self.C)
        eps = torch.randn(self.kf, generator=self.gen, device=DEV)
        zf = self.mu_f[c] + self.tau_c[c] * (eps @ self.L_list[c].t())
        y = c
        return zf, y


# =========================
# RP matrix from meta
# =========================
def rp_from_meta(meta: Dict) -> torch.Tensor:
    d = int(meta["feature_dim"])
    k = int(meta["rp_dim"])
    seed = int(meta["rp_seed"])
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.randn(d, k, generator=g) / math.sqrt(d)
    return R.to(torch.float32).to(DEV)


# =========================
# Run per dataset
# =========================
def run_one_dataset(ds_name: str, stats_root: str):
    print(f"\n=== DATASET: {ds_name} | Stats: {stats_root} ===")
    files = list_clients_in(stats_root)
    if not files:
        print(f"[ERR] Nessun client in {stats_root}")
        return None

    agg = aggregate_from_clients(files)
    C, k = agg["C"], agg["k"]
    mu_z = agg["mu_z"]
    pri = agg["priors"]
    var_diag = agg["var_diag"]
    Sigma_pool_z = agg["Sigma_pool_z"]
    logpri = torch.log(pri.clamp_min(1e-12))

    # Fisher subspace
    if ds_name.upper() in ("SST2", "AG_NEWS"):
        energy = 0.9995
        max_k = max(16, min(FISHER_MAX_K_DEFAULT, k))
    else:
        energy = FISHER_ENERGY_DEFAULT
        max_k = min(FISHER_MAX_K_DEFAULT, k)

    V64, kf = fisher_subspace_by_energy(mu_z, pri, Sigma_pool_z,
                                        alpha=LDA_SHRINK, energy=energy, max_k=max_k)
    print(f"[INFO] Fisher subspace in z: kf={kf}")
    mu_zf64 = (mu_z.to(torch.float64) @ V64)
    Sig_pool_zf64 = (V64.t() @ Sigma_pool_z.to(torch.float64) @ V64)

    lda_logits_fn, W64_lda, b64_lda = lda_logits_zf_fn(mu_zf64, Sig_pool_zf64, logpri)
    nbz_logits_fn = nb_spherical_logits_zf_fn(mu_zf64, Sig_pool_zf64, logpri)

    # Whitening in z_f
    evals, evecs = torch.linalg.eigh(
        shrink_cov(Sig_pool_zf64, alpha=SYN_SHRINK) + 1e-6 * torch.eye(kf, dtype=torch.float64)
    )
    A_wh = (evecs @ torch.diag(evals.clamp_min(1e-9).rsqrt()) @ evecs.t()).to(torch.float32).to(DEV)

    # Dataset sintetico
    syn_ns = SYN_NS_PER_CLASS_MAP.get(ds_name.upper(), 3000)
    syn_ds = SynthZF_ClassCondPerClass(mu_zf64, V64, var_diag, ns_per_class=syn_ns, seed=123)
    syn_dl = DataLoader(
        syn_ds,
        batch_size=2048 if not IS_WIN else 1024,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    steps = max(1, math.ceil(len(syn_ds) / (2048 if not IS_WIN else 1024)))

    # Modelli
    fm = LinearHead(kf, C).to(DEV)
    ph = ProtoHyperResidual(kf, C, rank=min(128, kf)).to(DEV)
    opt_fm = torch.optim.AdamW(fm.parameters(), lr=LR, weight_decay=WD_FM)
    opt_ph = torch.optim.AdamW(ph.parameters(), lr=LR, weight_decay=WD_PH)
    sched_fm = torch.optim.lr_scheduler.CosineAnnealingLR(opt_fm, T_max=FM_EPOCHS)
    sched_ph = torch.optim.lr_scheduler.CosineAnnealingLR(opt_ph, T_max=PH_EPOCHS)

    # Warm-start FisherMix da LDA(z_f)
    with torch.no_grad():
        fm.fc.weight.copy_(W64_lda.to(torch.float32).t().to(DEV))
        fm.fc.bias.copy_(b64_lda.to(torch.float32).to(DEV))

    ce_hard = nn.CrossEntropyLoss(label_smoothing=LS_MAP.get(ds_name.upper(), 0.10))

    # ---- Train FisherMix
    for ep in range(1, FM_EPOCHS + 1):
        fm.train()
        tot = 0.0
        seen = 0
        it = iter(syn_dl)
        for _ in range(steps):
            zf, yb = next(it)
            zf = zf.to(DEV, non_blocking=True)
            yb = yb.to(DEV, non_blocking=True)
            zf_w = zf @ A_wh
            opt_fm.zero_grad(set_to_none=True)
            loss = ce_hard(fm(zf_w), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(fm.parameters(), MAX_NORM)
            opt_fm.step()
            bs = zf.size(0)
            tot += loss.item() * bs
            seen += bs
        sched_fm.step()
        if ep % 5 == 0:
            print(f"[FM] ep {ep:02d}/{FM_EPOCHS} | loss={tot/seen:.4f}")

    # ---- Train Proto-Hyper
    a_lda, a_nb = TEACHER_BLEND
    T = float(KD_T)
    alpha = float(KD_ALPHA)
    for ep in range(1, PH_EPOCHS + 1):
        ph.train()
        tot = 0.0
        seen = 0
        it = iter(syn_dl)
        for _ in range(steps):
            zf, yb = next(it)
            zf = zf.to(DEV, non_blocking=True)
            yb = yb.to(DEV, non_blocking=True)
            with torch.no_grad():
                tea = a_lda * lda_logits_fn(zf) + a_nb * nbz_logits_fn(zf)
                tea = standardize_logits_per_row(tea)
                base = standardize_logits_per_row(nbz_logits_fn(zf))
            logits = base + ph(zf)
            loss_kd = F.kl_div(
                F.log_softmax(logits / T, dim=1),
                F.softmax(tea / T, dim=1),
                reduction="batchmean",
            ) * (T * T)
            loss_ce = F.cross_entropy(logits, yb, label_smoothing=LS_MAP.get(ds_name.upper(), 0.10))
            loss = alpha * loss_kd + (1.0 - alpha) * loss_ce
            opt_ph.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ph.parameters(), MAX_NORM)
            opt_ph.step()
            bs = zf.size(0)
            tot += loss.item() * bs
            seen += bs
        sched_ph.step()
        if ep % 5 == 0:
            print(f"[PH] ep {ep:02d}/{PH_EPOCHS} | loss=ok")

    # ========================
    # Valutazione su test set
    # ========================
    dl, fe_cls, d_hidden, C_eval = load_eval_dataset_and_encoder(
        ds_name,
        agg["meta"].get("weights_tag", "distilbert-base-uncased"),
        max_len=128,
    )
    assert C_eval == C, "Mismatch classi tra client e test set"

    R = rp_from_meta(agg["meta"])
    V_dev32 = V64.to(torch.float32).to(DEV)

    logpri_dev = torch.log(pri.clamp_min(1e-12)).to(torch.float64)

    tot = 0
    cnt_nb_paper = cnt_nb_diag = cnt_lda = cnt_fm = cnt_ph = 0

    # Parametri NB_paper (CPU, ma li porto sul device di Z quando servono)
    with torch.no_grad():
        sigma2_z = float(torch.trace(Sigma_pool_z).item() / float(k))
        mu_nb = mu_z.to(torch.float64)
        logp_nb = logpri.to(torch.float64)

    @torch.no_grad()
    def nb_paper_logits_z(Z: torch.Tensor):
        # Z: [B,k] su DEV
        Z64 = Z.to(torch.float64)                   # stesso device di Z
        mu = mu_nb.to(Z64.device)                  # porta mu_nb sul device di Z
        lp = logp_nb.to(Z64.device)                # porta logp_nb sul device di Z
        k_ = Z64.shape[1]
        inv2 = 0.5 / float(max(sigma2_z, 1e-12))
        logdet = 0.5 * k_ * math.log(max(sigma2_z, 1e-12))
        x2 = (Z64 * Z64).sum(dim=1)
        mu2 = (mu * mu).sum(dim=1)
        xmu = Z64 @ mu.t()
        quad = inv2 * (x2.unsqueeze(1) - 2.0 * xmu + mu2.unsqueeze(0))
        return (lp.unsqueeze(0) - quad - logdet)

    @torch.no_grad()
    def nb_diag_logits_z(Z: torch.Tensor):
        # Z: [B,k] su DEV
        Z64 = Z.to(torch.float64)
        mu = mu_z.to(torch.float64).to(Z64.device)
        var = var_diag.to(torch.float64).clamp_min(1e-9).to(Z64.device)
        logp = logpri_dev.to(torch.float64).to(Z64.device)
        dif = Z64.unsqueeze(1) - mu.unsqueeze(0)
        inv = 1.0 / var
        quad = 0.5 * (dif * dif * inv.unsqueeze(0)).sum(dim=2)
        logdet = 0.5 * torch.log(var).sum(dim=1)
        return (logp.unsqueeze(0) - quad - logdet.unsqueeze(0))

    with torch.no_grad():
        for batch_inputs, y in dl:
            X = fe_cls(batch_inputs)          # [B,d_hidden] su DEV
            Z = X @ R                         # [B,k] su DEV
            Zf = Z @ V_dev32                  # [B,kf] su DEV

            logits_nb_p = nb_paper_logits_z(Z)
            cnt_nb_paper += int((logits_nb_p.argmax(1).cpu() == y).sum())

            logits_nb_d = nb_diag_logits_z(Z)
            cnt_nb_diag += int((logits_nb_d.argmax(1).cpu() == y).sum())

            logits_lda = lda_logits_fn(Zf)
            cnt_lda += int((logits_lda.argmax(1).cpu() == y).sum())

            pred_fm = fm(Zf @ A_wh).argmax(1).cpu()
            cnt_fm += int((pred_fm == y).sum())

            base = standardize_logits_per_row(nbz_logits_fn(Zf))
            pred_ph = (base + ph(Zf)).argmax(1).cpu()
            cnt_ph += int((pred_ph == y).sum())

            tot += y.numel()

    res = dict(
        NB_paper=100.0 * cnt_nb_paper / float(tot),
        NB_diag=100.0 * cnt_nb_diag / float(tot),
        LDA=100.0 * cnt_lda / float(tot),
        FisherMix=100.0 * cnt_fm / float(tot),
        ProtoHyper=100.0 * cnt_ph / float(tot),
        k=k,
        k_f=kf,
    )

    print("\n================== FINAL SUMMARY (NLP / RP + Fisher) ==================")
    print(f"Dataset: {ds_name} | C={C} | k(RP)={k} | k_f(Fisher)={kf}")
    print(f"NB_paper (z) : {res['NB_paper']:7.2f}%")
    print(f"NB_diag  (z) : {res['NB_diag']:7.2f}%")
    print(f"LDA   (z_f)  : {res['LDA']:7.2f}%")
    print(f"FisherMix    : {res['FisherMix']:7.2f}%")
    print(f"Proto-Hyper  : {res['ProtoHyper']:7.2f}%")
    print("=======================================================================")

    return res


# =========================
# MAIN
# =========================
def main():
    set_seed(42)
    print(f"[INFO] Device: {DEVICE}")

    datasets = ["SST2", "AG_NEWS", "DBPEDIA_14"]
    all_out = {}

    for ds in datasets:
        stats_root = ROOT_TPL.format(DATASET=ds.upper())
        out = run_one_dataset(ds, stats_root)
        if out is not None:
            all_out[ds] = out

    if all_out:
        print("\n===================== SUMMARY ALL DATASETS =====================")
        for ds in datasets:
            if ds not in all_out:
                continue
            r = all_out[ds]
            print(
                f"{ds:11s} | NB_paper: {r['NB_paper']:7.2f}% | "
                f"NB_diag: {r['NB_diag']:7.2f}% | LDA: {r['LDA']:7.2f}% | "
                f"FisherMix: {r['FisherMix']:7.2f}% | ProtoHyper: {r['ProtoHyper']:7.2f}% "
                f"| k={r['k']}, k_f={r['k_f']}"
            )
        print("===============================================================")


if __name__ == "__main__":
    main()






