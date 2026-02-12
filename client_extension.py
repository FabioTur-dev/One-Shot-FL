#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
client_extension.py

Genera i payload client per:
- FedCGS  : (A_per_class, N_per_class, B_global, S_per_class opzionale, D_per_class diag)
- FedPFT  : GMM diag per classe (lista di componenti), robusto e stabile
- DENSE   : testa lineare per client via Ridge (forma chiusa) con bias, streaming (no cache X)

Dataset: CIFAR10, CIFAR100, SVHN
Backbone: ResNet-18 IMAGENET1K_V1, fc=Identity, eval, input 224x224 (fair ICLR)

Ottimizzazioni chiave:
- Una sola passata sul train per α (smistamento verso i 10 client durante l'iterazione).
- Accumuli in float32 (cast a float64 solo al salvataggio se serve).
- DENSE in forma chiusa (no training epoch), usando statistiche (ZZ, ZY) di un vettore z=[f;1].
- GMM con k-means++ numericamente stabile; cap sulla memoria con reservoir per classe.

Output:
./oneshot_bench/{DATASET}/{ALPHA}/clients/FedCGS/client_XX/fedcgs_client.pt
./oneshot_bench/{DATASET}/{ALPHA}/clients/FedPFT/client_XX/fedpft_client.pt
./oneshot_bench/{DATASET}/{ALPHA}/clients/DENSE/client_XX/local_head.pt
"""

import os, math, random, warnings, json
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models

# ========================
# Config
# ========================
DATA_ROOT   = "./data"
OUT_ROOT    = "./oneshot_bench"
NUM_CLIENTS = 10
ALPHAS      = (0.05, 0.1, 0.5)
SEED        = 42

DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY  = torch.cuda.is_available()
NUM_WORKERS = 2  # Windows-friendly
BATCH_SIZE  = 256 if torch.cuda.is_available() else 128

# Fair ICLR: input 224, IMAGENET1K_V1
IMG_SIZE    = 224

# FedPFT
FEDPFT_K            = 5           # n. componenti per classe (diag)
FEDPFT_MAX_PER_CLASS= 4000        # reservoir per classe per contenere memoria
FEDPFT_INIT_REPS    = 3           # ripetizioni kmeans++ (sceglie loss migliore)

# DENSE (ridge, forma chiusa)
DENSE_RIDGE_LAMBDA  = 1e-3

# Salvataggi opzionali
SAVE_S_PER_CLASS    = True        # richiesto per QDA_full lato server; metti False per più velocità/spazio
SAVE_JSON_META      = True

warnings.filterwarnings("ignore", message="You are using `torch.load` with `weights_only=False`")
warnings.filterwarnings("ignore", message="Anti-alias option is always applied for PIL Image input.")


# ========================
# Utils
# ========================
def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def svhn_map10_to_0(y):
    y = int(y)
    return 0 if y == 10 else y

class IndexDataset(Dataset):
    """Wrap dataset -> returns (img, target, idx)."""
    def __init__(self, base_ds):
        self.base = base_ds
        self.has_targets = hasattr(base_ds, "targets") or hasattr(base_ds, "labels")
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x, int(y), idx

def build_resnet18_imagenet():
    """ResNet18 ImageNet, fc=Identity; tfm 224 + Normalize; eval; channels_last+AMP safe."""
    mean, std = (0.485,0.456,0.406), (0.229,0.224,0.225)
    try:
        from torchvision.models import ResNet18_Weights
        w = ResNet18_Weights.IMAGENET1K_V1
        m = models.resnet18(weights=w)
        if hasattr(w, "meta") and isinstance(w.meta, dict):
            mean = tuple(float(x) for x in w.meta.get("mean", mean))
            std  = tuple(float(x) for x in w.meta.get("std", std))
    except Exception:
        try:    m = models.resnet18(pretrained=True)
        except: m = models.resnet18(pretrained=False)
    m.fc = nn.Identity()
    m.eval().to(DEVICE)
    m.to(memory_format=torch.channels_last)

    tfm = transforms.Compose([
        transforms.Resize(IMG_SIZE, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return m, 512, tfm, IMG_SIZE, mean, std, "IMAGENET1K_V1"

def get_train_ds(name: str, tfm):
    name = name.upper()
    if name == "CIFAR10":
        ds = datasets.CIFAR10(DATA_ROOT, train=True, transform=tfm, download=True); C=10
        labels = np.array(ds.targets, dtype=np.int64)
    elif name == "CIFAR100":
        ds = datasets.CIFAR100(DATA_ROOT, train=True, transform=tfm, download=True); C=100
        labels = np.array(ds.targets, dtype=np.int64)
    elif name == "SVHN":
        ds = datasets.SVHN(DATA_ROOT, split="train", transform=tfm,
                           target_transform=transforms.Lambda(svhn_map10_to_0), download=True); C=10
        labels = np.array(ds.labels, dtype=np.int64)
        labels = np.where(labels==10, 0, labels)
    else:
        raise ValueError(name)
    return ds, C, labels

def dirichlet_split_by_label(labels: np.ndarray, num_clients: int, alpha: float, seed: int) -> List[List[int]]:
    """Split non-IID Dirichlet per classe (fair): restituisce liste di indici per ciascun client."""
    rng = np.random.RandomState(seed)
    C = int(labels.max()) + 1
    client_indices = [[] for _ in range(num_clients)]
    for c in range(C):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        # proporzioni per client
        if alpha == float('inf'):
            # IID
            props = np.ones(num_clients) / num_clients
        else:
            props = rng.dirichlet([alpha]*num_clients)
        counts = (props * len(idx)).astype(int)
        # aggiusta differenze
        diff = len(idx) - counts.sum()
        for k in range(abs(diff)):
            counts[k % num_clients] += 1 if diff > 0 else -1
        # slice
        s = 0
        for i in range(num_clients):
            e = s + max(0, counts[i])
            if e > s:
                client_indices[i].extend(idx[s:e].tolist())
            s = e
    # shuffle finale per client
    for i in range(num_clients):
        rng.shuffle(client_indices[i])
    return client_indices

@torch.no_grad()
def extract_features(fe: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    xb = xb.to(device=DEVICE, memory_format=torch.channels_last, non_blocking=True)
    if DEVICE == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.float16):
            f = fe(xb)
    else:
        f = fe(xb)
    return f.to("cpu").to(torch.float32)  # CPU float32


# ========================
# FedPFT: GMM diag robusto
# ========================
def kmeans_pp(x: torch.Tensor, k: int, reps: int = 1, seed: int = 0) -> torch.Tensor:
    """k-means++ initialization; ritorna centri [k,d]."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    n, d = x.shape
    best_loss, best_centers = None, None
    for _ in range(reps):
        centers = torch.empty(k, d, dtype=torch.float32)
        # primo centro a caso
        idx0 = torch.randint(0, n, (1,), generator=g).item()
        centers[0] = x[idx0]
        # successivi
        dist2 = ((x - centers[0])**2).sum(1)
        for j in range(1, k):
            probs = (dist2.clamp_min(1e-12) / dist2.sum().clamp_min(1e-12))
            idx = torch.multinomial(probs, 1, generator=g).item()
            centers[j] = x[idx]
            dist2 = torch.minimum(dist2, ((x - centers[j])**2).sum(1))
        # loss
        loss = dist2.mean().item()
        if (best_loss is None) or (loss < best_loss):
            best_loss = loss; best_centers = centers.clone()
    return best_centers

def fit_gmm_diag_per_class(x: torch.Tensor, k: int, seed: int = 0) -> Dict[str, torch.Tensor]:
    """
    x: [n,d] float32
    Ritorna dict {'mu':[k,d], 'var':[k,d], 'pi':[k]} (clamped, stabili).
    Se n<k -> riduce k.
    """
    x = x.to(torch.float32, copy=True)
    n, d = x.shape
    if n == 0:
        return dict(mu=torch.zeros(0, d), var=torch.zeros(0, d), pi=torch.zeros(0))
    k = min(k, n)
    # init
    centers = kmeans_pp(x, k, reps=FEDPFT_INIT_REPS, seed=seed)
    # una singola E-step per partizionare
    # assegnazione hard
    dist2 = torch.cdist(x, centers, p=2.0)  # [n,k]
    idx = dist2.argmin(1)  # [n]
    # parametri per cluster
    mu_list, var_list, pi_list = [], [], []
    for j in range(k):
        mask = (idx == j)
        nj = int(mask.sum().item())
        if nj == 0:
            continue
        Xj = x[mask]
        muj = Xj.mean(0)
        varj = Xj.var(0, unbiased=False).clamp_min(1e-6)
        pj = nj / float(n)
        mu_list.append(muj.unsqueeze(0))
        var_list.append(varj.unsqueeze(0))
        pi_list.append(pj)
    if not mu_list:
        # fallback: tutto un cluster
        muj = x.mean(0); varj = x.var(0, unbiased=False).clamp_min(1e-6)
        return dict(mu=muj.unsqueeze(0), var=varj.unsqueeze(0), pi=torch.ones(1))
    mu = torch.cat(mu_list, 0)
    var = torch.cat(var_list, 0).clamp_min(1e-6)
    pi = torch.tensor(pi_list, dtype=torch.float32)
    pi = (pi / pi.sum().clamp_min(1e-12)).clamp_min(1e-9)
    return dict(mu=mu, var=var, pi=pi)

def reservoir_update(buf_x: Optional[torch.Tensor], x_new: torch.Tensor, cap: int, g: np.random.RandomState):
    """Reservoir per contenere al più 'cap' righe in buf_x (CPU float32)."""
    if buf_x is None:
        take = x_new[:min(cap, x_new.size(0))].to(torch.float32).cpu()
        return take
    total = buf_x.size(0)
    need = max(0, cap - total)
    if need > 0:
        add = x_new[:min(need, x_new.size(0))].to(torch.float32).cpu()
        return torch.cat([buf_x, add], 0)
    # rimpiazzi random
    for i in range(x_new.size(0)):
        j = g.randint(0, total + 1 + i)
        if j < total:
            buf_x[j] = x_new[i].to(torch.float32)
    return buf_x


# ========================
# DENSE: Ridge in forma chiusa (streaming)
# ========================
def dense_update_stats(ZZ: torch.Tensor, ZY: torch.Tensor, f: torch.Tensor, y: torch.Tensor, C: int):
    """
    z = [f; 1] -> aggiorna:
      ZZ += z^T z  (d+1,d+1)
      ZY += z^T y_onehot  (d+1,C)
    """
    d = f.size(1)
    one = torch.ones(f.size(0), 1, dtype=torch.float32)
    z = torch.cat([f, one], dim=1)  # [B, d+1]
    # z^T z
    ZZ += z.t().mm(z)
    # z^T Y
    yoh = torch.zeros(f.size(0), C, dtype=torch.float32)
    yoh.scatter_(1, y.unsqueeze(1), 1.0)
    ZY += z.t().mm(yoh)
    return ZZ, ZY

def dense_solve_ridge(ZZ: torch.Tensor, ZY: torch.Tensor, lam: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    W_aug = (ZZ + λI)^-1 ZY   -> W (d×C), b (C)
    """
    d1 = ZZ.size(0)
    I = torch.eye(d1, dtype=torch.float32)
    A = ZZ + lam * I
    W_aug = torch.linalg.solve(A, ZY)  # [d+1, C]
    W = W_aug[:-1, :]
    b = W_aug[-1, :]
    return W.contiguous(), b.contiguous()


# ========================
# Main pipeline per dataset & alpha
# ========================
@dataclass
class MetaInfo:
    dataset_name: str
    num_clients: int
    dirichlet_alpha: float
    feature_dim: int
    num_classes: int
    backbone: str
    weights_tag: str
    img_size: int
    normalization_mean: Tuple[float, float, float]
    normalization_std: Tuple[float, float, float]
    has_S_per_class: bool

def make_out_dirs(dataset: str, alpha: float):
    root = os.path.join(OUT_ROOT, dataset, str(alpha))
    for sub in ["clients/FedCGS", "clients/FedPFT", "clients/DENSE"]:
        os.makedirs(os.path.join(root, sub), exist_ok=True)
    return root

def save_fedcgs_payload(out_dir: str, client_id: int, meta: MetaInfo,
                        A: torch.Tensor, N: torch.Tensor, B: torch.Tensor,
                        S: Optional[torch.Tensor], D: torch.Tensor):
    payload = dict(
        meta=asdict(meta),
        client_id=int(client_id),
        A_per_class=A.to(torch.float64).cpu(),
        N_per_class=N.to(torch.long).cpu(),
        B_global=B.to(torch.float64).cpu(),
        S_per_class=None if S is None else S.to(torch.float64).cpu(),
        D_per_class=D.to(torch.float64).cpu(),
    )
    cdir = os.path.join(out_dir, "clients", "FedCGS", f"client_{client_id:02d}")
    os.makedirs(cdir, exist_ok=True)
    torch.save(payload, os.path.join(cdir, "fedcgs_client.pt"))

def save_fedpft_payload(out_dir: str, client_id: int, C: int,
                        gmms: List[Optional[Dict[str,torch.Tensor]]],
                        class_counts: torch.Tensor):
    payload = dict(
        client_id=int(client_id),
        gmms=gmms,
        class_counts=class_counts.to(torch.long).cpu(),
    )
    cdir = os.path.join(out_dir, "clients", "FedPFT", f"client_{client_id:02d}")
    os.makedirs(cdir, exist_ok=True)
    torch.save(payload, os.path.join(cdir, "fedpft_client.pt"))

def save_dense_payload(out_dir: str, client_id: int, W: torch.Tensor, b: torch.Tensor):
    payload = dict(W=W.to(torch.float32).cpu(), b=b.to(torch.float32).cpu())
    cdir = os.path.join(out_dir, "clients", "DENSE", f"client_{client_id:02d}")
    os.makedirs(cdir, exist_ok=True)
    torch.save(payload, os.path.join(cdir, "local_head.pt"))

def run_for_dataset(dataset: str):
    set_seed(SEED)
    fe, d, tfm, img_size, mean, std, wtag = build_resnet18_imagenet()
    backbone_tag = "resnet18"
    ds_train, C, labels = get_train_ds(dataset, tfm)
    print(f"\n[{dataset}] train samples={len(ds_train)} | C={C}")

    meta = MetaInfo(
        dataset_name=dataset, num_clients=NUM_CLIENTS, dirichlet_alpha=0.0,  # alpha verrà sovrascritto
        feature_dim=d, num_classes=C, backbone=backbone_tag, weights_tag=wtag,
        img_size=img_size, normalization_mean=tuple(float(x) for x in mean),
        normalization_std=tuple(float(x) for x in std), has_S_per_class=SAVE_S_PER_CLASS
    )

    base = IndexDataset(ds_train)
    for alpha in ALPHAS:
        # split non-IID Dirichlet per etichetta
        splits = dirichlet_split_by_label(labels, NUM_CLIENTS, alpha, SEED)
        owner = np.full(len(base), -1, dtype=np.int64)
        for cid, idxs in enumerate(splits):
            owner[idxs] = cid
        assert (owner >= 0).all(), "Owner non assegnato per qualche indice"
        meta.dirichlet_alpha = float(alpha)

        root = make_out_dirs(dataset, alpha)

        # --- inizializza accumulatori per client ---
        # FedCGS
        A  = [torch.zeros(C, d, dtype=torch.float32) for _ in range(NUM_CLIENTS)]
        N  = [torch.zeros(C, dtype=torch.long) for _ in range(NUM_CLIENTS)]
        B  = [torch.zeros(d, d, dtype=torch.float32) for _ in range(NUM_CLIENTS)]
        S  = [torch.zeros(C, d, d, dtype=torch.float32) if SAVE_S_PER_CLASS else None for _ in range(NUM_CLIENTS)]
        Dg = [torch.zeros(C, d, dtype=torch.float32) for _ in range(NUM_CLIENTS)]   # sum of squares per class

        # FedPFT reservoir per classe
        res_rs = np.random.RandomState(SEED)
        RES = [[None for _ in range(C)] for __ in range(NUM_CLIENTS)]  # RES[cid][cls] -> Tensor[n<=cap, d]
        CLS_COUNT = [torch.zeros(C, dtype=torch.long) for _ in range(NUM_CLIENTS)]

        # DENSE stats (streaming)
        ZZ = [torch.zeros(d+1, d+1, dtype=torch.float32) for _ in range(NUM_CLIENTS)]
        ZY = [torch.zeros(d+1, C, dtype=torch.float32)    for _ in range(NUM_CLIENTS)]

        # --- single pass sul train ---
        dl = DataLoader(base, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, persistent_workers=False)
        for xb, yb, ib in dl:
            # feature
            f = extract_features(fe, xb)   # [B,d] CPU float32
            ib_np = ib.numpy()
            yb_np = yb.numpy()
            # dispatch per client nel batch
            for cid in np.unique(owner[ib_np]):
                mask = (owner[ib_np] == cid)
                if not np.any(mask): continue
                fcid = f[mask]                   # [m,d]
                ycid = torch.from_numpy(yb_np[mask]).to(torch.long)  # [m]

                # ---- FedCGS accumuli ----
                # global
                B[cid] += fcid.t().mm(fcid)
                # per-classe
                for cls in ycid.unique().tolist():
                    cmask = (ycid == cls)
                    F = fcid[cmask]                                # [m_c, d]
                    A[cid][cls] += F.sum(0)
                    N[cid][cls] += int(cmask.sum().item())
                    if S[cid] is not None:
                        S[cid][cls] += F.t().mm(F)
                    Dg[cid][cls] += (F * F).sum(0)

                # ---- FedPFT reservoir ----
                for cls in ycid.unique().tolist():
                    cmask = (ycid == cls)
                    F = fcid[cmask]
                    RES[cid][cls] = reservoir_update(RES[cid][cls], F, FEDPFT_MAX_PER_CLASS, res_rs)
                    CLS_COUNT[cid][cls] += int(F.size(0))

                # ---- DENSE stats ----
                ZZ[cid], ZY[cid] = dense_update_stats(ZZ[cid], ZY[cid], fcid, ycid, C)

        # --- salva per client ---
        for cid in range(NUM_CLIENTS):
            # FedCGS
            save_fedcgs_payload(root, cid, meta, A[cid], N[cid], B[cid], S[cid], Dg[cid])

            # FedPFT
            gmms: List[Optional[Dict[str,torch.Tensor]]] = []
            for cls in range(C):
                Xi = RES[cid][cls]
                if Xi is None or Xi.size(0) == 0:
                    gmms.append(None)
                else:
                    gmms.append(fit_gmm_diag_per_class(Xi, FEDPFT_K, seed=SEED+cid*131+cls))
            save_fedpft_payload(root, cid, C, gmms, CLS_COUNT[cid])

            # DENSE
            W, b = dense_solve_ridge(ZZ[cid], ZY[cid], DENSE_RIDGE_LAMBDA)
            save_dense_payload(root, cid, W, b)

            print(f"[OK] {dataset} α={alpha} | client {cid:02d} salvato.")

        if SAVE_JSON_META:
            with open(os.path.join(root, "meta.json"), "w") as f:
                json.dump(asdict(meta), f, indent=2)
        print(f"[DONE] {dataset} α={alpha} -> {os.path.abspath(root)}")


def main():
    set_seed(SEED)
    for ds in ["CIFAR10", "CIFAR100", "SVHN"]:
        run_for_dataset(ds)

if __name__ == "__main__":
    main()






