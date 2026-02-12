#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
language_server.py  (Dense / Co-Boost baselines — server-side definitivo)

- Carica i checkpoint dei client salvati da language_client.py
- Ricostruisce z_train e z_test (stesso encoder DistilBERT + RP)
- Usa come pool U = z_train ∪ z_test per fare Knowledge Distillation:
    * Dense: uno studente MLP che imita l'ensemble dei client (teacher)
    * Co-Boosting: due studenti, il secondo si concentra sugli errori del primo
- Valuta su z_test con le label vere.

Assume che language_client.py sia nella stessa repo e che abbia già:
    - load_splits, build_encoder, compute_z_matrix, build_rp_matrix
    - LinearHead, DATASETS, ALPHAS, K_RP, MODEL_NAME, SEED, DEVICE, RP_SEED, set_seed
"""

import os
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ========= IMPORT dal client =========
from language_client import (
    load_splits,
    build_encoder,
    compute_z_matrix,
    build_rp_matrix,
    LinearHead,
    DATASETS,   # es: ["BANKING77", "CLINC150", "AG_NEWS", "DBPEDIA_14", "SST2"]
    ALPHAS,     # es: [0.5, 0.01]
    K_RP,
    MODEL_NAME,
    SEED,
    DEVICE,
    RP_SEED,
    set_seed,
)

# ========= CONFIG server-side =========
ROOT_TPL = "./oneshot_bench/{DATASET}/{ALPHA}/clients/DenseCB"

BATCH_SIZE_STUDENT = 256
LR_DENSE = 5e-3
WD_DENSE = 1e-4
EPOCHS_DENSE = 60

EPOCHS_COBOOST_1 = 60
EPOCHS_COBOOST_2 = 40
LR_COBOOST = 5e-3
WD_COBOOST = 1e-4

KD_TEMPERATURE = 2.0  # temperatura per le soft-label
MAX_NORM = 5.0        # grad clipping

DEVICE_TORCH = torch.device(DEVICE)


# ========= MLP Student (più capace della sola Linear) =========
class MLPStudent(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, z):
        return self.net(z)


# ========= Utility per i client heads =========
def load_client_heads(dataset_name: str, alpha: float, C_expected: int) -> List[nn.Module]:
    """
    Carica tutti i client_* per il dataset e la specifica alpha,
    usando lo stesso path del client-side.
    """
    root = ROOT_TPL.format(DATASET=dataset_name.upper(), ALPHA=str(alpha))
    heads: List[nn.Module] = []

    if not os.path.isdir(root):
        raise FileNotFoundError(f"[ERR] Directory client heads non trovata: {root}")

    i = 0
    while True:
        client_dir = os.path.join(root, f"client_{i:02d}")
        if not os.path.isdir(client_dir):
            break
        p = os.path.join(client_dir, "head.pt")
        if not os.path.isfile(p):
            break

        payload = torch.load(p, map_location="cpu")
        meta = payload.get("meta", {})
        state_dict = payload["state_dict"]

        C_meta = meta.get("num_classes", None)
        if C_meta is not None and C_meta != C_expected:
            raise ValueError(
                f"[ERR] num_classes mismatch per {p}: meta={C_meta}, atteso={C_expected}"
            )

        head = LinearHead(K_RP, C_expected)
        head.load_state_dict(state_dict)
        head.to(DEVICE_TORCH)
        head.eval()
        for param in head.parameters():
            param.requires_grad_(False)

        heads.append(head)
        i += 1

    if len(heads) == 0:
        raise RuntimeError(f"[ERR] Nessun client head trovato in {root}")

    print(f"[INFO] {dataset_name} | alpha={alpha}: caricati {len(heads)} client heads da {root}")
    return heads


@torch.no_grad()
def compute_teacher_logits(client_heads: List[nn.Module],
                           Z: torch.Tensor,
                           C: int) -> torch.Tensor:
    """
    Somma i logit di tutti i client in ensemble.
    Z è su CPU; i modelli vengono portati su DEVICE.
    Ritorna un tensore [N, C] su CPU.
    """
    Z_dev = Z.to(DEVICE_TORCH, non_blocking=True)
    logits_sum = torch.zeros(Z_dev.size(0), C, device=DEVICE_TORCH)

    for h in client_heads:
        logits_sum += h(Z_dev)

    return logits_sum.cpu()


@torch.no_grad()
def evaluate_client_ensemble_on_Z(client_heads: List[nn.Module],
                                  Z: torch.Tensor,
                                  y: torch.Tensor) -> float:
    """
    Accuracy dell'ensemble dei client su Z, usando le label vere y.
    """
    C = int(y.max().item()) + 1
    logits = compute_teacher_logits(client_heads, Z, C)  # [N,C] su CPU
    pred = logits.argmax(dim=1)
    correct = (pred == y).sum().item()
    acc = 100.0 * correct / float(y.numel())
    return acc


@torch.no_grad()
def evaluate_model_on_Z(model: nn.Module,
                        Z: torch.Tensor,
                        y: torch.Tensor) -> float:
    model.eval()
    ds = TensorDataset(Z, y)
    dl = DataLoader(ds, batch_size=BATCH_SIZE_STUDENT, shuffle=False)

    correct = 0
    total = 0
    for zb, yb in dl:
        zb = zb.to(DEVICE_TORCH, non_blocking=True)
        yb = yb.to(DEVICE_TORCH, non_blocking=True)
        logits = model(zb)
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())

    if total == 0:
        return 0.0
    return 100.0 * correct / float(total)


@torch.no_grad()
def evaluate_ensemble_on_Z(model1: nn.Module,
                           model2: nn.Module,
                           Z: torch.Tensor,
                           y: torch.Tensor) -> float:
    """
    Ensemble semplice: media dei logit di model1 e model2.
    """
    model1.eval()
    model2.eval()
    ds = TensorDataset(Z, y)
    dl = DataLoader(ds, batch_size=BATCH_SIZE_STUDENT, shuffle=False)

    correct = 0
    total = 0
    for zb, yb in dl:
        zb = zb.to(DEVICE_TORCH, non_blocking=True)
        yb = yb.to(DEVICE_TORCH, non_blocking=True)
        logits1 = model1(zb)
        logits2 = model2(zb)
        logits = (logits1 + logits2) / 2.0
        pred = logits.argmax(dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())

    if total == 0:
        return 0.0
    return 100.0 * correct / float(total)


# ========= TRAINING: DENSE (KD soft-label su U = train ∪ test) =========
def train_dense_student_on_U(Z_all: torch.Tensor,
                             teacher_logits_all: torch.Tensor,
                             C: int,
                             epochs: int = EPOCHS_DENSE) -> nn.Module:
    """
    Dense-style: uno studente MLP impara ad imitare l'ensemble dei client
    su tutto il pool U = z_train ∪ z_test, usando soft-label (KD).
    """
    device = DEVICE_TORCH

    # Pre-computiamo le soft-label con temperatura
    T = KD_TEMPERATURE
    with torch.no_grad():
        teacher_probs_all = F.softmax(teacher_logits_all / T, dim=1)  # [N,C] su CPU

    # Dataset: (z, idx) per recuperare le prob del teacher
    N = Z_all.size(0)
    idx_all = torch.arange(N, dtype=torch.long)
    ds = TensorDataset(Z_all, idx_all)
    dl = DataLoader(ds, batch_size=BATCH_SIZE_STUDENT, shuffle=True, drop_last=False)

    student = MLPStudent(K_RP, C).to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=LR_DENSE, weight_decay=WD_DENSE)
    kd_loss = nn.KLDivLoss(reduction="batchmean")

    for ep in range(1, epochs + 1):
        student.train()
        tot_loss = 0.0
        tot_count = 0

        for zb, idxb in dl:
            zb = zb.to(device, non_blocking=True)
            pb = teacher_probs_all[idxb].to(device, non_blocking=True)  # [B,C]

            logits = student(zb) / T
            log_p = F.log_softmax(logits, dim=1)
            loss = kd_loss(log_p, pb) * (T * T)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), MAX_NORM)
            opt.step()

            bs = zb.size(0)
            tot_loss += loss.item() * bs
            tot_count += bs

        avg_loss = tot_loss / max(1, tot_count)
        print(f"[DENSE] ep {ep:02d}/{epochs} | kd_loss={avg_loss:.4f}")

    # quick sanity: agreement con teacher su tutto U
    with torch.no_grad():
        student.eval()
        logits_s = []
        dl_eval = DataLoader(TensorDataset(Z_all), batch_size=BATCH_SIZE_STUDENT, shuffle=False)
        for (zb,) in dl_eval:
            zb = zb.to(device, non_blocking=True)
            logits_s.append(student(zb).cpu())
        logits_s = torch.cat(logits_s, dim=0)  # [N,C]
        pred_s = logits_s.argmax(dim=1)
        pred_t = teacher_logits_all.argmax(dim=1)
        agree = (pred_s == pred_t).float().mean().item() * 100.0
        print(f"[CHECK] DenseStudent vs Teacher argmax agreement su U = {agree:.2f}%")

    return student


# ========= TRAINING: CO-BOOSTING (due studenti, il secondo sugli errori) =========
def train_cobo_students_on_U(Z_all: torch.Tensor,
                             teacher_logits_all: torch.Tensor,
                             C: int,
                             epochs1: int = EPOCHS_COBOOST_1,
                             epochs2: int = EPOCHS_COBOOST_2) -> Tuple[nn.Module, nn.Module]:
    """
    Co-Boosting-style:
      - Student1: come Dense (KD su tutto U)
      - Student2: KD focalizzato sui punti in cui Student1 e Teacher non sono d'accordo
    """
    device = DEVICE_TORCH
    N = Z_all.size(0)
    T = KD_TEMPERATURE

    # Pre-computiamo teacher soft e hard
    with torch.no_grad():
        teacher_probs_all = F.softmax(teacher_logits_all / T, dim=1)  # [N,C] CPU
        teacher_hard = teacher_logits_all.argmax(dim=1)               # [N] CPU

    # --- Round 1: student1 su tutto U (Dense-style) ---
    print("\n[COBOOST] Round 1 (student1 = Dense-style su tutto U)")
    student1 = MLPStudent(K_RP, C).to(device)
    opt1 = torch.optim.AdamW(student1.parameters(), lr=LR_COBOOST, weight_decay=WD_COBOOST)
    kd_loss = nn.KLDivLoss(reduction="batchmean")

    ds_all = TensorDataset(Z_all, torch.arange(N, dtype=torch.long))
    dl_all = DataLoader(ds_all, batch_size=BATCH_SIZE_STUDENT, shuffle=True, drop_last=False)

    for ep in range(1, epochs1 + 1):
        student1.train()
        tot_loss = 0.0
        tot_count = 0

        for zb, idxb in dl_all:
            zb = zb.to(device, non_blocking=True)
            pb = teacher_probs_all[idxb].to(device, non_blocking=True)

            logits = student1(zb) / T
            log_p = F.log_softmax(logits, dim=1)
            loss = kd_loss(log_p, pb) * (T * T)

            opt1.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student1.parameters(), MAX_NORM)
            opt1.step()

            bs = zb.size(0)
            tot_loss += loss.item() * bs
            tot_count += bs

        avg_loss = tot_loss / max(1, tot_count)
        print(f"[COBOOST] S1 ep {ep:02d}/{epochs1} | kd_loss={avg_loss:.4f}")

    # --- Troviamo gli errori di student1 rispetto al teacher ---
    with torch.no_grad():
        student1.eval()
        logits_s1 = []
        dl_eval = DataLoader(TensorDataset(Z_all), batch_size=BATCH_SIZE_STUDENT, shuffle=False)
        for (zb,) in dl_eval:
            zb = zb.to(device, non_blocking=True)
            logits_s1.append(student1(zb).cpu())
        logits_s1 = torch.cat(logits_s1, dim=0)
        pred_s1 = logits_s1.argmax(dim=1)
        mismatch_mask = (pred_s1 != teacher_hard)  # [N] bool
        mismatch_idx = mismatch_mask.nonzero(as_tuple=False).view(-1)

    num_mismatch = mismatch_idx.numel()
    print(f"[COBOOST] #sample dove S1 != Teacher: {num_mismatch} / {N}")

    if num_mismatch == 0:
        print("[COBOOST] Nessun mismatch: S1 ~ Teacher ovunque, S2 identico a S1.")
        return student1, student1

    # --- Round 2: student2 solo sui mismatch ---
    print("\n[COBOOST] Round 2 (student2 focalizzato sugli errori di student1)")
    student2 = MLPStudent(K_RP, C).to(device)
    opt2 = torch.optim.AdamW(student2.parameters(), lr=LR_COBOOST, weight_decay=WD_COBOOST)

    Z_m = Z_all[mismatch_idx]                      # [Nm, K]
    probs_m = teacher_probs_all[mismatch_idx]      # [Nm, C]
    idx_m = torch.arange(Z_m.size(0), dtype=torch.long)
    ds_m = TensorDataset(Z_m, idx_m)
    dl_m = DataLoader(ds_m, batch_size=BATCH_SIZE_STUDENT, shuffle=True, drop_last=False)

    for ep in range(1, epochs2 + 1):
        student2.train()
        tot_loss = 0.0
        tot_count = 0

        for zb, idxb in dl_m:
            zb = zb.to(device, non_blocking=True)
            pb = probs_m[idxb].to(device, non_blocking=True)

            logits = student2(zb) / T
            log_p = F.log_softmax(logits, dim=1)
            loss = kd_loss(log_p, pb) * (T * T)

            opt2.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(student2.parameters(), MAX_NORM)
            opt2.step()

            bs = zb.size(0)
            tot_loss += loss.item() * bs
            tot_count += bs

        avg_loss = tot_loss / max(1, tot_count)
        print(f"[COBOOST] S2 ep {ep:02d}/{epochs2} | kd_loss={avg_loss:.4f}")

    # Quick check agreement
    with torch.no_grad():
        student2.eval()
        logits_s2 = []
        dl_eval2 = DataLoader(TensorDataset(Z_all), batch_size=BATCH_SIZE_STUDENT, shuffle=False)
        for (zb,) in dl_eval2:
            zb = zb.to(device, non_blocking=True)
            logits_s2.append(student2(zb).cpu())
        logits_s2 = torch.cat(logits_s2, dim=0)
        pred_s2 = logits_s2.argmax(dim=1)
        agree_s1 = (pred_s1 == teacher_hard).float().mean().item() * 100.0
        agree_s2 = (pred_s2 == teacher_hard).float().mean().item() * 100.0
        print(f"[CHECK] CoBoost S1 vs Teacher agreement su U = {agree_s1:.2f}%")
        print(f"[CHECK] CoBoost S2 vs Teacher agreement su U = {agree_s2:.2f}%")

    return student1, student2


# ========= MAIN PER DATASET + ALPHA =========
def run_one_dataset_alpha(ds_name: str,
                          alpha: float,
                          Z_train: torch.Tensor,
                          Z_test: torch.Tensor,
                          Y_train: torch.Tensor,
                          Y_test: torch.Tensor,
                          C: int) -> Dict[str, float]:
    print(f"\n=== DATASET: {ds_name} | alpha={alpha} (Dense / Co-Boost, KD su U=train∪test) ===")

    # Pool U = train ∪ test
    Z_all = torch.cat([Z_train, Z_test], dim=0)  # [N_all, K]
    print(f"[INFO] {ds_name} | alpha={alpha}: |U| = {Z_all.size(0)} (train+test)")

    # Client heads (teacher)
    client_heads = load_client_heads(ds_name, alpha, C_expected=C)

    # Ensemble diretto dei client su test
    acc_ens_clients = evaluate_client_ensemble_on_Z(client_heads, Z_test, Y_test)
    print(f"[INFO] {ds_name} | alpha={alpha} | Client ensemble acc (test) = {acc_ens_clients:.2f}%")

    # Teacher logits su U
    teacher_logits_all = compute_teacher_logits(client_heads, Z_all, C)  # [N_all, C] CPU

    # Info su distribuzione hard label del teacher su test
    with torch.no_grad():
        logits_test_teacher = teacher_logits_all[Z_train.size(0):]  # ultima parte = test
        y_teacher_test = logits_test_teacher.argmax(dim=1)
        uniq, cnt = torch.unique(y_teacher_test, return_counts=True)
        print("[INFO] Teacher pseudo-label distribuzione su test:")
        print(f"    #classi usate dal teacher (test): {len(uniq)} / {C}")

    # ----- Dense -----
    print("\n[RUN] Dense-style baseline (KD soft-label su U=train∪test)")
    dense_student = train_dense_student_on_U(Z_all, teacher_logits_all, C, epochs=EPOCHS_DENSE)
    acc_dense = evaluate_model_on_Z(dense_student, Z_test, Y_test)
    print(f"[RESULT] {ds_name} | alpha={alpha} | DenseStudent acc (test) = {acc_dense:.2f}%")

    # ----- Co-Boosting -----
    print("\n[RUN] Co-Boosting-style baseline (2 studenti, KD su U)")
    stu1, stu2 = train_cobo_students_on_U(
        Z_all, teacher_logits_all, C,
        epochs1=EPOCHS_COBOOST_1,
        epochs2=EPOCHS_COBOOST_2,
    )
    acc_cobo_s1 = evaluate_model_on_Z(stu1, Z_test, Y_test)
    acc_cobo_s2 = evaluate_model_on_Z(stu2, Z_test, Y_test)
    acc_cobo_ens = evaluate_ensemble_on_Z(stu1, stu2, Z_test, Y_test)

    print(f"[RESULT] {ds_name} | alpha={alpha} | CoBoost student1 acc (test) = {acc_cobo_s1:.2f}%")
    print(f"[RESULT] {ds_name} | alpha={alpha} | CoBoost student2 acc (test) = {acc_cobo_s2:.2f}%")
    print(f"[RESULT] {ds_name} | alpha={alpha} | CoBoost ensemble acc (test) = {acc_cobo_ens:.2f}%")

    return dict(
        acc_clients_ens=acc_ens_clients,
        acc_dense=acc_dense,
        acc_cobo_s1=acc_cobo_s1,
        acc_cobo_s2=acc_cobo_s2,
        acc_cobo_ens=acc_cobo_ens,
    )


def run_one_dataset(ds_name: str) -> Dict[float, Dict[str, float]]:
    """
    Calcola Z_train / Z_test una sola volta per dataset,
    poi cicla su tutte le ALPHAS.
    """
    # Carichiamo i dati esattamente come nel client
    print(f"\n==================== DATASET: {ds_name} ====================")
    texts_train, labels_train, texts_test, labels_test, class_names, C = load_splits(ds_name)
    print(f"[INFO] {ds_name}: {len(texts_train)} train, {len(texts_test)} test, C={C}")

    # Encoder + RP (stessa configurazione del client)
    tok, enc = build_encoder(MODEL_NAME)
    d_hidden = enc.config.hidden_size
    R = build_rp_matrix(d_hidden, K_RP, seed=RP_SEED)

    # z_train e z_test reali
    Z_train = compute_z_matrix(texts_train, tok, enc, R, desc=f"{ds_name}-train")
    Z_test = compute_z_matrix(texts_test, tok, enc, R, desc=f"{ds_name}-test")
    Y_train = torch.tensor(labels_train, dtype=torch.long)
    Y_test = torch.tensor(labels_test, dtype=torch.long)

    print(f"[INFO] {ds_name}: |train|={Z_train.size(0)}, |test|={Z_test.size(0)}, C={C}")

    per_alpha: Dict[float, Dict[str, float]] = {}
    for alpha in ALPHAS:
        res = run_one_dataset_alpha(
            ds_name=ds_name,
            alpha=alpha,
            Z_train=Z_train,
            Z_test=Z_test,
            Y_train=Y_train,
            Y_test=Y_test,
            C=C,
        )
        per_alpha[alpha] = res

    return per_alpha


def main():
    set_seed(SEED)
    print(f"[INFO] Device: {DEVICE}")

    # summary[dataset][alpha] = dict(...)
    summary: Dict[str, Dict[float, Dict[str, float]]] = {}

    for ds in DATASETS:
        summary[ds] = run_one_dataset(ds)

    # ====== TABELLA FINALE ======
    print("\n===================== SUMMARY (Dense / Co-Boost baselines, KD su U=train∪test) =====================")
    header = (
        f"{'Dataset':11s} | {'alpha':6s} | "
        f"{'ClientEns':9s} | {'Dense':7s} | "
        f"{'CoBoost S1':10s} | {'CoBoost S2':10s} | {'CoBoost Ens':11s}"
    )
    print(header)
    print("-" * len(header))

    for ds in DATASETS:
        if ds not in summary:
            continue
        for alpha in ALPHAS:
            if alpha not in summary[ds]:
                continue
            r = summary[ds][alpha]
            print(
                f"{ds:11s} | {alpha:6.2f} | "
                f"{r['acc_clients_ens']:9.2f}% | "
                f"{r['acc_dense']:7.2f}% | "
                f"{r['acc_cobo_s1']:10.2f}% | "
                f"{r['acc_cobo_s2']:10.2f}% | "
                f"{r['acc_cobo_ens']:11.2f}%"
            )

    print("================================================================================================")


if __name__ == "__main__":
    main()




