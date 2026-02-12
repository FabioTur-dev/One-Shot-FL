#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OFL Baselines - Version C-2 (FULL PAPER VERSION)
================================================
- FULL MODEL TRAINING (client-side)
- FULL MODEL DENSE-STUDENT DISTILLATION (server-side)
- FULL CO-BOOST (server-side)
- Augmentations forti come FedDF / Co-Boost
- Backbone = ResNet18 (random init → fair one-shot)

Output:
  oneshot_full/{DS}/{ALPHA}/clients/client_x_model.pt
  oneshot_full/{DS}/{ALPHA}/server/
"""

import os, random, numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms, models

# ============================================================
# CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

DATASETS = ["CIFAR10", "CIFAR100", "SVHN"]
ALPHAS = [0.05, 0.10, 0.50]
NUM_CLIENTS = 10

LOCAL_EPOCHS = 5      # FULL MODEL TRAIN PER CLIENT
BATCH = 128
LR_CLIENT = 1e-3
WD_CLIENT = 1e-4

# Server distillation student
LR_STUDENT = 5e-4
EPOCHS_DENSE = 400

# Co-boost
EPOCHS_CO_S1 = 200
EPOCHS_CO_S2 = 150

OUTDIR = "./oneshot_full"
os.makedirs(OUTDIR, exist_ok=True)

def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# ============================================================
# Dirichlet split
# ============================================================
def dirichlet_split(labels, C, K, alpha):
    labels = np.array(labels)
    splits = [[] for _ in range(K)]
    for c in range(C):
        idx = np.where(labels == c)[0]
        if len(idx)==0: continue
        np.random.shuffle(idx)
        dist = np.random.dirichlet([alpha]*K)
        counts = (dist * len(idx)).round().astype(int)
        diff = len(idx) - counts.sum()
        for i in range(abs(diff)):
            counts[i % K] += 1 if diff>0 else -1
        s=0
        for k in range(K):
            e=s+counts[k]
            if e>s: splits[k].extend(idx[s:e].tolist())
            s=e
    for k in range(K): random.shuffle(splits[k])
    return splits

# ============================================================
# Backbone = ResNet18 random init
# ============================================================
def build_model(C):
    net = models.resnet18(weights=None)
    net.fc = nn.Linear(512, C)
    return net.to(DEVICE)

# Strong augmentation per FedDF
def build_transform_train():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3,0.3,0.3,0.1),
        transforms.ToTensor(),
    ])

def build_transform_test():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])

class SVHNWrap(Dataset):
    def __init__(self, ds, T): self.ds,self.T=ds,T
    def __len__(self): return len(self.ds)
    def __getitem__(self,i):
        x=self.ds.data[i].transpose(1,2,0)
        y=int(self.ds.labels[i])
        return self.T(Image.fromarray(x)), y

def load_dataset(name):
    Ttrain=build_transform_train()
    Ttest =build_transform_test()

    if name=="CIFAR10":
        tr=datasets.CIFAR10("./data",train=True,download=True)
        te=datasets.CIFAR10("./data",train=False,download=True)
        C=10
        class Wrap(Dataset):
            def __init__(self,ds,T):self.ds, self.T=ds,T
            def __getitem__(self,i): x,y=self.ds[i]; return self.T(x),y
            def __len__(self):return len(self.ds)
        return Wrap(tr,Ttrain), Wrap(te,Ttest), C

    if name=="CIFAR100":
        tr=datasets.CIFAR100("./data",train=True,download=True)
        te=datasets.CIFAR100("./data",train=False,download=True)
        C=100
        class Wrap(Dataset):
            def __init__(self,ds,T):self.ds,self.T=ds,T
            def __getitem__(self,i): x,y=self.ds[i]; return self.T(x),y
            def __len__(self):return len(self.ds)
        return Wrap(tr,Ttrain),Wrap(te,Ttest),C

    if name=="SVHN":
        tr=datasets.SVHN("./data",split="train",download=True)
        te=datasets.SVHN("./data",split="test",download=True)
        return SVHNWrap(tr,Ttrain), SVHNWrap(te,Ttest), 10

# ============================================================
# CLIENT TRAINING (FULL MODEL)
# ============================================================
def train_full_model(model, dl):
    opt = torch.optim.AdamW(model.parameters(),lr=LR_CLIENT,weight_decay=WD_CLIENT)
    ce = nn.CrossEntropyLoss()
    model.train()
    for ep in range(LOCAL_EPOCHS):
        for x,y in dl:
            x,y = x.to(DEVICE),y.to(DEVICE)
            logits=model(x)
            loss=ce(logits,y)
            opt.zero_grad(); loss.backward(); opt.step()

# ============================================================
# SERVER: ENSEMBLE
# ============================================================
@torch.no_grad()
def server_ensemble(models, dl_test):
    for m in models: m.eval()
    probs=[]
    for x,y in dl_test:
        x=x.to(DEVICE)
        p = torch.stack([F.softmax(m(x),dim=1) for m in models]).mean(0)
        probs.append(p.cpu())
    P = torch.cat(probs)
    y_true=torch.cat([y for _,y in dl_test])
    acc=(P.argmax(1)==y_true).float().mean().item()*100
    return acc

# ============================================================
# SERVER: DENSE STUDENT (FedDF-like)
# ============================================================
def train_dense_student(models, dl_test, C):
    student = build_model(C)
    opt = torch.optim.Adam(student.parameters(), lr=LR_STUDENT)
    student.train()

    teachers = models
    for m in teachers: m.eval()

    for step in range(EPOCHS_DENSE):
        for x,y in dl_test:
            x=x.to(DEVICE)
            with torch.no_grad():
                T_logits = torch.stack([m(x) for m in teachers]).mean(0)
            S_logits = student(x)
            loss = F.kl_div(
                F.log_softmax(S_logits/1.0,dim=1),
                F.softmax(T_logits/1.0,dim=1),
                reduction="batchmean"
            )
            opt.zero_grad(); loss.backward(); opt.step()

    # Eval
    student.eval()
    acc=0; tot=0
    for x,y in dl_test:
        x,y=x.to(DEVICE),y.to(DEVICE)
        pred=student(x).argmax(1)
        acc += (pred.cpu()==y.cpu()).sum().item()
        tot += y.size(0)
    return 100*acc/tot

# ============================================================
# SERVER: CO-BOOST (full-model)
# ============================================================
def train_coboost(models, dl_test, C):

    # Teacher ensemble logits
    @torch.no_grad()
    def ensemble_logits(x):
        return torch.stack([m(x) for m in models]).mean(0)

    # -------------------------
    # S1 (first student)
    # -------------------------
    S1 = build_model(C)
    opt1 = torch.optim.Adam(S1.parameters(), lr=LR_STUDENT)
    S1.train()
    for step in range(EPOCHS_CO_S1):
        for x,_ in dl_test:
            x=x.to(DEVICE)
            with torch.no_grad():
                T = ensemble_logits(x)
            S = S1(x)
            loss=F.kl_div(F.log_softmax(S,dim=1), F.softmax(T,dim=1), reduction="batchmean")
            opt1.zero_grad(); loss.backward(); opt1.step()

    # hardness weighting
    S1.eval()
    all_x=[]; all_T=[]; all_S1=[]
    for x,y in dl_test:
        all_x.append(x)
        with torch.no_grad():
            all_T.append(ensemble_logits(x.to(DEVICE)).cpu())
            all_S1.append(S1(x.to(DEVICE)).cpu())
    X = torch.cat(all_x)
    Tlog = torch.cat(all_T)
    S1log= torch.cat(all_S1)
    conf = F.softmax(S1log,dim=1).max(1).values
    hard = (1.0-conf)
    weights = (hard/hard.mean()).clamp(0.3,4.0)

    # -------------------------
    # S2 (boosted student)
    # -------------------------
    S2 = build_model(C)
    S2.train()
    opt2 = torch.optim.Adam(S2.parameters(), lr=LR_STUDENT)
    N = X.size(0)

    for step in range(EPOCHS_CO_S2):
        idx=torch.randint(0,N,(BATCH,))
        xb = X[idx].to(DEVICE)
        Tb = Tlog[idx].to(DEVICE)
        wb = weights[idx].to(DEVICE)
        S2b = S2(xb)
        loss = (F.kl_div(F.log_softmax(S2b,dim=1),F.softmax(Tb,dim=1), reduction="none").sum(1)*wb).mean()
        opt2.zero_grad(); loss.backward(); opt2.step()

    # Evaluate S1, S2, ENS
    def eval_model(M):
        M.eval()
        correct=0; tot=0
        for x,y in dl_test:
            x,y=x.to(DEVICE),y.to(DEVICE)
            pred=M(x).argmax(1)
            correct+=(pred.cpu()==y.cpu()).sum().item()
            tot+=y.size(0)
        return 100*correct/tot

    S1_acc=eval_model(S1)
    S2_acc=eval_model(S2)

    # ensemble S1+S2
    @torch.no_grad()
    def eval_ens():
        correct=0; tot=0
        for x,y in dl_test:
            x,y=x.to(DEVICE),y.to(DEVICE)
            p=(F.softmax(S1(x),dim=1)+F.softmax(S2(x),dim=1))/2
            pred=p.argmax(1)
            correct+=(pred.cpu()==y.cpu()).sum().item()
            tot+=y.size(0)
        return 100*correct/tot

    return S1_acc, S2_acc, eval_ens()

# ============================================================
# MAIN
# ============================================================
def main():
    set_seed(SEED)

    for dset in DATASETS:
        print(f"\n======================\n### DATASET: {dset}\n======================")

        # load DS
        train_ds, test_ds, C = load_dataset(dset)
        full_train_y = []
        for _,y in DataLoader(train_ds,batch_size=512):
            full_train_y.append(y)
        y_all = torch.cat(full_train_y).numpy()

        # For each α
        for alpha in ALPHAS:
            print(f"\n--- α = {alpha} ---")

            # split
            splits = dirichlet_split(labels=y_all, C=C, K=NUM_CLIENTS, alpha=alpha)
            client_models=[]
            client_Zy=[]

            # CLIENT TRAINING
            for cid, idx in enumerate(splits):
                if len(idx)==0:
                    print(f"[CLIENT {cid:02d}] empty shard → skip")
                    continue

                print(f"[CLIENT {cid:02d}] train full ResNet on {len(idx)} samples")

                sub = Subset(train_ds, idx)
                dl = DataLoader(sub, batch_size=BATCH, shuffle=True)

                model = build_model(C)
                train_full_model(model, dl)

                path = f"{OUTDIR}/{dset}/{alpha}/clients"
                os.makedirs(path, exist_ok=True)
                torch.save(model.state_dict(), f"{path}/client_{cid}_model.pt")
                client_models.append(model)

            # SERVER PHASE
            dl_test = DataLoader(test_ds, batch_size=BATCH, shuffle=False)

            print("[ENSEMBLE] evaluating...")
            acc_ens = server_ensemble(client_models, dl_test)
            print(f"[ENSEMBLE] acc = {acc_ens:.2f}%")

            print("[DENSE-STUDENT] training...")
            acc_dense = train_dense_student(client_models, dl_test, C)
            print(f"[DENSE] acc = {acc_dense:.2f}%")

            print("[CO-BOOST] training...")
            s1,s2,ens = train_coboost(client_models, dl_test, C)
            print(f"[COBOOST] S1={s1:.2f}% | S2={s2:.2f}% | ENS={ens:.2f}%")


if __name__ == "__main__":
    main()
