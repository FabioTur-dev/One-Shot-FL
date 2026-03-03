#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
=====================================================================================
GH-OFL | CLIENT | X-SPACE | FLEXIBLE CIFAR100C STATS (ICLR CAMERA READY)
=====================================================================================

Supports:

1) legacy_holdout
2) single_corruption
3) cross_corruption  ✅ TRUE FEDERATED CORRUPTION SPLIT
=====================================================================================
"""

from __future__ import annotations
import os
import random
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models

# ------------------------------------------------
# CONSTANTS
# ------------------------------------------------
FEATURE_DIM = 512

IMAGENET_MEAN = (0.485,0.456,0.406)
IMAGENET_STD  = (0.229,0.224,0.225)

ALL_CORRUPTIONS = [
    "brightness","contrast","defocus_blur","elastic_transform",
    "fog","frost","gaussian_blur","gaussian_noise",
    "glass_blur","impulse_noise","jpeg_compression",
    "motion_blur","pixelate","saturate",
    "shot_noise","snow","spatter",
    "speckle_noise","zoom_blur"
]

# ------------------------------------------------
# UTILS
# ------------------------------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_yaml(path):
    with open(path,"r") as f:
        return yaml.safe_load(f)

# ------------------------------------------------
# BACKBONE
# ------------------------------------------------
def build_backbone(device):

    from torchvision.models import ResNet18_Weights

    model = models.resnet18(
        weights=ResNet18_Weights.IMAGENET1K_V1
    )

    model.fc = nn.Identity()
    model.eval().to(device)

    for p in model.parameters():
        p.requires_grad_(False)

    return model,"IMAGENET1K_V1"


def build_transform():
    return transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN,IMAGENET_STD),
    ])

# ------------------------------------------------
# CIFAR100C DATASET
# ------------------------------------------------
class CIFAR100CSubset(torch.utils.data.Dataset):

    def __init__(self,root,corruptions,severities,transform):

        self.root=root
        self.transform=transform

        self.labels_all=np.load(os.path.join(root,"labels.npy"))

        self.items=[]
        self.cache={}

        for corr in corruptions:
            arr=np.load(
                os.path.join(root,f"{corr}.npy"),
                mmap_mode="r"
            )

            for s in severities:
                lo=(s-1)*10000
                hi=s*10000
                for p in range(lo,hi):
                    self.items.append((corr,p))

    def __len__(self):
        return len(self.items)

    def _get(self,c):
        if c not in self.cache:
            self.cache[c]=np.load(
                os.path.join(self.root,f"{c}.npy"),
                mmap_mode="r"
            )
        return self.cache[c]

    def __getitem__(self,idx):

        from PIL import Image

        corr,p=self.items[idx]
        img=self._get(corr)[p]

        img=Image.fromarray(np.array(img))

        if self.transform:
            img=self.transform(img)

        y=int(self.labels_all[p])
        return img,y


# ------------------------------------------------
# DIRICHLET SPLIT
# ------------------------------------------------
def dirichlet_split(labels,C,K,alpha,seed):

    set_seed(seed)

    labels=torch.tensor(labels)
    clients=[[] for _ in range(K)]

    for c in range(C):

        idx=torch.where(labels==c)[0]
        idx=idx[torch.randperm(len(idx))]

        dist=torch.distributions.Dirichlet(
            torch.full((K,),alpha)
        ).sample()

        counts=(dist*len(idx)).long()

        s=0
        for k in range(K):
            e=s+counts[k]
            clients[k]+=idx[s:e].tolist()
            s=e

    return clients


# ------------------------------------------------
# STATS
# ------------------------------------------------
@torch.no_grad()
def compute_stats(model,loader,C,device):

    d=FEATURE_DIM

    A=torch.zeros(C,d,dtype=torch.float64,device=device)
    N=torch.zeros(C,dtype=torch.long,device=device)
    B=torch.zeros(d,d,dtype=torch.float64,device=device)
    S=torch.zeros(C,d,d,dtype=torch.float64,device=device)

    for x,y in loader:

        x=x.to(device)
        y=y.to(device)

        z=model(x).double()

        B+=z.t()@z

        for c in y.unique():
            m=(y==c)
            xc=z[m]

            ci=int(c)
            A[ci]+=xc.sum(0)
            N[ci]+=xc.size(0)
            S[ci]+=xc.t()@xc

    return dict(
        A_per_class=A.cpu(),
        N_per_class=N.cpu(),
        B_global=B.cpu(),
        S_per_class=S.cpu()
    )


# ------------------------------------------------
# MAIN
# ------------------------------------------------
def main():

    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    args=parser.parse_args()

    cfg=load_yaml(args.config)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(cfg["seed"])

    model,tag=build_backbone(device)
    tfm=build_transform()

    mode=cfg["cifar100c_mode"]
    severities=tuple(cfg["severities_train"])

    print("\n========== CIFAR100C MODE ==========")
    print("mode:",mode)

    client_datasets=[]

    # ============================================================
    # CROSS CORRUPTION (FIXED VERSION)
    # ============================================================
    if mode=="cross_corruption":

        pool=cfg["corruptions_train"]

        client_corruptions=[
            [pool[i % len(pool)]]
            for i in range(cfg["num_clients"])
        ]

        print("\nClient corruption assignment:")
        for i,c in enumerate(client_corruptions):
            print(f" client {i:02d} -> {c[0]}")

        for corr in client_corruptions:

            ds_client=CIFAR100CSubset(
                "./data/CIFAR-100-C",
                corr,
                severities,
                tfm
            )

            labels=[
                ds_client.labels_all[p]
                for (_,p) in ds_client.items
            ]

            split=dirichlet_split(
                labels,
                100,
                1,
                cfg["dirichlet_alpha"],
                cfg["seed"]
            )[0]

            client_datasets.append((ds_client,split))

    # ============================================================
    # SINGLE / LEGACY
    # ============================================================
    else:

        if mode=="single_corruption":
            corruptions=[cfg["single_corruption_name"]]
        else:
            corruptions=ALL_CORRUPTIONS

        ds=CIFAR100CSubset(
            "./data/CIFAR-100-C",
            corruptions,
            severities,
            tfm
        )

        labels=[ds.labels_all[p] for (_,p) in ds.items]

        splits=dirichlet_split(
            labels,
            100,
            cfg["num_clients"],
            cfg["dirichlet_alpha"],
            cfg["seed"]
        )

        for cid in range(cfg["num_clients"]):
            client_datasets.append((ds,splits[cid]))

    # ------------------------------------------------
    # OUTPUT
    # ------------------------------------------------
    out_dir=f"./client_stats/CIFAR100C_{mode.upper()}/resnet18-{tag}"
    os.makedirs(out_dir,exist_ok=True)

    for cid,(ds_client,idx) in enumerate(client_datasets):

        loader=DataLoader(
            Subset(ds_client,idx),
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg["num_workers"]
        )

        stats=compute_stats(model,loader,100,device)

        payload=dict(
            client_id=cid,
            meta=dict(
                mode=mode,
                backbone="resnet18",
                weights_tag=tag,
                feature_dim=FEATURE_DIM
            ),
            **stats
        )

        torch.save(
            payload,
            os.path.join(out_dir,f"client_{cid:02d}.pt")
        )

        print(f"[OK] client {cid}")

    print("\nDONE.")


if __name__=="__main__":
    main()