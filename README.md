<p align="center">
  <img src="assets/GH_OFL_logo.png" width="300">
</p>

# GH-OFL — The Gaussian Head Family  
### One-Shot Federated Learning from Client Global Statistics  
**ICLR 2026 — Official Camera-Ready Implementation**

---

## 📌 Overview

**GH-OFL (Gaussian-Head One-Shot Federated Learning)** is a statistics-driven federated learning framework where clients transmit global feature statistics only, enabling server-side closed-form and lightweight trainable classifiers — without gradient exchange, multi-round optimization, or raw data transmission.

This repository contains the refined camera-ready implementation associated with:

Turazza, Picone, Mamei.  
*The Gaussian-Head OFL Family: One-Shot Federated Learning from Client Global Statistics.*  
ICLR 2026.

<p align="center">📄 <a href="https://arxiv.org/abs/2602.01186"><b>Read the paper on arXiv (2602.01186)</b></a></p>

---

# 🧠 Core Idea

Instead of sharing gradients or local model weights, each client computes and transmits:

- Per-class feature sums  
- Diagonal second moments  
- Full covariance matrices (optional, enables QDA)  
- Global second-order statistics  

The server reconstructs Gaussian decision heads analytically and optionally refines them in a Fisher subspace.

✔ One-shot communication  
✔ Statistics-only federation  
✔ No gradient aggregation  
✔ No raw data exchange  
✔ No iterative client training  

---

# 📊 Implemented Heads

## Closed-Form (x-space)

- **GH-NBdiag** — Diagonal Gaussian classifier  
- **GH-LDA** — Pooled covariance (shrinkage = 0.05)  
- **GH-QDAfull** — Full class covariance (GPU optimized)  

## Trainable (Fisher space)

- **FisherMix** — Cosine classifier on Fisher projections  
- **Proto-Hyper** — Low-rank residual adapter with knowledge distillation  

Proto-Hyper formulation:

Student(z_f) = Standardize(LDA_f(z_f)) + LowRankResidual(z_f)  
Teacher = λ · QDA_f + (1 − λ) · LDA_f  
Loss = Knowledge Distillation (KL) + Cross-Entropy  

Inference uses the student model only.

---

# 📁 Repository Structure

GH-OFL/  
│  
├── client_cifar10.py  
├── server_cifar10.py  
├── client_cifar100.py  
├── server_cifar100.py  
├── client_svhn.py  
├── server_svhn.py  
│  
├── data/  
├── client_stats_X/  
└── README.md  

All scripts follow a unified taxonomy:

GH-OFL | DATASET | ROLE | SPACE  

Example:  
GH-OFL | CIFAR-100 | SERVER | x-space  

---

# ⚙️ Requirements

- Python ≥ 3.9  
- PyTorch ≥ 2.0  
- Torchvision ≥ 0.15  
- CUDA optional (recommended for QDA)

Install dependencies:

pip install torch torchvision numpy  

---

# 🚀 How to Run the Code (Step-by-Step)

## STEP 1 — Generate Client Statistics

Run the client script for the desired dataset.

Example (CIFAR-100):

python client_cifar100.py  

What happens:

1. Dataset is downloaded automatically (if not present).
2. Dirichlet split is generated (α defined inside the script).
3. ResNet-18 extracts 512-dimensional features.
4. Each client accumulates statistics in float64.
5. Client payloads are saved to:

./client_stats_X/CIFAR100/resnet18-IMAGENET1K_V1_TRAIN_A{alpha}_X512/

Repeat for other datasets:

python client_cifar10.py  
python client_svhn.py  

You only need to generate statistics once per α configuration.

---

## STEP 2 — Run Server Evaluation

Open the corresponding server script and verify:

STATS_ROOT = "./client_stats_X/..."

Make sure it matches the directory generated in Step 1.

Then run:

python server_cifar100.py  

What the server does:

1. Loads all client .pt files.
2. Aggregates global statistics.
3. Computes:
   - GH-NBdiag
   - GH-LDA
   - GH-QDAfull (if S_per_class_x is available)
4. Builds the Fisher subspace.
5. Synthesizes Fisher-space samples.
6. Trains FisherMix and Proto-Hyper.
7. Evaluates on the test set.

---

## Switching Dirichlet α

Inside the server script:

STATS_ROOT = "./client_stats_X/CIFAR100/...A0p1_X512"

Change to:

...A0p5_X512

No other modifications required.

---

# 🖥 GPU vs CPU Behavior

- Closed-form NB and LDA run on CPU (float64 for stability).
- QDAfull runs on GPU (float32, Cholesky-based, chunked).
- FisherMix and Proto-Hyper train on GPU if available.
- Full CPU execution is supported (slower but correct).

---

# 🔄 Code Update Notice (Important)

This repository corresponds to the final refined camera-ready implementation.

The code has been:

- Refactored for strict numerical stability (explicit float64 accumulation)
- Unified across CIFAR-10, CIFAR-100, and SVHN
- GPU-optimized for full QDA
- Deterministically seeded
- Cleaned for artifact reproducibility
- Explicitly symmetrized covariance matrices
- Shrinkage handling made consistent across heads

---

## ⚠ Result Differences vs Paper

Due to:

- Improved shrinkage stabilization
- Explicit covariance symmetrization
- Deterministic seed control
- Fisher-space numerical conditioning
- Minor hyperparameter normalization refinements
- Hardware-dependent floating point behavior

Results produced by this repository may differ slightly from those reported in the paper tables.

Differences are typically small and stem from stability and reproducibility improvements.

The implementation remains fully consistent with the theoretical formulation described in the paper.

---

# 🔒 Design Principles

- Statistics-only federation  
- One-shot communication  
- Closed-form analytical heads  
- Controlled Fisher refinement  
- Explicit dtype/device separation  
- Deterministic and reproducible  
- Artifact-review ready  

---

# 📖 Citation

If you use this code, please cite:

@inproceedings{turazza2026ghofl,  
  title={The Gaussian-Head OFL Family: One-Shot Federated Learning from Client Global Statistics},  
  author={Turazza, Fabio and Picone, Marco and Mamei, Marco},  
  booktitle={International Conference on Learning Representations (ICLR)},  
  year={2026}  
}

---

# 🏛 Affiliations

Department of Sciences and Methods for Engineering (DISMI)  
Artificial Intelligence Research and Innovation Center (AIRI)  
University of Modena and Reggio Emilia, Italy  

---

## GH-OFL  
### Rethinking Federation Beyond Gradient Aggregation
