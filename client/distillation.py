# File: client/distillation.py

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from .atom import FeatureExtractor

class ATOMDistiller:
    """
    Implements ATOM: Attention Mixer for Efficient Dataset Distillation,
    correggendo mismatch dimensioni e aggregazione per layer.
    """
    def __init__(self,
                 backbone: torch.nn.Module,
                 layers: list,
                 synthetic_images: torch.Tensor,
                 synthetic_labels: torch.Tensor,
                 device: torch.device,
                 lr: float = 0.1,
                 momentum: float = 0.9):
        # Estrattore non addestrato con hook multilayer
        self.extractor = FeatureExtractor(backbone, layers).to(device).eval()
        self.layers    = layers
        self.device    = device

        # Immagini sintetiche come parametro ottimizzabile
        self.S = synthetic_images.to(device).requires_grad_(True)
        self.y = synthetic_labels.to(device)

        # Ottimizzatore sui pixel sintetici
        self.optimizer = optim.SGD([self.S], lr=lr, momentum=momentum)

    def compute_attentions(self, feats: torch.Tensor):
        """
        Calcola spatial attention (As) e channel attention (Ac)
        su feats di shape [B, D].
        """
        As = F.softmax(feats, dim=1)                            # [B, D]
        Ac = F.softmax(feats.mean(dim=0, keepdim=True), dim=1)  # [1, D]
        return As, Ac

    def distill(self, loader: DataLoader, iterations: int = 1000):
        """
        Per ogni iterazione:
          1. Preleva un batch reale x_r
          2. Estrae feature multilayer reali e sintetiche
          3. Per ogni layer: appiattisce, calcola attenzioni, aggrega con mean
          4. Calcola loss MSE layer-wise e somma
          5. Aggiorna i pixel sintetici
        """
        data_iter = iter(loader)
        for _ in range(iterations):
            try:
                x_r, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x_r, _ = next(data_iter)

            x_r = x_r.to(self.device)
            feats_r   = self.extractor(x_r)    # dict {layer: [B, C, H, W]}
            feats_syn = self.extractor(self.S)  # dict {layer: [N, C, H, W]}

            total_loss = 0.0
            for l in self.layers:
                fr_l = feats_r[l].view(feats_r[l].size(0), -1)    # [B, D]
                fs_l = feats_syn[l].view(feats_syn[l].size(0), -1)  # [N, D]

                As_r, Ac_r = self.compute_attentions(fr_l)
                As_s, Ac_s = self.compute_attentions(fs_l)

                fr_w = (As_r * Ac_r * fr_l).mean(dim=0, keepdim=True)   # [1, D]
                fs_w = (As_s * Ac_s * fs_l).mean(dim=0, keepdim=True)   # [1, D]

                total_loss += F.mse_loss(fr_w, fs_w)

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

        return self.S.detach(), self.y


