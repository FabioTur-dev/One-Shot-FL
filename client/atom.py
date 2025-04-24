import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    """
    Untrained backbone for multilayer feature extraction (ATOM protocol).
    """
    def __init__(self, base_model: nn.Module, layers: list):
        super().__init__()
        self.base = base_model
        self.layers = layers
        # Hook storage
        self.features = {}
        for name, module in self.base.named_modules():
            if name in layers:
                module.register_forward_hook(self._hook(name))

    def _hook(self, name):
        def fn(_, __, output):
            self.features[name] = output
        return fn

    def forward(self, x: torch.Tensor) -> dict:
        _ = self.base(x)
        return {name: self.features[name] for name in self.layers}