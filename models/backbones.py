# models/backbones.py

from typing import Tuple
import torch
import torch.nn as nn
from torchvision import models


def build_backbone(name: str, device: torch.device) -> Tuple[nn.Module, int, str]:
    """
    Build ImageNet-pretrained backbone as pure feature extractor.

    Returns:
        model       : nn.Module (fc removed)
        feature_dim : int
        weights_tag : str
    """

    name = name.lower()
    weights_tag = "IMAGENET1K_V1"

    if name == "resnet18":
        from torchvision.models import ResNet18_Weights
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        feature_dim = 512
        model.fc = nn.Identity()

    elif name == "resnet50":
        from torchvision.models import ResNet50_Weights
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        feature_dim = 2048
        model.fc = nn.Identity()

    elif name == "mobilenet_v2":
        from torchvision.models import MobileNet_V2_Weights
        model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        feature_dim = 1280
        model.classifier = nn.Identity()

    elif name == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        feature_dim = 1280
        model.classifier = nn.Identity()


    elif name == "vgg16":
        from torchvision.models import VGG16_Weights
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # Keep classifier up to penultimate layer (4096-dim)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
        feature_dim = 4096

    else:
        raise ValueError(f"Unsupported backbone: {name}")

    model.eval().to(device)

    for p in model.parameters():
        p.requires_grad_(False)

    return model, feature_dim, weights_tag