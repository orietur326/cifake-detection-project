"""Model definitions for CIFAKE Part 2 deep learning baselines."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


def build_resnet18_cifake(num_classes: int = 2) -> nn.Module:
    """Build a ResNet-18 adapted for CIFAKE 32x32 RGB images.

    Modifications from the default ImageNet-style ResNet-18:
    - conv1 uses 3x3 kernel, stride 1, padding 1
    - maxpool is replaced by Identity
    - final fc outputs ``num_classes`` logits
    """
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
