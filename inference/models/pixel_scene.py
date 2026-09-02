from __future__ import annotations

import torch
from torch import nn


class SatQueryPixelModel(nn.Module):
    """Small segmentation/classification model designed for a 6 GB GPU."""

    def __init__(self, in_channels: int = 6, classes: int = 3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 24, 3, padding=1), nn.GroupNorm(6, 24), nn.GELU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.GroupNorm(6, 48), nn.GELU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 48, 2, stride=2), nn.GELU(),
            nn.ConvTranspose2d(48, 32, 2, stride=2), nn.GELU(),
            nn.Conv2d(32, classes, 1),
        )
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, classes))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        return self.decoder(features), self.classifier(features)

