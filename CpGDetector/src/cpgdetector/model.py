from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel // 2)
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.residual(x)


class MultiTaskCpGNet(nn.Module):
    """Base-level CpG island segmentation with a window-level auxiliary head."""

    def __init__(
        self,
        channels: list[int] | tuple[int, ...] = (64, 128, 256),
        kernels: list[int] | tuple[int, ...] = (7, 5, 3),
        dilations: list[int] | tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.10,
    ):
        super().__init__()
        if not (len(channels) == len(kernels) == len(dilations)):
            raise ValueError("channels, kernels, and dilations must have the same length")
        blocks = []
        in_channels = 4
        for out_channels, kernel, dilation in zip(channels, kernels, dilations):
            blocks.append(ConvBlock(in_channels, int(out_channels), int(kernel), int(dilation), float(dropout)))
            in_channels = int(out_channels)
        self.encoder = nn.Sequential(*blocks)
        self.base_head = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.window_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        base_logits = self.base_head(features).squeeze(1)
        window_logits = self.window_head(features)
        return {
            "base_logits": base_logits,
            "window_logits": window_logits,
        }
