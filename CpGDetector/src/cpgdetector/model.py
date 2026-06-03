from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        dilation = max(1, int(dilation))
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


class TaskAdapter(nn.Module):
    """Light task-specific projection before a task head."""

    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionWindowHead(nn.Module):
    """Window-level head with per-base linear projection and learned pooling."""

    def __init__(self, in_channels: int, hidden_channels: int | None, dropout: float):
        super().__init__()
        hidden_channels = int(hidden_channels or in_channels)
        self.projection = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention_logits = nn.Conv1d(hidden_channels, 1, kernel_size=1)
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_channels),
            nn.Linear(hidden_channels, max(1, hidden_channels // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, hidden_channels // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        weights = torch.softmax(self.attention_logits(projected), dim=-1)
        pooled = torch.sum(projected * weights, dim=-1)
        return self.output(pooled)


class MultiTaskCpGNet(nn.Module):
    """Base-level CpG island segmentation with a window-level auxiliary head."""

    def __init__(
        self,
        channels: list[int] | tuple[int, ...] = (64, 128, 256),
        kernels: list[int] | tuple[int, ...] = (7, 5, 3),
        dilations: list[int] | tuple[int, ...] = (1, 2, 4),
        dropout: float = 0.10,
        window_hidden_channels: int | None = None,
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
        self.base_adapter = TaskAdapter(in_channels, dropout)
        self.window_adapter = TaskAdapter(in_channels, dropout)
        self.base_head = nn.Conv1d(in_channels, 1, kernel_size=1)
        self.window_head = AttentionWindowHead(in_channels, window_hidden_channels, dropout)
        self.loss_log_vars = nn.ParameterDict(
            {
                "base": nn.Parameter(torch.zeros(())),
                "window": nn.Parameter(torch.zeros(())),
            }
        )
        self.gradnorm_log_weights = nn.ParameterDict(
            {
                "base": nn.Parameter(torch.zeros(())),
                "window": nn.Parameter(torch.zeros(())),
            }
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encoder(x)
        base_features = self.base_adapter(features)
        window_features = self.window_adapter(features)
        base_logits = self.base_head(base_features).squeeze(1)
        window_logits = self.window_head(window_features)
        return {
            "base_logits": base_logits,
            "window_logits": window_logits,
        }
