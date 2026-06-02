from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.reshape(probs.shape[0], -1)
    targets = targets.reshape(targets.shape[0], -1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    mask: torch.Tensor,
    fraction: torch.Tensor,
    lambda_window: float,
    lambda_dice: float,
    base_pos_weight: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    pos_weight = None
    if base_pos_weight is not None:
        pos_weight = torch.tensor(float(base_pos_weight), device=mask.device)
    base_bce = F.binary_cross_entropy_with_logits(outputs["base_logits"], mask, pos_weight=pos_weight)
    dice = dice_loss_from_logits(outputs["base_logits"], mask)
    window_bce = F.binary_cross_entropy_with_logits(outputs["window_logits"], fraction)
    total = base_bce + float(lambda_dice) * dice + float(lambda_window) * window_bce
    parts = {
        "loss": float(total.detach().cpu()),
        "base_bce": float(base_bce.detach().cpu()),
        "dice": float(dice.detach().cpu()),
        "window_bce": float(window_bce.detach().cpu()),
    }
    return total, parts
