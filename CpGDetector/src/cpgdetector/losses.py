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


def normalized_gradnorm_weights(
    gradnorm_log_weights: torch.nn.ParameterDict | dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    stacked = torch.stack([gradnorm_log_weights["base"], gradnorm_log_weights["window"]])
    weights = torch.softmax(stacked, dim=0) * 2.0
    return {"base": weights[0], "window": weights[1]}


def loss_components(
    outputs: dict[str, torch.Tensor],
    mask: torch.Tensor,
    fraction: torch.Tensor,
    lambda_window: float,
    lambda_dice: float,
    base_pos_weight: float | None = None,
) -> dict[str, torch.Tensor]:
    pos_weight = None
    if base_pos_weight is not None:
        pos_weight = torch.tensor(float(base_pos_weight), device=mask.device)
    base_bce = F.binary_cross_entropy_with_logits(outputs["base_logits"], mask, pos_weight=pos_weight)
    dice = dice_loss_from_logits(outputs["base_logits"], mask)
    window_bce = F.binary_cross_entropy_with_logits(outputs["window_logits"], fraction)
    base_task = base_bce + float(lambda_dice) * dice
    window_task = float(lambda_window) * window_bce
    base_fraction = torch.sigmoid(outputs["base_logits"]).mean(dim=1, keepdim=True)
    window_fraction = torch.sigmoid(outputs["window_logits"])
    consistency = F.mse_loss(window_fraction, base_fraction)
    return {
        "base_bce": base_bce,
        "dice": dice,
        "window_bce": window_bce,
        "base_task": base_task,
        "window_task": window_task,
        "consistency": consistency,
    }


def scalar_loss_parts(
    total: torch.Tensor,
    components: dict[str, torch.Tensor],
    lambda_consistency: float,
    base_weight: float,
    window_weight: float,
    extra: dict[str, float] | None = None,
) -> dict[str, float]:
    parts = {
        "loss": float(total.detach().cpu()),
        "base_bce": float(components["base_bce"].detach().cpu()),
        "dice": float(components["dice"].detach().cpu()),
        "window_bce": float(components["window_bce"].detach().cpu()),
        "base_task": float(components["base_task"].detach().cpu()),
        "window_task": float(components["window_task"].detach().cpu()),
        "consistency": float(components["consistency"].detach().cpu()),
        "lambda_consistency": float(lambda_consistency),
        "base_loss_weight": float(base_weight),
        "window_loss_weight": float(window_weight),
    }
    if extra:
        parts.update(extra)
    return parts


def multitask_loss(
    outputs: dict[str, torch.Tensor],
    mask: torch.Tensor,
    fraction: torch.Tensor,
    lambda_window: float,
    lambda_dice: float,
    lambda_consistency: float = 0.0,
    mtl_method: str = "fixed",
    loss_log_vars: torch.nn.ParameterDict | dict[str, torch.Tensor] | None = None,
    gradnorm_log_weights: torch.nn.ParameterDict | dict[str, torch.Tensor] | None = None,
    base_pos_weight: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    components = loss_components(outputs, mask, fraction, lambda_window, lambda_dice, base_pos_weight)
    base_task = components["base_task"]
    window_task = components["window_task"]
    consistency = components["consistency"]

    method = str(mtl_method).lower()
    if method in {"fixed", "static", "none"}:
        total = base_task + window_task
        base_weight = 1.0
        window_weight = 1.0
    elif method in {"uncertainty", "uncertainty_weighting"}:
        if loss_log_vars is None:
            raise ValueError("mtl_method='uncertainty' requires loss_log_vars")
        base_log_var = loss_log_vars["base"]
        window_log_var = loss_log_vars["window"]
        base_weight_tensor = torch.exp(-base_log_var)
        window_weight_tensor = torch.exp(-window_log_var)
        total = base_weight_tensor * base_task + base_log_var + window_weight_tensor * window_task + window_log_var
        base_weight = float(base_weight_tensor.detach().cpu())
        window_weight = float(window_weight_tensor.detach().cpu())
    elif method == "gradnorm":
        if gradnorm_log_weights is None:
            raise ValueError("mtl_method='gradnorm' requires gradnorm_log_weights")
        weights = normalized_gradnorm_weights(gradnorm_log_weights)
        total = weights["base"] * base_task + weights["window"] * window_task
        base_weight = float(weights["base"].detach().cpu())
        window_weight = float(weights["window"].detach().cpu())
    else:
        raise ValueError(f"Unknown multitask learning method: {mtl_method}")

    total = total + float(lambda_consistency) * consistency
    return total, scalar_loss_parts(total, components, lambda_consistency, base_weight, window_weight)
