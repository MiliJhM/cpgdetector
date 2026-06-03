from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .baseline import (
    evaluate_logistic_baseline,
    evaluate_traditional_baseline,
    logistic_baseline_scores,
    traditional_baseline_scores,
)
from .data import CpGAnnotations, CpGWindowDataset, GenomeStore, valid_encoded_window
from .losses import (
    loss_components,
    multitask_loss,
    normalized_gradnorm_weights,
    scalar_loss_parts,
    window_fraction_logits,
    window_presence_target,
)
from .metrics import BinaryMetricAccumulator, best_threshold, classification_metrics, regression_metrics
from .model import MultiTaskCpGNet
from .utils import load_yaml, resolve_device, save_json, save_yaml, set_seed
from .visualize import plot_baseline_comparison, plot_region_prediction, plot_roc_pr_curves, plot_training_curves


def build_dataset(config: dict, split: str, genome: GenomeStore, annotations: CpGAnnotations) -> CpGWindowDataset:
    data_cfg = config["data"]
    if split == "train":
        chroms = data_cfg["train_chroms"]
        max_windows = data_cfg.get("max_train_windows")
        mode = "train"
        seed_offset = 0
    elif split == "val":
        chroms = data_cfg["val_chroms"]
        max_windows = data_cfg.get("max_val_windows")
        mode = "val"
        seed_offset = 100
    elif split == "test":
        chroms = data_cfg["test_chroms"]
        max_windows = data_cfg.get("max_test_windows")
        mode = "test"
        seed_offset = 200
    else:
        raise ValueError(f"Unknown split: {split}")
    return CpGWindowDataset(
        genome=genome,
        annotations=annotations,
        chroms=chroms,
        window_size=data_cfg["window_size"],
        stride=data_cfg["stride"],
        max_windows=max_windows,
        seed=int(config["seed"]) + seed_offset,
        mode=mode,
        positive_fraction=data_cfg.get("positive_fraction", 0.45),
        hard_negative_fraction=data_cfg.get("hard_negative_fraction", 0.25),
        min_gc_for_hard_negative=data_cfg.get("min_gc_for_hard_negative", 0.50),
        boundary_flank=data_cfg.get("boundary_flank", data_cfg.get("boundary_fl", 2000)),
    )


def dataloader(dataset: CpGWindowDataset, config: dict, shuffle: bool) -> DataLoader:
    train_cfg = config["training"]
    pin_memory = bool(train_cfg.get("pin_memory", False)) and torch.cuda.is_available()
    num_workers = int(train_cfg.get("num_workers", 0))
    kwargs = {}
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(train_cfg.get("persistent_workers", True))
        kwargs["prefetch_factor"] = int(train_cfg.get("prefetch_factor", 4))
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=cpg_collate,
        **kwargs,
    )


def cpg_collate(batch: list[dict]) -> dict[str, torch.Tensor | list[str] | list[int]]:
    seq_idx = torch.stack([item["seq_idx"] for item in batch], dim=0).long()
    x = F.one_hot(seq_idx, num_classes=4).permute(0, 2, 1).contiguous().float()
    mask = torch.stack([item["mask"] for item in batch], dim=0).float()
    fraction = torch.stack([item["fraction"] for item in batch], dim=0).float()
    has_cpg = torch.stack([item["has_cpg"] for item in batch], dim=0).float()
    return {
        "x": x,
        "mask": mask,
        "fraction": fraction,
        "has_cpg": has_cpg,
        "chrom": [item["chrom"] for item in batch],
        "start": [int(item["start"]) for item in batch],
    }


def estimate_pos_weight(dataset: CpGWindowDataset, max_items: int) -> float:
    positives = 0.0
    total = 0.0
    limit = min(max_items, len(dataset))
    for idx in range(limit):
        mask = dataset[idx]["mask"]
        positives += float(mask.sum().item())
        total += float(mask.numel())
    negatives = max(total - positives, 1.0)
    positives = max(positives, 1.0)
    return float(min(negatives / positives, 50.0))


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


def load_model_state_compatible(model: torch.nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    meaningful_missing = [
        key
        for key in missing
        if not key.startswith("loss_log_vars.") and not key.startswith("gradnorm_log_weights.")
    ]
    if meaningful_missing or unexpected:
        raise RuntimeError(
            f"Checkpoint is incompatible. Missing keys: {meaningful_missing}; unexpected keys: {unexpected}"
        )


def maybe_compile_model(model: torch.nn.Module, enabled: bool) -> torch.nn.Module:
    if not enabled:
        return model
    try:
        return torch.compile(model)
    except Exception as exc:
        print(f"Warning: torch.compile failed; continuing without compilation: {exc}", file=sys.stderr)
        return model


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict):
    train_cfg = config["training"]
    name = str(train_cfg.get("scheduler", "none")).lower()
    if name in {"none", "off", "false"}:
        return None, "none"
    if name == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(train_cfg.get("plateau_factor", 0.5)),
            patience=int(train_cfg.get("plateau_patience", 3)),
            min_lr=float(train_cfg.get("min_lr", 0.0)),
        )
        return scheduler, name
    if name == "warmup_cosine":
        epochs = int(train_cfg["epochs"])
        warmup_epochs = max(0, int(train_cfg.get("warmup_epochs", 0)))
        min_lr = float(train_cfg.get("min_lr", 0.0))
        base_lr = float(train_cfg["lr"])
        min_factor = min_lr / base_lr if base_lr > 0 else 0.0

        def lr_lambda(epoch_idx: int) -> float:
            epoch_num = epoch_idx + 1
            if warmup_epochs > 0 and epoch_num <= warmup_epochs:
                return max(epoch_num / warmup_epochs, min_factor)
            decay_epochs = max(1, epochs - warmup_epochs)
            progress = min(1.0, max(0.0, (epoch_num - warmup_epochs) / decay_epochs))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_factor + (1.0 - min_factor) * cosine

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), name
    raise ValueError(f"Unknown scheduler: {name}")


def window_task_settings(config: dict) -> dict:
    train_cfg = config["training"]
    return {
        "window_target_mode": str(train_cfg.get("window_target_mode", "mixed")),
        "window_presence_threshold": float(train_cfg.get("window_presence_threshold", 0.05)),
        "lambda_window_fraction": float(train_cfg.get("lambda_window_fraction", 1.0)),
        "window_fraction_loss": str(train_cfg.get("window_fraction_loss", "smooth_l1")),
    }

def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def monitor_settings(config: dict) -> dict:
    train_cfg = config["training"]
    monitor_cfg = dict(train_cfg.get("monitor", {}))
    return {
        "base_metric": str(monitor_cfg.get("base_metric", "val_base_best_f1")),
        "window_metric": str(monitor_cfg.get("window_metric", "val_window_pr_auc")),
        "base_weight": float(monitor_cfg.get("base_weight", 0.70)),
        "window_weight": float(monitor_cfg.get("window_weight", 0.30)),
    }


def composite_monitor_score(row: dict[str, float], config: dict) -> float:
    settings = monitor_settings(config)
    base_metric = settings["base_metric"]
    window_metric = settings["window_metric"]
    if base_metric not in row:
        raise KeyError(f"Monitor base metric '{base_metric}' is not available in epoch metrics")
    if window_metric not in row:
        raise KeyError(f"Monitor window metric '{window_metric}' is not available in epoch metrics")
    base_weight = settings["base_weight"]
    window_weight = settings["window_weight"]
    if base_weight < 0 or window_weight < 0:
        raise ValueError("Monitor weights must be non-negative")
    total_weight = base_weight + window_weight
    if total_weight <= 0:
        raise ValueError("At least one monitor weight must be positive")
    base_value = float(row[base_metric])
    window_value = float(row[window_metric])
    if not math.isfinite(base_value):
        base_value = 0.0
    if not math.isfinite(window_value):
        window_value = 0.0
    return (base_weight * base_value + window_weight * window_value) / total_weight


def loss_log_vars_for(model: torch.nn.Module, config: dict) -> torch.nn.ParameterDict | None:
    method = str(config["training"].get("mtl_method", "fixed")).lower()
    if method not in {"uncertainty", "uncertainty_weighting"}:
        return None
    return getattr(unwrap_model(model), "loss_log_vars")


def gradnorm_log_weights_for(model: torch.nn.Module, config: dict) -> torch.nn.ParameterDict | None:
    method = str(config["training"].get("mtl_method", "fixed")).lower()
    if method != "gradnorm":
        return None
    return getattr(unwrap_model(model), "gradnorm_log_weights")


def current_loss_weights(model: torch.nn.Module) -> dict[str, float]:
    unwrapped = unwrap_model(model)
    weights = {}
    log_vars = getattr(unwrapped, "loss_log_vars", None)
    if log_vars is not None:
        weights.update({f"uncertainty_{key}_loss_weight": float(torch.exp(-value.detach()).cpu()) for key, value in log_vars.items()})
    gradnorm_log_weights = getattr(unwrapped, "gradnorm_log_weights", None)
    if gradnorm_log_weights is not None:
        gradnorm_weights = normalized_gradnorm_weights(gradnorm_log_weights)
        weights.update({f"gradnorm_{key}_loss_weight": float(value.detach().cpu()) for key, value in gradnorm_weights.items()})
    return weights


def optimizer_parameter_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    decay_params = []
    no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("loss_log_vars.") or name.startswith("gradnorm_log_weights."):
            no_decay_params.append(parameter)
        else:
            decay_params.append(parameter)
    groups = [{"params": decay_params, "weight_decay": float(weight_decay)}]
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return groups


def shared_encoder_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in unwrap_model(model).encoder.parameters() if parameter.requires_grad]


def detached_global_grad_norm(loss: torch.Tensor, parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    grads = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    terms = [grad.detach().pow(2).sum() for grad in grads if grad is not None]
    if not terms:
        return torch.zeros((), dtype=loss.dtype, device=loss.device)
    return torch.sqrt(torch.stack(terms).sum() + 1e-12)


def gradnorm_multitask_loss(
    outputs: dict[str, torch.Tensor],
    mask: torch.Tensor,
    fraction: torch.Tensor,
    model: torch.nn.Module,
    config: dict,
    base_pos_weight: float | None,
    gradnorm_state: dict,
) -> tuple[torch.Tensor, dict[str, float]]:
    train_cfg = config["training"]
    components = loss_components(
        outputs,
        mask,
        fraction,
        lambda_window=train_cfg["lambda_window"],
        lambda_dice=train_cfg["lambda_dice"],
        base_pos_weight=base_pos_weight,
        **window_task_settings(config)
    )
    gradnorm_log_weights = gradnorm_log_weights_for(model, config)
    if gradnorm_log_weights is None:
        raise ValueError("GradNorm requested but model does not expose gradnorm_log_weights")
    weights = normalized_gradnorm_weights(gradnorm_log_weights)
    task_losses = torch.stack([components["base_task"], components["window_task"]])

    if "initial_task_losses" not in gradnorm_state:
        gradnorm_state["initial_task_losses"] = task_losses.detach().clamp_min(1e-8)
    initial_losses = gradnorm_state["initial_task_losses"].to(device=task_losses.device, dtype=task_losses.dtype)

    shared_params = shared_encoder_parameters(model)
    base_grad_norm = detached_global_grad_norm(components["base_task"], shared_params)
    window_grad_norm = detached_global_grad_norm(components["window_task"], shared_params)
    raw_grad_norms = torch.stack([base_grad_norm, window_grad_norm])
    weighted_grad_norms = torch.stack([weights["base"], weights["window"]]) * raw_grad_norms

    loss_ratios = task_losses.detach().clamp_min(1e-8) / initial_losses
    inverse_train_rates = loss_ratios / loss_ratios.mean().clamp_min(1e-8)
    alpha = float(train_cfg.get("gradnorm_alpha", 1.5))
    target_grad_norms = weighted_grad_norms.detach().mean() * inverse_train_rates.pow(alpha)
    gradnorm_loss = torch.sum(torch.abs(weighted_grad_norms - target_grad_norms.detach()))

    model_loss = (
        weights["base"].detach() * components["base_task"]
        + weights["window"].detach() * components["window_task"]
        + float(train_cfg.get("lambda_consistency", 0.0)) * components["consistency"]
    )
    total = model_loss + float(train_cfg.get("gradnorm_lambda", 1.0)) * gradnorm_loss
    parts = scalar_loss_parts(
        total,
        components,
        float(train_cfg.get("lambda_consistency", 0.0)),
        float(weights["base"].detach().cpu()),
        float(weights["window"].detach().cpu()),
        extra={
            "gradnorm_loss": float(gradnorm_loss.detach().cpu()),
            "gradnorm_base_grad_norm": float(weighted_grad_norms[0].detach().cpu()),
            "gradnorm_window_grad_norm": float(weighted_grad_norms[1].detach().cpu()),
            "gradnorm_base_raw_grad_norm": float(raw_grad_norms[0].detach().cpu()),
            "gradnorm_window_raw_grad_norm": float(raw_grad_norms[1].detach().cpu()),
            "gradnorm_base_target_grad_norm": float(target_grad_norms[0].detach().cpu()),
            "gradnorm_window_target_grad_norm": float(target_grad_norms[1].detach().cpu()),
            "gradnorm_base_inverse_rate": float(inverse_train_rates[0].detach().cpu()),
            "gradnorm_window_inverse_rate": float(inverse_train_rates[1].detach().cpu()),
        },
    )
    return total, parts


def train_one_epoch(
    model: MultiTaskCpGNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    config: dict,
    base_pos_weight: float | None,
    gradnorm_state: dict | None = None,
) -> dict[str, float]:
    model.train()
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    totals: dict[str, float] = {}
    count = 0
    gradnorm_state = gradnorm_state if gradnorm_state is not None else {}
    mtl_method = str(config["training"].get("mtl_method", "fixed")).lower()
    for batch in tqdm(loader, desc="train", leave=False, ascii=True):
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        fraction = batch["fraction"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(x)
            if mtl_method == "gradnorm":
                loss, parts = gradnorm_multitask_loss(
                    outputs,
                    mask,
                    fraction,
                    model=model,
                    config=config,
                    base_pos_weight=base_pos_weight,
                    gradnorm_state=gradnorm_state,
                )
            else:
                loss, parts = multitask_loss(
                    outputs,
                    mask,
                    fraction,
                    lambda_window=config["training"]["lambda_window"],
                    lambda_dice=config["training"]["lambda_dice"],
                    lambda_consistency=config["training"].get("lambda_consistency", 0.0),
                    mtl_method=config["training"].get("mtl_method", "fixed"),
                    loss_log_vars=loss_log_vars_for(model, config),
                    base_pos_weight=base_pos_weight,
                    **window_task_settings(config)
                )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = x.shape[0]
        count += batch_size
        for key in parts:
            totals[key] = totals.get(key, 0.0) + parts[key] * batch_size
    return {f"train_{key}": value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: MultiTaskCpGNet,
    loader: DataLoader,
    device: torch.device,
    config: dict,
    base_threshold: float | None,
    base_pos_weight: float | None,
) -> tuple[dict[str, float], float, dict[str, float]]:
    model.eval()
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    losses: dict[str, float] = {}
    count = 0
    thresholds = [float(x) for x in config["training"]["threshold_grid"]]
    bins = int(config["training"].get("metric_bins", 2048))
    base_acc = BinaryMetricAccumulator(thresholds, bins=bins, device=device)
    window_acc = BinaryMetricAccumulator([0.5], bins=bins, device=device)
    frac_sse = torch.zeros((), dtype=torch.float64, device=device)
    frac_sae = torch.zeros((), dtype=torch.float64, device=device)
    frac_count = torch.zeros((), dtype=torch.float64, device=device)
    for batch in tqdm(loader, desc="eval", leave=False, ascii=True):
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        fraction = batch["fraction"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(x)
            loss, parts = multitask_loss(
                outputs,
                mask,
                fraction,
                lambda_window=config["training"]["lambda_window"],
                lambda_dice=config["training"]["lambda_dice"],
                lambda_consistency=config["training"].get("lambda_consistency", 0.0),
                mtl_method=config["training"].get("mtl_method", "fixed"),
                loss_log_vars=loss_log_vars_for(model, config),
                gradnorm_log_weights=gradnorm_log_weights_for(model, config),
                base_pos_weight=base_pos_weight,
                **window_task_settings(config)
            )
        batch_size = x.shape[0]
        count += batch_size
        for key in parts:
            losses[key] = losses.get(key, 0.0) + parts[key] * batch_size
        base_scores = torch.sigmoid(outputs["base_logits"])
        window_scores = torch.sigmoid(outputs["window_logits"])
        fraction_scores = torch.sigmoid(window_fraction_logits(outputs))
        base_acc.update(mask, base_scores)
        window_acc.update(window_presence_target(fraction, window_task_settings(config)["window_presence_threshold"]), window_scores)
        frac_sse += torch.sum((fraction.double() - fraction_scores.double()) ** 2)
        frac_sae += torch.sum(torch.abs(fraction.double() - fraction_scores.double()))
        frac_count += fraction.numel()

    if base_threshold is None:
        selected_threshold, base_best_metrics = base_acc.best_f1()
    else:
        selected_threshold = float(base_threshold)
        base_best_metrics = base_acc.metrics_at(selected_threshold)
    result = {f"val_{key}": value / max(count, 1) for key, value in losses.items()}
    result.update({f"val_base_{k}": v for k, v in base_best_metrics.items()})
    result.update({f"val_window_{k}": v for k, v in window_acc.metrics_at(0.5).items()})
    result.update(
        {
            "val_fraction_mse": float((frac_sse / torch.clamp(frac_count, min=1.0)).detach().cpu()),
            "val_fraction_mae": float((frac_sae / torch.clamp(frac_count, min=1.0)).detach().cpu()),
        }
    )
    return result, selected_threshold, base_best_metrics


@torch.no_grad()
def collect_base_scores(model: MultiTaskCpGNet, loader: DataLoader, device: torch.device, config: dict) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    targets: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    for batch in tqdm(loader, desc="threshold", leave=False, ascii=True):
        x = batch["x"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(x)
        targets.append(batch["mask"].numpy().reshape(-1))
        scores.append(torch.sigmoid(outputs["base_logits"]).detach().cpu().numpy().reshape(-1))
    return np.concatenate(targets), np.concatenate(scores)


@torch.no_grad()
def collect_model_curve_scores(
    model: MultiTaskCpGNet,
    loader: DataLoader,
    device: torch.device,
    config: dict,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    model.eval()
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    base_targets: list[np.ndarray] = []
    base_scores: list[np.ndarray] = []
    window_targets: list[np.ndarray] = []
    window_scores: list[np.ndarray] = []
    for batch in tqdm(loader, desc="curves", leave=False, ascii=True):
        x = batch["x"].to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(x)
        base_targets.append(batch["mask"].numpy().reshape(-1))
        base_scores.append(torch.sigmoid(outputs["base_logits"]).detach().cpu().numpy().reshape(-1))
        fraction = batch["fraction"].numpy().reshape(-1)
        threshold = window_task_settings(config)["window_presence_threshold"]
        window_targets.append((fraction >= threshold).astype(np.int32))
        window_scores.append(torch.sigmoid(outputs["window_logits"]).detach().cpu().numpy().reshape(-1))
    return {
        "CNN base segmentation": (np.concatenate(base_targets), np.concatenate(base_scores)),
        "CNN window head": (np.concatenate(window_targets), np.concatenate(window_scores)),
    }


def save_curve_scores(series: dict[str, tuple[np.ndarray, np.ndarray]], output_path: str | Path) -> None:
    payload: dict[str, np.ndarray] = {"labels": np.asarray(list(series.keys()), dtype=object)}
    for idx, (_label, (targets, scores)) in enumerate(series.items()):
        payload[f"series_{idx}_targets"] = np.asarray(targets)
        payload[f"series_{idx}_scores"] = np.asarray(scores)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **payload)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the CpG multitask segmentation model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args(argv)


    config = load_yaml(args.config)
    if args.run_dir:
        config["output"]["run_dir"] = args.run_dir
    run_dir = Path(config["output"]["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "config.yaml")
    set_seed(int(config["seed"]))

    # Try Triton availability early; torch.compile is useful on A100/Linux but optional.
    if config["training"].get("compile", True):
        try:
            import triton  # noqa: F401
        except ImportError:
            print("Warning: Triton is not available. If you have a compatible NVIDIA GPU, consider installing Triton for potential performance improvements.", file=sys.stderr)


    device = resolve_device(config.get("device", "auto"))
    if config.get("device") == "cuda" and device.type != "cuda":
        raise RuntimeError("Config requests CUDA, but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    genome = GenomeStore(config["data"]["genome_dir"])
    annotations = CpGAnnotations(config["data"]["cpg_table"])
    train_ds = build_dataset(config, "train", genome, annotations)
    val_ds = build_dataset(config, "val", genome, annotations)
    print(f"Dataset sizes: train={len(train_ds)}, val={len(val_ds)}")

    base_pos_weight = config["training"].get("base_pos_weight")
    if base_pos_weight is None:
        base_pos_weight = estimate_pos_weight(train_ds, int(config["training"].get("estimate_pos_weight_batches", 2048)))
    print(f"Base positive weight: {base_pos_weight:.4f}")

    train_loader = dataloader(train_ds, config, shuffle=True)
    val_loader = dataloader(val_ds, config, shuffle=False)
    model = MultiTaskCpGNet(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(model, float(config["training"]["weight_decay"])),
        lr=float(config["training"]["lr"]),
    )
    scheduler, scheduler_name = build_scheduler(optimizer, config)
    scaler = torch.amp.GradScaler(device.type, enabled=bool(config["training"].get("amp", True)) and device.type == "cuda")


    compile_enabled = bool(config["training"].get("compile", True))
    if str(config["training"].get("mtl_method", "fixed")).lower() == "gradnorm" and compile_enabled:
        print("Warning: disabling torch.compile for GradNorm because it uses explicit autograd.grad calls.")
        compile_enabled = False
    model = maybe_compile_model(model, compile_enabled)

    metrics_rows: list[dict[str, float]] = []
    best_metric = -math.inf
    best_epoch = 0
    best_row: dict[str, float] = {}
    best_threshold_value = 0.5
    patience = int(config["training"].get("early_stopping_patience", 5))
    gradnorm_state: dict = {}
    monitor_cfg = monitor_settings(config)
    print(
        "Monitor score: "
        f"{monitor_cfg['base_weight']:.3g}*{monitor_cfg['base_metric']} + "
        f"{monitor_cfg['window_weight']:.3g}*{monitor_cfg['window_metric']}"
    )
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        print(f"Epoch {epoch}")
        epoch_lr = current_lr(optimizer)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            config,
            base_pos_weight,
            gradnorm_state=gradnorm_state,
        )
        val_metrics, best_threshold_value, base_metrics = evaluate(model, val_loader, device, config, None, base_pos_weight)
        row = {"epoch": epoch, "lr": epoch_lr, "threshold": best_threshold_value, **train_metrics, **val_metrics}
        row.update({f"val_base_best_{k}": v for k, v in base_metrics.items()})
        monitor_score = composite_monitor_score(row, config)
        row["monitor_score"] = monitor_score
        metrics_rows.append(row)
        pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)
        print(
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"monitor_score={monitor_score:.4f} "
            f"val_base_pr_auc={val_metrics['val_base_pr_auc']:.4f} "
            f"val_base_f1={base_metrics['f1']:.4f} threshold={best_threshold_value:.2f} "
            f"val_window_pr_auc={val_metrics['val_window_pr_auc']:.4f} "
            f"val_window_f1={val_metrics['val_window_f1']:.4f} "
            f"val_window_precision={val_metrics['val_window_precision']:.4f} "
            f"val_window_recall={val_metrics['val_window_recall']:.4f} "
            f"val_fraction_mae={val_metrics['val_fraction_mae']:.4f} "
            f"train_consistency={train_metrics.get('train_consistency', 0.0):.4f} "
            f"gradnorm_loss={train_metrics.get('train_gradnorm_loss', 0.0):.4f} "
            f"base_w={train_metrics.get('train_base_loss_weight', 1.0):.3g} "
            f"window_w={train_metrics.get('train_window_loss_weight', 1.0):.3g} "
            f"lr={epoch_lr:.3g}"
        )
        score = monitor_score
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            best_row = dict(row)
            torch.save(
                {
                    "model_state": unwrap_model(model).state_dict(),
                    "config": config,
                    "threshold": best_threshold_value,
                    "base_pos_weight": base_pos_weight,
                    "epoch": epoch,
                    "monitor_score": monitor_score,
                    "monitor": monitor_cfg,
                },
                run_dir / "best_model.pt",
            )
        elif epoch - best_epoch >= patience:
            print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}")
            break
        if scheduler is not None and scheduler_name == "reduce_on_plateau":
            scheduler.step(monitor_score)
        elif scheduler is not None:
            scheduler.step()

    plot_training_curves(run_dir / "metrics.csv", run_dir / "training_curves.png")
    baseline_metrics = evaluate_traditional_baseline(val_ds, threshold=0.5, max_items=min(5000, len(val_ds)))
    logistic_metrics = evaluate_logistic_baseline(train_ds, val_ds, max_train=min(10000, len(train_ds)), max_val=min(5000, len(val_ds)))
    summary = {
        "run_dir": str(run_dir),
        "device": str(device),
        "cuda_device": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "best_epoch": best_epoch,
        "best_monitor_score": best_metric,
        "best_val_base_f1": best_row.get("val_base_best_f1"),
        "best_val_window_pr_auc": best_row.get("val_window_pr_auc"),
        "monitor": monitor_cfg,
        "base_threshold": best_row.get("threshold", best_threshold_value),
        "base_pos_weight": base_pos_weight,
        "scheduler": scheduler_name,
        "final_lr": current_lr(optimizer),
        "mtl_method": config["training"].get("mtl_method", "fixed"),
        "lambda_consistency": config["training"].get("lambda_consistency", 0.0),
        "loss_weights": current_loss_weights(model),
        "traditional_baseline_window": baseline_metrics,
        "logistic_baseline_window": logistic_metrics,
    }
    save_json(summary, run_dir / "summary.json")
    plot_baseline_comparison(run_dir / "metrics.csv", run_dir / "summary.json", run_dir / "baseline_comparison.png")
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    load_model_state_compatible(unwrap_model(model), checkpoint["model_state"])
    curve_series = collect_model_curve_scores(model, val_loader, device, config)
    curve_series["Traditional rule"] = traditional_baseline_scores(val_ds, max_items=min(5000, len(val_ds)))
    curve_series["Logistic baseline"] = logistic_baseline_scores(
        train_ds,
        val_ds,
        max_train=min(10000, len(train_ds)),
        max_val=min(5000, len(val_ds)),
    )
    save_curve_scores(curve_series, run_dir / "curve_scores.npz")
    plot_roc_pr_curves(
        curve_series,
        run_dir / "roc_pr_curves.png",
        note="Traditional and logistic baselines are window-level. CNN base segmentation is base-level; CNN window head is window-level.",
    )
    print(f"Saved run artifacts to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
