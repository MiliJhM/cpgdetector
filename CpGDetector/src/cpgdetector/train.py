from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .baseline import (
    evaluate_logistic_baseline,
    evaluate_traditional_baseline,
    logistic_baseline_scores,
    traditional_baseline_scores,
)
from .data import CpGAnnotations, CpGWindowDataset, GenomeStore
from .losses import multitask_loss
from .metrics import best_threshold, classification_metrics, regression_metrics
from .model import MultiTaskCpGNet
from .utils import load_yaml, resolve_device, save_json, save_yaml, set_seed
from .visualize import plot_baseline_comparison, plot_roc_pr_curves, plot_training_curves


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
    return DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=shuffle,
        num_workers=int(train_cfg.get("num_workers", 0)),
        pin_memory=pin_memory,
    )


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


def train_one_epoch(
    model: MultiTaskCpGNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    config: dict,
    base_pos_weight: float | None,
) -> dict[str, float]:
    model.train()
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    totals = {"loss": 0.0, "base_bce": 0.0, "dice": 0.0, "window_bce": 0.0}
    count = 0
    for batch in tqdm(loader, desc="train", leave=False, ascii=True):
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        fraction = batch["fraction"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(x)
            loss, parts = multitask_loss(
                outputs,
                mask,
                fraction,
                lambda_window=config["training"]["lambda_window"],
                lambda_dice=config["training"]["lambda_dice"],
                base_pos_weight=base_pos_weight,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_size = x.shape[0]
        count += batch_size
        for key in totals:
            totals[key] += parts[key] * batch_size
    return {f"train_{key}": value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: MultiTaskCpGNet,
    loader: DataLoader,
    device: torch.device,
    config: dict,
    base_threshold: float,
    base_pos_weight: float | None,
) -> dict[str, float]:
    model.eval()
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    losses = {"loss": 0.0, "base_bce": 0.0, "dice": 0.0, "window_bce": 0.0}
    count = 0
    base_targets: list[np.ndarray] = []
    base_scores: list[np.ndarray] = []
    window_targets: list[np.ndarray] = []
    window_scores: list[np.ndarray] = []
    fractions: list[np.ndarray] = []
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
                base_pos_weight=base_pos_weight,
            )
        batch_size = x.shape[0]
        count += batch_size
        for key in losses:
            losses[key] += parts[key] * batch_size
        base_targets.append(mask.detach().cpu().numpy().reshape(-1))
        base_scores.append(torch.sigmoid(outputs["base_logits"]).detach().cpu().numpy().reshape(-1))
        frac_np = fraction.detach().cpu().numpy().reshape(-1)
        fractions.append(frac_np)
        window_targets.append((frac_np > 0).astype(np.float32))
        window_scores.append(torch.sigmoid(outputs["window_logits"]).detach().cpu().numpy().reshape(-1))

    y_base = np.concatenate(base_targets)
    s_base = np.concatenate(base_scores)
    y_window = np.concatenate(window_targets)
    s_window = np.concatenate(window_scores)
    y_fraction = np.concatenate(fractions)
    result = {f"val_{key}": value / max(count, 1) for key, value in losses.items()}
    result.update({f"val_base_{k}": v for k, v in classification_metrics(y_base, s_base, base_threshold).items()})
    result.update({f"val_window_{k}": v for k, v in classification_metrics(y_window, s_window, 0.5).items()})
    result.update({f"val_fraction_{k}": v for k, v in regression_metrics(y_fraction, s_window).items()})
    return result


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
        window_targets.append((fraction > 0).astype(np.int32))
        window_scores.append(torch.sigmoid(outputs["window_logits"]).detach().cpu().numpy().reshape(-1))
    return {
        "CNN base segmentation": (np.concatenate(base_targets), np.concatenate(base_scores)),
        "CNN window head": (np.concatenate(window_targets), np.concatenate(window_scores)),
    }


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

    # Try triton availability early, since it can cause obscure errors if the CUDA version is incompatible. It's not strictly required, so we can still run without it if it's not available.
    triton_available = False
    if config['training'].get('compile', True):
        try:
            import triton  # noqa: F401
            triton_available = True
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"]))
    scaler = torch.amp.GradScaler(device.type, enabled=bool(config["training"].get("amp", True)) and device.type == "cuda")


    model = torch.compile(model) if bool(config["training"].get("compile", True)) else model

    metrics_rows: list[dict[str, float]] = []
    best_metric = -math.inf
    best_epoch = 0
    best_threshold_value = 0.5
    patience = int(config["training"].get("early_stopping_patience", 5))
    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        print(f"Epoch {epoch}")
        train_metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, config, base_pos_weight)
        y_base, s_base = collect_base_scores(model, val_loader, device, config)
        best_threshold_value, base_metrics = best_threshold(y_base, s_base, [float(x) for x in config["training"]["threshold_grid"]])
        val_metrics = evaluate(model, val_loader, device, config, best_threshold_value, base_pos_weight)
        row = {"epoch": epoch, "threshold": best_threshold_value, **train_metrics, **val_metrics}
        row.update({f"val_base_best_{k}": v for k, v in base_metrics.items()})
        metrics_rows.append(row)
        pd.DataFrame(metrics_rows).to_csv(run_dir / "metrics.csv", index=False)
        print(
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"val_base_pr_auc={val_metrics['val_base_pr_auc']:.4f} "
            f"val_base_f1={base_metrics['f1']:.4f} threshold={best_threshold_value:.2f}"
        )
        score = base_metrics["f1"]
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "threshold": best_threshold_value,
                    "base_pos_weight": base_pos_weight,
                    "epoch": epoch,
                },
                run_dir / "best_model.pt",
            )
        elif epoch - best_epoch >= patience:
            print(f"Early stopping after epoch {epoch}; best epoch was {best_epoch}")
            break

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
        "best_val_base_f1": best_metric,
        "base_threshold": best_threshold_value,
        "base_pos_weight": base_pos_weight,
        "traditional_baseline_window": baseline_metrics,
        "logistic_baseline_window": logistic_metrics,
    }
    save_json(summary, run_dir / "summary.json")
    plot_baseline_comparison(run_dir / "metrics.csv", run_dir / "summary.json", run_dir / "baseline_comparison.png")
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    curve_series = collect_model_curve_scores(model, val_loader, device, config)
    curve_series["Traditional rule"] = traditional_baseline_scores(val_ds, max_items=min(5000, len(val_ds)))
    curve_series["Logistic baseline"] = logistic_baseline_scores(
        train_ds,
        val_ds,
        max_train=min(10000, len(train_ds)),
        max_val=min(5000, len(val_ds)),
    )
    plot_roc_pr_curves(
        curve_series,
        run_dir / "roc_pr_curves.png",
        note="Traditional and logistic baselines are window-level. CNN base segmentation is base-level; CNN window head is window-level.",
    )
    print(f"Saved run artifacts to {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
