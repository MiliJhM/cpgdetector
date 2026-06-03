from __future__ import annotations

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def plot_training_curves(metrics_csv: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(metrics_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3 if "lr" in df else 2, figsize=(15 if "lr" in df else 11, 4))
    axes[0].plot(df["epoch"], df["train_loss"], marker="o", label="train loss")
    if "val_loss" in df:
        axes[0].plot(df["epoch"], df["val_loss"], marker="o", label="val loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    for col in ["val_base_pr_auc", "val_base_f1", "val_window_pr_auc"]:
        if col in df:
            axes[1].plot(df["epoch"], df[col], marker="o", label=col)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    if "lr" in df:
        axes[2].plot(df["epoch"], df["lr"], marker="o", color="tab:purple", label="learning rate")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("Learning Rate")
        axes[2].set_yscale("log")
        axes[2].legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_baseline_comparison(metrics_csv: str | Path, summary_json: str | Path, output_path: str | Path) -> None:
    """Plot ROC-AUC, PR-AUC, F1, precision, recall, and accuracy against baselines."""
    metrics_df = pd.read_csv(metrics_csv)
    with open(summary_json, "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    if "val_base_best_f1" in metrics_df:
        best_row = metrics_df.loc[metrics_df["val_base_best_f1"].idxmax()]
    elif "val_base_f1" in metrics_df:
        best_row = metrics_df.loc[metrics_df["val_base_f1"].idxmax()]
    else:
        best_row = metrics_df.iloc[-1]

    metric_keys = ["roc_auc", "pr_auc", "f1", "precision", "recall", "accuracy"]
    metric_labels = ["ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "Accuracy"]
    series = {
        "CNN base\n(segmentation)": _cnn_metrics(best_row, "val_base_best", metric_keys),
        "CNN window\n(aux head)": _cnn_metrics(best_row, "val_window", metric_keys),
        "Traditional\nrule": _baseline_metrics(summary.get("traditional_baseline_window", {}), metric_keys),
        "Logistic\nbaseline": _baseline_metrics(summary.get("logistic_baseline_window", {}), metric_keys),
    }

    x = np.arange(len(metric_keys))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 5))
    for offset, (name, values) in enumerate(series.items()):
        ax.bar(x + (offset - 1.5) * width, values, width=width, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("CNN and Baseline Metric Comparison")
    ax.legend(ncol=2)
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.01,
        -0.22,
        "Note: traditional and logistic baselines are window-level; the CNN base segmentation head is shown as base-level reference.",
        transform=ax.transAxes,
        fontsize=9,
        va="top",
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_roc_pr_curves(
    series: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: str | Path,
    note: str | None = None,
) -> None:
    """Plot ROC and precision-recall curves for named target/score series."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for label, (y_true, y_score) in series.items():
        y_true = np.asarray(y_true).astype(np.int32).reshape(-1)
        y_score = np.asarray(y_score).astype(np.float64).reshape(-1)
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = roc_auc_score(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score)
        axes[0].plot(fpr, tpr, lw=2, label=f"{label} (AUC={roc_auc:.3f})")
        axes[1].plot(recall, precision, lw=2, label=f"{label} (AP={pr_auc:.3f})")

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1, label="Random")
    axes[0].set_title("ROC Curve")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].set_title("Precision-Recall Curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.25)
    if note:
        fig.text(0.01, 0.01, note, fontsize=9)
    fig.tight_layout(rect=(0, 0.05 if note else 0, 1, 1))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _cnn_metrics(row: pd.Series, prefix: str, metric_keys: list[str]) -> list[float]:
    values = []
    for key in metric_keys:
        column = f"{prefix}_{key}"
        values.append(float(row[column]) if column in row and pd.notna(row[column]) else np.nan)
    return values


def _baseline_metrics(metrics: dict, metric_keys: list[str]) -> list[float]:
    return [float(metrics.get(key, np.nan)) for key in metric_keys]
