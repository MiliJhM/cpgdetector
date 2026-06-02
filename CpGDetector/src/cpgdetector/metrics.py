from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_metric(func, y_true: np.ndarray, y_score: np.ndarray | None = None, y_pred: np.ndarray | None = None) -> float:
    try:
        if y_pred is not None:
            return float(func(y_true, y_pred, zero_division=0))
        return float(func(y_true, y_score))
    except Exception:
        return float("nan")


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = y_true.astype(np.int32).reshape(-1)
    y_score = y_score.astype(np.float64).reshape(-1)
    y_pred = (y_score >= threshold).astype(np.int32)
    return {
        "roc_auc": safe_metric(roc_auc_score, y_true, y_score),
        "pr_auc": safe_metric(average_precision_score, y_true, y_score),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": safe_metric(precision_score, y_true, y_pred=y_pred),
        "recall": safe_metric(recall_score, y_true, y_pred=y_pred),
        "f1": safe_metric(f1_score, y_true, y_pred=y_pred),
    }


def best_threshold(y_true: np.ndarray, y_score: np.ndarray, grid: list[float]) -> tuple[float, dict[str, float]]:
    best_t = float(grid[0])
    best_metrics = classification_metrics(y_true, y_score, best_t)
    for threshold in grid[1:]:
        metrics = classification_metrics(y_true, y_score, float(threshold))
        if metrics["f1"] > best_metrics["f1"]:
            best_t = float(threshold)
            best_metrics = metrics
    return best_t, best_metrics


def regression_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = y_true.reshape(-1).astype(np.float64)
    y_score = y_score.reshape(-1).astype(np.float64)
    mse = float(np.mean((y_true - y_score) ** 2))
    mae = float(np.mean(np.abs(y_true - y_score)))
    return {"mse": mse, "mae": mae}
