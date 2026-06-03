from __future__ import annotations

import numpy as np
import torch
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


class BinaryMetricAccumulator:
    """GPU-friendly binary metric accumulator with threshold grid and histogram AUC."""

    def __init__(self, thresholds: list[float], bins: int, device: torch.device):
        self.thresholds = torch.tensor(thresholds, dtype=torch.float32, device=device)
        self.bins = int(bins)
        self.device = device
        n_thresholds = len(thresholds)
        self.tp = torch.zeros(n_thresholds, dtype=torch.float64, device=device)
        self.fp = torch.zeros(n_thresholds, dtype=torch.float64, device=device)
        self.tn = torch.zeros(n_thresholds, dtype=torch.float64, device=device)
        self.fn = torch.zeros(n_thresholds, dtype=torch.float64, device=device)
        self.pos_hist = torch.zeros(self.bins, dtype=torch.float64, device=device)
        self.neg_hist = torch.zeros(self.bins, dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, targets: torch.Tensor, scores: torch.Tensor) -> None:
        targets = targets.reshape(-1).to(self.device).bool()
        scores = scores.reshape(-1).to(self.device).float().clamp_(0.0, 1.0)
        preds = scores[:, None] >= self.thresholds[None, :]
        truth = targets[:, None]
        self.tp += (preds & truth).sum(dim=0, dtype=torch.float64)
        self.fp += (preds & ~truth).sum(dim=0, dtype=torch.float64)
        self.tn += (~preds & ~truth).sum(dim=0, dtype=torch.float64)
        self.fn += (~preds & truth).sum(dim=0, dtype=torch.float64)
        self._update_hist(targets, scores)

    @torch.no_grad()
    def _update_hist(self, targets: torch.Tensor, scores: torch.Tensor) -> None:
        bin_idx = torch.clamp((scores * (self.bins - 1)).long(), 0, self.bins - 1)
        pos_idx = bin_idx[targets]
        neg_idx = bin_idx[~targets]
        if pos_idx.numel():
            self.pos_hist += torch.bincount(pos_idx, minlength=self.bins).to(self.device, dtype=torch.float64)
        if neg_idx.numel():
            self.neg_hist += torch.bincount(neg_idx, minlength=self.bins).to(self.device, dtype=torch.float64)

    def metrics_at(self, threshold: float) -> dict[str, float]:
        idx = int(torch.argmin(torch.abs(self.thresholds - float(threshold))).item())
        return self._metrics_from_counts(self.tp[idx], self.fp[idx], self.tn[idx], self.fn[idx])

    def best_f1(self) -> tuple[float, dict[str, float]]:
        precision = self.tp / torch.clamp(self.tp + self.fp, min=1.0)
        recall = self.tp / torch.clamp(self.tp + self.fn, min=1.0)
        f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-12)
        idx = int(torch.argmax(f1).item())
        threshold = float(self.thresholds[idx].detach().cpu())
        return threshold, self._metrics_from_counts(self.tp[idx], self.fp[idx], self.tn[idx], self.fn[idx])

    def _metrics_from_counts(self, tp: torch.Tensor, fp: torch.Tensor, tn: torch.Tensor, fn: torch.Tensor) -> dict[str, float]:
        precision = tp / torch.clamp(tp + fp, min=1.0)
        recall = tp / torch.clamp(tp + fn, min=1.0)
        accuracy = (tp + tn) / torch.clamp(tp + fp + tn + fn, min=1.0)
        f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-12)
        aucs = self.approx_auc()
        return {
            "roc_auc": aucs["roc_auc"],
            "pr_auc": aucs["pr_auc"],
            "accuracy": float(accuracy.detach().cpu()),
            "precision": float(precision.detach().cpu()),
            "recall": float(recall.detach().cpu()),
            "f1": float(f1.detach().cpu()),
        }

    def approx_auc(self) -> dict[str, float]:
        pos = torch.flip(self.pos_hist, dims=[0])
        neg = torch.flip(self.neg_hist, dims=[0])
        total_pos = torch.clamp(pos.sum(), min=1.0)
        total_neg = torch.clamp(neg.sum(), min=1.0)
        tp_cum = torch.cumsum(pos, dim=0)
        fp_cum = torch.cumsum(neg, dim=0)
        tpr = tp_cum / total_pos
        fpr = fp_cum / total_neg
        precision = tp_cum / torch.clamp(tp_cum + fp_cum, min=1.0)
        recall = tpr
        fpr = torch.cat([torch.zeros(1, device=self.device, dtype=torch.float64), fpr])
        tpr = torch.cat([torch.zeros(1, device=self.device, dtype=torch.float64), tpr])
        precision = torch.cat([torch.ones(1, device=self.device, dtype=torch.float64), precision])
        recall = torch.cat([torch.zeros(1, device=self.device, dtype=torch.float64), recall])
        roc_auc = torch.trapz(tpr, fpr)
        delta_recall = recall[1:] - recall[:-1]
        pr_auc = torch.sum(precision[1:] * delta_recall)
        return {
            "roc_auc": float(roc_auc.detach().cpu()),
            "pr_auc": float(pr_auc.detach().cpu()),
        }
