from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import gc_fraction
from .metrics import classification_metrics


def cpg_observed_expected(seq: str) -> float:
    c = seq.count("C")
    g = seq.count("G")
    cpg = sum(1 for i in range(len(seq) - 1) if seq[i : i + 2] == "CG")
    if c == 0 or g == 0:
        return 0.0
    return cpg * len(seq) / (c * g)


def traditional_window_score(seq: str) -> float:
    gc = gc_fraction(seq)
    oe = cpg_observed_expected(seq)
    if gc >= 0.50 and oe >= 0.60:
        return min(1.0, 0.5 * gc / 0.50 + 0.5 * oe / 0.60)
    return 0.5 * gc / 0.50 * min(1.0, oe / 0.60)


def window_features(seq: str) -> list[float]:
    c = seq.count("C")
    g = seq.count("G")
    cpg = sum(1 for i in range(len(seq) - 1) if seq[i : i + 2] == "CG")
    gc = gc_fraction(seq)
    oe = cpg_observed_expected(seq)
    return [
        gc,
        oe,
        c / len(seq),
        g / len(seq),
        cpg / max(len(seq) - 1, 1),
    ]


def evaluate_traditional_baseline(dataset, threshold: float = 0.5, max_items: int | None = None) -> dict[str, float]:
    y_true, y_score = traditional_baseline_scores(dataset, max_items=max_items)
    return classification_metrics(y_true, y_score, threshold)


def traditional_baseline_scores(dataset, max_items: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[float] = []
    y_score: list[float] = []
    limit = len(dataset) if max_items is None else min(max_items, len(dataset))
    for idx in range(limit):
        item = dataset[idx]
        chrom = item["chrom"]
        start = int(item["start"])
        seq = dataset.genome.subseq(chrom, start, start + dataset.window_size)
        y_true.append(float(item["has_cpg"].item()))
        y_score.append(traditional_window_score(seq))
    return np.asarray(y_true, dtype=np.int32), np.asarray(y_score, dtype=np.float64)


def _features_and_labels(dataset, max_items: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    x_rows: list[list[float]] = []
    y: list[float] = []
    limit = len(dataset) if max_items is None else min(max_items, len(dataset))
    for idx in range(limit):
        item = dataset[idx]
        chrom = item["chrom"]
        start = int(item["start"])
        seq = dataset.genome.subseq(chrom, start, start + dataset.window_size)
        x_rows.append(window_features(seq))
        y.append(float(item["has_cpg"].item()))
    return np.asarray(x_rows, dtype=np.float64), np.asarray(y, dtype=np.int32)


def evaluate_logistic_baseline(train_dataset, val_dataset, max_train: int = 10000, max_val: int = 5000) -> dict[str, float]:
    y_val, scores = logistic_baseline_scores(train_dataset, val_dataset, max_train=max_train, max_val=max_val)
    if len(np.unique(y_val)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    return classification_metrics(y_val, scores, threshold=0.5)


def logistic_baseline_scores(
    train_dataset,
    val_dataset,
    max_train: int = 10000,
    max_val: int = 5000,
) -> tuple[np.ndarray, np.ndarray]:
    x_train, y_train = _features_and_labels(train_dataset, max_train)
    x_val, y_val = _features_and_labels(val_dataset, max_val)
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return y_val, np.zeros_like(y_val, dtype=np.float64)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(x_train, y_train)
    scores = model.predict_proba(x_val)[:, 1]
    return y_val, scores
