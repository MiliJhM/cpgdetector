from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass

import numpy as np

from .postprocess import BedInterval


@dataclass(frozen=True)
class IntervalStats:
    truth_count: int
    pred_count: int
    matched_truth: int
    matched_pred: int
    precision: float
    recall: float
    f1: float
    mean_best_iou: float
    mean_boundary_error: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def intervals_from_annotation(annotations, chrom: str) -> list[BedInterval]:
    return [BedInterval(chrom, int(start), int(end), 1.0) for start, end in annotations.intervals(chrom)]


def interval_iou(a: BedInterval, b: BedInterval) -> float:
    intersection = max(0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    if union <= 0:
        return 0.0
    return intersection / union


def boundary_error(a: BedInterval, b: BedInterval) -> float:
    return (abs(a.start - b.start) + abs(a.end - b.end)) / 2.0


def evaluate_intervals(
    truth: list[BedInterval],
    pred: list[BedInterval],
    min_iou: float = 0.1,
) -> IntervalStats:
    truth_matched = set()
    pred_matched = set()
    best_ious: list[float] = []
    boundary_errors: list[float] = []
    for p_idx, p in enumerate(pred):
        if not truth:
            continue
        ious = np.array([interval_iou(p, t) for t in truth], dtype=np.float64)
        best_idx = int(np.argmax(ious))
        best_iou = float(ious[best_idx])
        best_ious.append(best_iou)
        if best_iou >= min_iou:
            pred_matched.add(p_idx)
            truth_matched.add(best_idx)
            boundary_errors.append(boundary_error(p, truth[best_idx]))
    precision = len(pred_matched) / len(pred) if pred else 0.0
    recall = len(truth_matched) / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return IntervalStats(
        truth_count=len(truth),
        pred_count=len(pred),
        matched_truth=len(truth_matched),
        matched_pred=len(pred_matched),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        mean_best_iou=float(np.mean(best_ious)) if best_ious else 0.0,
        mean_boundary_error=float(np.mean(boundary_errors)) if boundary_errors else float("nan"),
    )
