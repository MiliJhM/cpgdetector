from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class BedInterval:
    chrom: str
    start: int
    end: int
    score: float


def probabilities_to_intervals(
    chrom: str,
    probabilities: np.ndarray,
    threshold: float,
    min_length: int,
    max_gap: int,
) -> list[BedInterval]:
    positive = probabilities >= float(threshold)
    intervals: list[BedInterval] = []
    start: int | None = None
    last_positive: int | None = None
    gap = 0

    for idx, is_positive in enumerate(positive):
        if is_positive:
            if start is None:
                start = idx
            last_positive = idx
            gap = 0
            continue
        if start is not None and last_positive is not None:
            gap += 1
            if gap > max_gap:
                end = last_positive + 1
                _append_interval(intervals, chrom, probabilities, start, end, min_length)
                start = None
                last_positive = None
                gap = 0

    if start is not None and last_positive is not None:
        end = last_positive + 1
        _append_interval(intervals, chrom, probabilities, start, end, min_length)
    return intervals


def _append_interval(
    intervals: list[BedInterval],
    chrom: str,
    probabilities: np.ndarray,
    start: int,
    end: int,
    min_length: int,
) -> None:
    if end <= start or end - start < min_length:
        return
    score = float(probabilities[start:end].mean())
    intervals.append(BedInterval(chrom, int(start), int(end), score))


def write_bed(intervals: list[BedInterval], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in intervals:
            handle.write(f"{item.chrom}\t{item.start}\t{item.end}\t{item.score:.6f}\n")


def write_bedgraph(chrom: str, probabilities: np.ndarray, path: str | Path, decimals: int = 2) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scale = int(10**decimals)
    rounded = np.round(probabilities * scale).astype(np.int64)
    with open(path, "w", encoding="utf-8") as handle:
        if len(rounded) == 0:
            return
        start = 0
        current = rounded[0]
        for idx in range(1, len(rounded)):
            if rounded[idx] != current:
                handle.write(f"{chrom}\t{start}\t{idx}\t{current / scale:.6f}\n")
                start = idx
                current = rounded[idx]
        handle.write(f"{chrom}\t{start}\t{len(rounded)}\t{current / scale:.6f}\n")
