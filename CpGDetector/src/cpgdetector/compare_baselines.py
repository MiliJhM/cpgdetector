from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .visualize import plot_baseline_comparison, plot_roc_pr_curves


def _strip_metric_prefix(metrics: dict, prefix: str) -> dict[str, float]:
    stripped = {}
    for key, value in metrics.items():
        if key.startswith(prefix):
            stripped[key[len(prefix) :]] = value
    return stripped


def _load_curve_scores(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = np.load(path, allow_pickle=True)
    labels = [str(label) for label in data["labels"]]
    series = {}
    for idx, label in enumerate(labels):
        series[label] = (data[f"series_{idx}_targets"], data[f"series_{idx}_scores"])
    return series


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Regenerate baseline comparison plots for a run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dnabert2-dir", default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    dnabert2_dir = Path(args.dnabert2_dir) if args.dnabert2_dir else run_dir / "dnabert2_baseline"
    dnabert2_metrics_path = dnabert2_dir / "dnabert2_metrics.json"
    extra_metrics = {}
    if dnabert2_metrics_path.exists():
        with open(dnabert2_metrics_path, "r", encoding="utf-8") as handle:
            dnabert2_metrics = json.load(handle)
        extra_metrics["DNABERT2\nwindow"] = _strip_metric_prefix(dnabert2_metrics.get("validation", {}), "val_")

    plot_baseline_comparison(
        run_dir / "metrics.csv",
        run_dir / "summary.json",
        run_dir / "baseline_comparison.png",
        extra_window_baselines=extra_metrics,
    )

    curve_scores_path = run_dir / "curve_scores.npz"
    if curve_scores_path.exists():
        curve_series = _load_curve_scores(curve_scores_path)
        dnabert2_scores_path = dnabert2_dir / "dnabert2_val_scores.npz"
        if dnabert2_scores_path.exists():
            dnabert2_scores = np.load(dnabert2_scores_path)
            curve_series["DNABERT2 window"] = (dnabert2_scores["labels"], dnabert2_scores["scores"])
        plot_roc_pr_curves(
            curve_series,
            run_dir / "roc_pr_curves.png",
            note="Traditional, logistic, CNN window, and DNABERT2 are window-level. CNN base segmentation is base-level reference.",
        )

    print(f"Regenerated baseline comparison plots in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
