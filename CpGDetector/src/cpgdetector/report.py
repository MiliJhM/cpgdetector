from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a Markdown report from a CpGDetector run.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "report.md"
    with open(run_dir / "summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    interval_metrics = None
    if (run_dir / "interval_metrics.json").exists():
        with open(run_dir / "interval_metrics.json", "r", encoding="utf-8") as handle:
            interval_metrics = json.load(handle)
    metrics = pd.read_csv(run_dir / "metrics.csv")
    best_idx = metrics["val_base_best_f1"].idxmax() if "val_base_best_f1" in metrics else metrics["val_base_f1"].idxmax()
    best = metrics.loc[best_idx]
    baseline = summary.get("traditional_baseline_window", {})
    logistic = summary.get("logistic_baseline_window", {})
    text = f"""# CpG Island Detection Report

## Objective

This project trains a multitask 1D CNN to reproduce UCSC hg38 CpG island annotations on GRCh38.p14 primary chromosomes. The main head predicts a base-level CpG island probability mask, and an auxiliary attention-pooling head predicts window-level CpG island presence strength.

## Data

- Genome FASTA directory: `{summary.get("run_dir", "")}/../../data/dna`
- CpG labels: `data/cpgIslandExt.tsv`
- Train windows: {summary.get("train_windows")}
- Validation windows: {summary.get("val_windows")}
- Device: {summary.get("device")} ({summary.get("cuda_device")})

The UCSC table uses 0-based half-open coordinates. Labels are generated as base-level masks where bases inside a CpG island interval are 1 and other bases are 0.

## Model And Training

- Shared 1D convolution encoder with residual length-preserving convolution blocks.
- Base segmentation head: `Conv1d(..., 1, kernel_size=1)` producing one logit per base.
- Window auxiliary head: task-specific 1x1 linear projection, learned attention pooling over sequence positions, and an MLP output.
- Loss: base BCE + Dice loss + weighted window BCE.
- Learning-rate scheduler: {summary.get("scheduler", "NA")}; final LR: {summary.get("final_lr", "NA")}.
- CUDA AMP is enabled when a CUDA device is available.

## Validation Results

- Best epoch: {int(summary.get("best_epoch", 0))}
- Base probability threshold selected on validation: {summary.get("base_threshold")}
- Best base-level F1: {summary.get("best_val_base_f1")}
- Base-level PR-AUC at best row: {best.get("val_base_pr_auc", "NA")}
- Window-level PR-AUC at best row: {best.get("val_window_pr_auc", "NA")}

![Baseline metric comparison](baseline_comparison.png)

The traditional and logistic baselines are window-level comparisons. The CNN base segmentation metrics are shown as base-level reference; the CNN window auxiliary head is the directly comparable neural-network window-level output.

![ROC and precision-recall curves](roc_pr_curves.png)

## Traditional Baseline

The traditional baseline scores windows using GC fraction and observed/expected CpG ratio.

- ROC-AUC: {baseline.get("roc_auc", "NA")}
- PR-AUC: {baseline.get("pr_auc", "NA")}
- F1: {baseline.get("f1", "NA")}

## Logistic Regression Baseline

The logistic regression baseline uses GC fraction, observed/expected CpG ratio, C fraction, G fraction, and CpG dinucleotide density.

- ROC-AUC: {logistic.get("roc_auc", "NA")}
- PR-AUC: {logistic.get("pr_auc", "NA")}
- F1: {logistic.get("f1", "NA")}

## Interval-Level Evaluation

{format_interval_metrics(interval_metrics)}

## Artifacts

- Model checkpoint: `{run_dir / "best_model.pt"}`
- Metrics table: `{run_dir / "metrics.csv"}`
- Training curves: `{run_dir / "training_curves.png"}`
- Baseline comparison plot: `{run_dir / "baseline_comparison.png"}`
- ROC/PR curve plot: `{run_dir / "roc_pr_curves.png"}`
- Prediction BED after running prediction: `{run_dir / "predicted_cpg_islands.bed"}`
- Optional prediction signal track: `{run_dir / "predicted_cpg_signal.bedGraph"}`

## Notes

UCSC CpG island annotations are closely related to traditional CpG island rules. The model should therefore be interpreted as learning the UCSC annotation pattern, not as discovering an independent biological definition. Reported performance should emphasize PR-AUC, F1, boundary quality, and hard-negative behavior rather than accuracy alone.
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    print(f"Wrote report to {output}")
    return 0


def format_interval_metrics(interval_metrics: dict | None) -> str:
    if not interval_metrics:
        return "Interval-level evaluation has not been run yet."
    lines = [f"- Threshold: {interval_metrics.get('threshold')}", f"- Minimum IoU for match: {interval_metrics.get('min_iou')}"]
    for chrom, stats in interval_metrics.get("chromosomes", {}).items():
        lines.extend(
            [
                f"- {chrom}: precision={stats.get('precision')}, recall={stats.get('recall')}, "
                f"F1={stats.get('f1')}, mean_best_iou={stats.get('mean_best_iou')}, "
                f"mean_boundary_error={stats.get('mean_boundary_error')}",
            ]
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
