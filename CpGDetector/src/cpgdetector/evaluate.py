from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import CpGAnnotations, GenomeStore
from .interval_metrics import evaluate_intervals, intervals_from_annotation
from .model import MultiTaskCpGNet
from .postprocess import probabilities_to_intervals
from .predict import predict_chromosome
from .train import load_model_state_compatible
from .utils import normalize_chrom, resolve_device, save_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run chromosome-level prediction and interval evaluation.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--chrom", action="append", default=None)
    parser.add_argument("--min-iou", type=float, default=0.10)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    device = resolve_device(config.get("device", "auto"))
    model = MultiTaskCpGNet(**config["model"])
    load_model_state_compatible(model, checkpoint["model_state"])
    model.to(device)
    genome = GenomeStore(config["data"]["genome_dir"])
    annotations = CpGAnnotations(config["data"]["cpg_table"])
    chroms = args.chrom or config["data"].get("test_chroms", ["chr22"])
    threshold = float(checkpoint.get("threshold", 0.5))
    per_chrom = {}
    for chrom in chroms:
        chrom = normalize_chrom(chrom)
        probs = predict_chromosome(
            model=model,
            genome=genome,
            chrom=chrom,
            window_size=int(config["data"]["window_size"]),
            stride=int(config["data"]["stride"]),
            batch_size=int(config.get("prediction", {}).get("batch_size", max(1024, int(config["training"]["batch_size"])))),
            device=device,
            amp=bool(config["training"].get("amp", True)),
        )
        pred = probabilities_to_intervals(
            chrom,
            probs,
            threshold=threshold,
            min_length=int(config["postprocess"]["min_length"]),
            max_gap=int(config["postprocess"]["max_gap"]),
        )
        truth = intervals_from_annotation(annotations, chrom)
        per_chrom[chrom] = evaluate_intervals(truth, pred, min_iou=args.min_iou).to_dict()
    output = Path(args.output) if args.output else Path(config["output"]["run_dir"]) / "interval_metrics.json"
    save_json({"threshold": threshold, "min_iou": args.min_iou, "chromosomes": per_chrom}, output)
    print(f"Wrote interval metrics to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
