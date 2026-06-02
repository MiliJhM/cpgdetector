from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .data import CpGAnnotations, GenomeStore, one_hot_encode, valid_acgt
from .model import MultiTaskCpGNet
from .postprocess import probabilities_to_intervals, write_bed, write_bedgraph
from .utils import load_yaml, normalize_chrom, resolve_device


@torch.no_grad()
def predict_chromosome(
    model: MultiTaskCpGNet,
    genome: GenomeStore,
    chrom: str,
    window_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    chrom = normalize_chrom(chrom)
    seq = genome.load(chrom)
    sums = np.zeros(len(seq), dtype=np.float32)
    counts = np.zeros(len(seq), dtype=np.float32)
    batch: list[np.ndarray] = []
    starts: list[int] = []
    model.eval()

    def flush() -> None:
        if not batch:
            return
        x = torch.from_numpy(np.stack(batch, axis=0)).to(device)
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(x)
            probs = torch.sigmoid(outputs["base_logits"]).detach().cpu().numpy()
        for start, prob in zip(starts, probs):
            end = start + window_size
            sums[start:end] += prob
            counts[start:end] += 1.0
        batch.clear()
        starts.clear()

    for start in tqdm(range(0, max(0, len(seq) - window_size + 1), stride), desc=f"predict {chrom}", ascii=True):
        sub = seq[start : start + window_size]
        if not valid_acgt(sub):
            continue
        batch.append(one_hot_encode(sub))
        starts.append(start)
        if len(batch) >= batch_size:
            flush()
    flush()
    probs = np.zeros_like(sums)
    valid = counts > 0
    probs[valid] = sums[valid] / counts[valid]
    return probs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict CpG island intervals from a trained model.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--chrom", action="append", default=None, help="UCSC-style chromosome, can be repeated.")
    parser.add_argument("--out-bed", default=None)
    parser.add_argument("--out-bedgraph", default=None)
    args = parser.parse_args(argv)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint["config"]
    if args.config:
        config.update(load_yaml(args.config))
    device = resolve_device(config.get("device", "auto"))
    model = MultiTaskCpGNet(**config["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    genome = GenomeStore(config["data"]["genome_dir"])
    chroms = args.chrom or config["data"].get("test_chroms", ["chr22"])
    threshold = float(checkpoint.get("threshold", 0.5))
    run_dir = Path(config["output"]["run_dir"])
    bed_path = Path(args.out_bed) if args.out_bed else run_dir / "predicted_cpg_islands.bed"
    write_signal = bool(args.out_bedgraph) or bool(config.get("prediction", {}).get("write_bedgraph", False))
    bedgraph_path = Path(args.out_bedgraph) if args.out_bedgraph else run_dir / "predicted_cpg_signal.bedGraph"
    all_intervals = []
    first_bedgraph = True
    for chrom in chroms:
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
        intervals = probabilities_to_intervals(
            normalize_chrom(chrom),
            probs,
            threshold=threshold,
            min_length=int(config["postprocess"]["min_length"]),
            max_gap=int(config["postprocess"]["max_gap"]),
        )
        all_intervals.extend(intervals)
        if write_signal:
            mode = "w" if first_bedgraph else "a"
            bedgraph_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = bedgraph_path.with_suffix(f".{normalize_chrom(chrom)}.tmp")
            write_bedgraph(normalize_chrom(chrom), probs, temp_path)
            with open(temp_path, "r", encoding="utf-8") as src, open(bedgraph_path, mode, encoding="utf-8") as dst:
                dst.write(src.read())
            temp_path.unlink()
            first_bedgraph = False
    write_bed(all_intervals, bed_path)
    print(f"Wrote {len(all_intervals)} intervals to {bed_path}")
    if write_signal:
        print(f"Wrote signal track to {bedgraph_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
