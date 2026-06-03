from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from .data import CpGAnnotations, GenomeStore, valid_encoded_window
from .model import MultiTaskCpGNet
from .train import load_model_state_compatible
from .utils import normalize_chrom, resolve_device


def _strip_compile_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key.removeprefix("_orig_mod."): value for key, value in state_dict.items()}


def _model_config_for_checkpoint(checkpoint: dict) -> dict:
    model_cfg = dict(checkpoint["config"]["model"])
    params = inspect.signature(MultiTaskCpGNet.__init__).parameters
    return {key: value for key, value in model_cfg.items() if key in params}


def choose_validation_island(
    annotations: CpGAnnotations,
    genome: GenomeStore,
    val_chroms: list[str],
    flank: int,
    min_island_length: int,
) -> tuple[str, int, int]:
    candidates: list[tuple[int, str, int, int]] = []
    for chrom in val_chroms:
        chrom = normalize_chrom(chrom)
        chrom_len = genome.length(chrom)
        for start, end in annotations.intervals(chrom):
            start = int(start)
            end = int(end)
            if end - start < min_island_length:
                continue
            if start - flank < 0 or end + flank > chrom_len:
                continue
            segment = genome.encoded_window(chrom, start - flank, end + flank)
            if valid_encoded_window(segment):
                distance_to_middle = abs((start + end) // 2 - chrom_len // 2)
                candidates.append((distance_to_middle, chrom, start, end))
    if not candidates:
        raise RuntimeError("No valid CpG island with requested flanks was found in validation chromosomes")
    _, chrom, start, end = sorted(candidates)[0]
    return chrom, start, end


@torch.no_grad()
def predict_region(
    model: MultiTaskCpGNet,
    genome: GenomeStore,
    chrom: str,
    region_start: int,
    region_end: int,
    window_size: int,
    stride: int,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    chrom = normalize_chrom(chrom)
    chrom_len = genome.length(chrom)
    pred_start = max(0, region_start - window_size + stride)
    pred_end = min(chrom_len, region_end + window_size - stride)
    starts = list(range(pred_start, max(pred_start, pred_end - window_size + 1), stride))
    if not starts or starts[-1] + window_size < region_end:
        starts.append(max(0, min(chrom_len - window_size, region_end - window_size)))

    sums = np.zeros(region_end - region_start, dtype=np.float32)
    counts = np.zeros(region_end - region_start, dtype=np.float32)
    batch: list[np.ndarray] = []
    batch_starts: list[int] = []
    model.eval()

    def flush() -> None:
        if not batch:
            return
        idx = torch.from_numpy(np.stack(batch, axis=0)).long().to(device)
        x = F.one_hot(idx, num_classes=4).permute(0, 2, 1).contiguous().float()
        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            outputs = model(x)
            probs = torch.sigmoid(outputs["base_logits"]).detach().float().cpu().numpy()
        for start, prob in zip(batch_starts, probs):
            left = max(start, region_start)
            right = min(start + window_size, region_end)
            if left >= right:
                continue
            sums[left - region_start : right - region_start] += prob[left - start : right - start]
            counts[left - region_start : right - region_start] += 1.0
        batch.clear()
        batch_starts.clear()

    for start in starts:
        seq = genome.encoded_window(chrom, start, start + window_size)
        if not valid_encoded_window(seq):
            continue
        batch.append(seq.astype(np.int64, copy=True))
        batch_starts.append(start)
        if len(batch) >= batch_size:
            flush()
    flush()
    probs = np.zeros_like(sums)
    valid = counts > 0
    probs[valid] = sums[valid] / counts[valid]
    return probs


def plot_region(
    chrom: str,
    region_start: int,
    region_end: int,
    island_start: int,
    island_end: int,
    probs: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    output: Path,
) -> None:
    x = np.arange(region_start, region_end)
    local_x = x - island_start
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(15, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 0.7, 0.7], "hspace": 0.08},
    )

    axes[0].plot(local_x, probs, color="#2563eb", linewidth=1.8, label="Model CpG island probability")
    axes[0].fill_between(local_x, probs, color="#60a5fa", alpha=0.22)
    axes[0].axhline(threshold, color="#dc2626", linestyle="--", linewidth=1.1, label=f"Base threshold = {threshold:.2f}")
    axes[0].axvline(0, color="#111827", linestyle=":", linewidth=1.0)
    axes[0].axvline(island_end - island_start, color="#111827", linestyle=":", linewidth=1.0)
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_ylabel("Probability")
    axes[0].legend(loc="upper right", frameon=False)
    axes[0].set_title(
        f"{chrom}:{region_start:,}-{region_end:,} | CpG island {chrom}:{island_start:,}-{island_end:,}",
        loc="left",
        fontsize=12,
        fontweight="bold",
    )

    axes[1].fill_between(local_x, mask, step="pre", color="#16a34a", alpha=0.75)
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].set_yticks([0, 1])
    axes[1].set_ylabel("Truth")

    pred_binary = (probs >= threshold).astype(np.float32)
    axes[2].fill_between(local_x, pred_binary, step="pre", color="#f97316", alpha=0.75)
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_yticks([0, 1])
    axes[2].set_ylabel("Pred")
    axes[2].set_xlabel("Position relative to CpG island start (bp)")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.18)
        ax.axvspan(0, island_end - island_start, color="#22c55e", alpha=0.08, linewidth=0)

    fig.text(0.01, 0.01, "Green band: UCSC CpG island annotation. Blue curve: per-base model output.", fontsize=9)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = resolve_device(args.device or config.get("device", "auto"))
    genome = GenomeStore(config["data"]["genome_dir"])
    annotations = CpGAnnotations(config["data"]["cpg_table"])
    val_chroms = [normalize_chrom(chrom) for chrom in config["data"]["val_chroms"]]
    chrom, island_start, island_end = choose_validation_island(
        annotations,
        genome,
        val_chroms,
        flank=int(args.flank),
        min_island_length=int(args.min_island_length),
    )
    region_start = max(0, island_start - int(args.flank))
    region_end = min(genome.length(chrom), island_end + int(args.flank))

    model = MultiTaskCpGNet(**_model_config_for_checkpoint(checkpoint))
    checkpoint["model_state"] = _strip_compile_prefix(checkpoint["model_state"])
    load_model_state_compatible(model, checkpoint["model_state"])
    model.to(device)

    probs = predict_region(
        model=model,
        genome=genome,
        chrom=chrom,
        region_start=region_start,
        region_end=region_end,
        window_size=int(config["data"]["window_size"]),
        stride=int(config["data"]["stride"]),
        batch_size=int(args.batch_size),
        device=device,
        amp=bool(config["training"].get("amp", True)),
    )
    mask = annotations.mask_fast(chrom, region_start, region_end, genome.length(chrom)).astype(np.float32)
    threshold = float(checkpoint.get("threshold", args.threshold))
    output = Path(args.output)
    plot_region(chrom, region_start, region_end, island_start, island_end, probs, mask, threshold, output)

    score_path = output.with_suffix(".npz")
    np.savez_compressed(
        score_path,
        chrom=np.asarray(chrom),
        region_start=np.asarray(region_start),
        region_end=np.asarray(region_end),
        island_start=np.asarray(island_start),
        island_end=np.asarray(island_end),
        probabilities=probs,
        truth=mask,
        threshold=np.asarray(threshold),
    )
    print(f"Region: {chrom}:{region_start}-{region_end}")
    print(f"CpG island: {chrom}:{island_start}-{island_end}")
    print(f"Wrote figure: {output}")
    print(f"Wrote scores: {score_path}")
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize per-base CpG island probabilities on a validation boundary region.")
    parser.add_argument("--checkpoint", default="runs/remote/best_model.pt")
    parser.add_argument("--output", default="runs/remote/validation_boundary_probability.png")
    parser.add_argument("--flank", type=int, default=2000)
    parser.add_argument("--min-island-length", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
