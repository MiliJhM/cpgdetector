from __future__ import annotations

import argparse
import time

import torch

from .data import CpGAnnotations, GenomeStore
from .losses import multitask_loss
from .model import MultiTaskCpGNet
from .train import build_dataset, dataloader
from .utils import load_yaml, resolve_device, set_seed


def format_bytes(value: int) -> str:
    units = ["B", "KB", "GB", "GB"]
    value_f = float(value)
    for unit in ["B", "KB", "MB", "GB"]:
        if value_f < 1024 or unit == "GB":
            return f"{value_f:.2f} {unit}"
        value_f /= 1024
    return f"{value_f:.2f} GB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile dataloader and model throughput.")
    parser.add_argument("--config", default="configs/smoke.yaml")
    parser.add_argument("--batches", type=int, default=20)
    args = parser.parse_args(argv)

    config = load_yaml(args.config)
    set_seed(int(config["seed"]))
    device = resolve_device(config.get("device", "auto"))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.benchmark = True

    t0 = time.perf_counter()
    genome = GenomeStore(config["data"]["genome_dir"])
    annotations = CpGAnnotations(config["data"]["cpg_table"])
    dataset = build_dataset(config, "train", genome, annotations)
    build_seconds = time.perf_counter() - t0
    loader = dataloader(dataset, config, shuffle=True)

    model = MultiTaskCpGNet(**config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]))
    amp_enabled = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    seen = 0
    data_time = 0.0
    compute_time = 0.0
    iterator = iter(loader)
    for _ in range(args.batches):
        batch_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        if device.type == "cuda":
            torch.cuda.synchronize()
        data_time += time.perf_counter() - batch_start

        compute_start = time.perf_counter()
        x = batch["x"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        fraction = batch["fraction"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(x)
            loss, _ = multitask_loss(
                outputs,
                mask,
                fraction,
                lambda_window=config["training"]["lambda_window"],
                lambda_dice=config["training"]["lambda_dice"],
                base_pos_weight=None,
            )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if device.type == "cuda":
            torch.cuda.synchronize()
        compute_time += time.perf_counter() - compute_start
        seen += int(x.shape[0])

    total_time = data_time + compute_time
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Peak allocated: {format_bytes(torch.cuda.max_memory_allocated())}")
        print(f"Peak reserved: {format_bytes(torch.cuda.max_memory_reserved())}")
    print(f"Dataset build time: {build_seconds:.3f}s")
    print(f"Samples processed: {seen}")
    print(f"Data loading time: {data_time:.3f}s")
    print(f"Compute time: {compute_time:.3f}s")
    print(f"Throughput: {seen / max(total_time, 1e-9):.1f} windows/s")
    print(f"Data fraction: {data_time / max(total_time, 1e-9):.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
