from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import Dataset

from .data import CpGAnnotations, CpGWindowDataset, GenomeStore
from .train import build_dataset
from .utils import load_yaml, resolve_device, save_json, save_yaml, set_seed
from .visualize import plot_roc_pr_curves


DEFAULT_MODEL_NAME = "zhihan1996/DNABERT-2-117M"


def _require_transformers():
    try:
        from transformers import (  # type: ignore
            AutoConfig,
            AutoModelForSequenceClassification,
            AutoTokenizer,
            EarlyStoppingCallback,
            Trainer,
            TrainingArguments,
            set_seed as hf_set_seed,
        )
    except ImportError as exc:
        raise ImportError(
            "DNABERT2 baseline requires optional HuggingFace dependencies. "
            "Install the project requirements again, or install: transformers accelerate datasets evaluate."
        ) from exc
    return {
        "AutoConfig": AutoConfig,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "EarlyStoppingCallback": EarlyStoppingCallback,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "hf_set_seed": hf_set_seed,
    }


def _disable_flash_attention_import_on_cpu(device: torch.device) -> None:
    if device.type == "cuda":
        return
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    import builtins

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if "flash_attn_triton" in name:
            raise ImportError("Disabled flash_attn_triton for CPU compatibility")
        return original_import(name, *args, **kwargs)

    builtins.__import__ = guarded_import


@dataclass(frozen=True)
class DNABERT2WindowDataset(Dataset):
    """Window-level DNABERT2 dataset backed by the existing CpGWindowDataset."""

    source: CpGWindowDataset
    label_fraction_threshold: float = 0.0

    def __len__(self) -> int:
        return len(self.source)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        spec = self.source.specs[idx]
        end = spec.start + self.source.window_size
        chrom_len = self.source.genome.length(spec.chrom)
        sequence = self.source.genome.subseq(spec.chrom, spec.start, end)
        fraction = float(self.source.annotations.mask_fast(spec.chrom, spec.start, end, chrom_len).mean())
        return {
            "sequence": sequence,
            "labels": int(fraction > self.label_fraction_threshold),
            "fraction": fraction,
            "chrom": spec.chrom,
            "start": int(spec.start),
        }


class DnaSequenceCollator:
    def __init__(self, tokenizer, max_length: int, pad_to_max_length: bool = True):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.padding = "max_length" if pad_to_max_length else "longest"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        sequences = [item["sequence"] for item in features]
        batch = self.tokenizer(
            sequences,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor([int(item["labels"]) for item in features], dtype=torch.long)
        return batch


def compute_metrics(eval_pred) -> dict[str, float]:
    predictions_raw, labels = eval_pred
    logits = predictions_raw[0] if isinstance(predictions_raw, (tuple, list)) else predictions_raw
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    if logits.ndim == 1 or logits.shape[-1] == 1:
        flat_logits = logits.reshape(-1)
        scores = 1.0 / (1.0 + np.exp(-flat_logits))
        predictions = (scores >= 0.5).astype(np.int64)
    else:
        shifted = logits - logits.max(axis=-1, keepdims=True)
        probs = np.exp(shifted) / np.exp(shifted).sum(axis=-1, keepdims=True)
        scores = probs[:, 1]
        predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    result = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": float(matthews_corrcoef(labels, predictions)) if len(np.unique(labels)) > 1 else 0.0,
    }
    if len(np.unique(labels)) > 1:
        result["roc_auc"] = float(roc_auc_score(labels, scores))
        result["pr_auc"] = float(average_precision_score(labels, scores))
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        result.update({"tp": float(tp), "tn": float(tn), "fp": float(fp), "fn": float(fn)})
        result["specificity"] = float(tn / (tn + fp)) if (tn + fp) else 0.0
    else:
        result.update({"roc_auc": float("nan"), "pr_auc": float("nan"), "specificity": 0.0})
    return result


def logits_to_positive_scores(predictions_raw) -> np.ndarray:
    logits = predictions_raw[0] if isinstance(predictions_raw, (tuple, list)) else predictions_raw
    logits = np.asarray(logits)
    if logits.ndim == 1 or logits.shape[-1] == 1:
        flat_logits = logits.reshape(-1)
        return 1.0 / (1.0 + np.exp(-flat_logits))
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_logits = np.exp(shifted)
    probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    return probs[:, 1]


def save_prediction_scores(trainer, dataset: Dataset, output_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    prediction = trainer.predict(dataset)
    labels = np.asarray(prediction.label_ids).astype(np.int32).reshape(-1)
    scores = logits_to_positive_scores(prediction.predictions).astype(np.float64).reshape(-1)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, labels=labels, scores=scores)
    return labels, scores


def _training_args_kwargs(output_dir: Path, cfg: dict[str, Any], seed: int, training_args_cls) -> dict[str, Any]:
    params = inspect.signature(training_args_cls.__init__).parameters
    strategy_key = "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"
    kwargs = {
        "output_dir": str(output_dir),
        "overwrite_output_dir": True,
        "num_train_epochs": float(cfg.get("epochs", 3)),
        "per_device_train_batch_size": int(cfg.get("batch_size", 8)),
        "per_device_eval_batch_size": int(cfg.get("eval_batch_size", 16)),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(cfg.get("lr", 3e-5)),
        "weight_decay": float(cfg.get("weight_decay", 0.01)),
        "warmup_steps": int(cfg.get("warmup_steps", 50)),
        "max_grad_norm": float(cfg.get("max_grad_norm", 1.0)),
        strategy_key: "steps",
        "eval_steps": int(cfg.get("eval_steps", 500)),
        "metric_for_best_model": str(cfg.get("metric_for_best_model", "f1")),
        "greater_is_better": True,
        "load_best_model_at_end": True,
        "save_strategy": "steps",
        "save_steps": int(cfg.get("save_steps", 500)),
        "save_total_limit": int(cfg.get("save_total_limit", 3)),
        "logging_strategy": "steps",
        "logging_steps": int(cfg.get("logging_steps", 100)),
        "report_to": str(cfg.get("report_to", "none")),
        "fp16": bool(cfg.get("fp16", True)) and torch.cuda.is_available(),
        "seed": int(seed),
        "data_seed": int(seed),
        "dataloader_num_workers": int(cfg.get("num_workers", 0)),
        "remove_unused_columns": False,
        "do_train": True,
        "do_eval": True,
    }
    if "overwrite_output_dir" in params:
        kwargs["overwrite_output_dir"] = True
    if "dataloader_pin_memory" in params:
        kwargs["dataloader_pin_memory"] = bool(cfg.get("pin_memory", torch.cuda.is_available()))
    return {key: value for key, value in kwargs.items() if key in params}


def _config_for_split(config: dict[str, Any], baseline_cfg: dict[str, Any], split: str) -> dict[str, Any]:
    split_cfg = dict(config)
    split_cfg["data"] = dict(config["data"])
    override_key = f"max_{split}_windows"
    if baseline_cfg.get(override_key) is not None:
        split_cfg["data"][override_key] = int(baseline_cfg[override_key])
    return split_cfg


def run(config_path: str | Path, run_dir_override: str | Path | None = None) -> int:
    config = load_yaml(config_path)
    baseline_cfg = dict(config.get("dnabert2_baseline", {}))
    seed = int(config.get("seed", 42))
    set_seed(seed)

    output_dir = Path(run_dir_override or baseline_cfg.get("output_dir", Path(config["output"]["run_dir"]) / "dnabert2_baseline"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, output_dir / "config.yaml")

    device = resolve_device(config.get("device", "auto"))
    _disable_flash_attention_import_on_cpu(device)
    hf = _require_transformers()
    hf["hf_set_seed"](seed)

    genome = GenomeStore(config["data"]["genome_dir"])
    annotations = CpGAnnotations(config["data"]["cpg_table"])
    train_ds = DNABERT2WindowDataset(
        build_dataset(_config_for_split(config, baseline_cfg, "train"), "train", genome, annotations),
        label_fraction_threshold=float(baseline_cfg.get("label_fraction_threshold", 0.0)),
    )
    val_ds = DNABERT2WindowDataset(
        build_dataset(_config_for_split(config, baseline_cfg, "val"), "val", genome, annotations),
        label_fraction_threshold=float(baseline_cfg.get("label_fraction_threshold", 0.0)),
    )
    test_ds = DNABERT2WindowDataset(
        build_dataset(_config_for_split(config, baseline_cfg, "test"), "test", genome, annotations),
        label_fraction_threshold=float(baseline_cfg.get("label_fraction_threshold", 0.0)),
    )
    print(f"DNABERT2 baseline datasets: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    model_name = str(baseline_cfg.get("model_name", DEFAULT_MODEL_NAME))
    tokenizer = hf["AutoTokenizer"].from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"

    model_config = hf["AutoConfig"].from_pretrained(model_name, trust_remote_code=True)
    model_config.num_labels = 2
    model_config.problem_type = "single_label_classification"
    model_config.pad_token_id = tokenizer.pad_token_id
    model = hf["AutoModelForSequenceClassification"].from_pretrained(
        model_name,
        config=model_config,
        trust_remote_code=True,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    training_args = hf["TrainingArguments"](
        **_training_args_kwargs(output_dir, baseline_cfg, seed, hf["TrainingArguments"])
    )
    collator = DnaSequenceCollator(
        tokenizer,
        max_length=int(baseline_cfg.get("model_max_length", 128)),
        pad_to_max_length=bool(baseline_cfg.get("pad_to_max_length", True)),
    )
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "data_collator": collator,
        "compute_metrics": compute_metrics,
        "callbacks": [
            hf["EarlyStoppingCallback"](
                early_stopping_patience=int(baseline_cfg.get("early_stopping_patience", 3))
            )
        ],
    }
    trainer_params = inspect.signature(hf["Trainer"].__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = hf["Trainer"](**trainer_kwargs)
    trainer.train()

    final_dir = output_dir / "final_model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    val_metrics = trainer.evaluate(val_ds, metric_key_prefix="val")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
    val_labels, val_scores = save_prediction_scores(trainer, val_ds, output_dir / "dnabert2_val_scores.npz")
    test_labels, test_scores = save_prediction_scores(trainer, test_ds, output_dir / "dnabert2_test_scores.npz")
    plot_roc_pr_curves(
        {
            "DNABERT2 validation": (val_labels, val_scores),
            "DNABERT2 test": (test_labels, test_scores),
        },
        output_dir / "dnabert2_roc_pr_curves.png",
        note="DNABERT2 baseline is window-level classification over CpGDetector windows.",
    )
    metrics = {
        "model_name": model_name,
        "output_dir": str(output_dir),
        "device": str(device),
        "train_windows": len(train_ds),
        "val_windows": len(val_ds),
        "test_windows": len(test_ds),
        "validation": val_metrics,
        "test": test_metrics,
        "validation_scores": str(output_dir / "dnabert2_val_scores.npz"),
        "test_scores": str(output_dir / "dnabert2_test_scores.npz"),
    }
    save_json(metrics, output_dir / "dnabert2_metrics.json")
    with open(output_dir / "dnabert2_metrics.txt", "w", encoding="utf-8") as handle:
        for section in ["validation", "test"]:
            handle.write(f"[{section}]\n")
            for key, value in sorted(metrics[section].items()):
                handle.write(f"{key}: {value}\n")
            handle.write("\n")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fine-tune DNABERT2 as a window-level CpG baseline.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    try:
        return run(args.config, args.output_dir)
    except ImportError as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
