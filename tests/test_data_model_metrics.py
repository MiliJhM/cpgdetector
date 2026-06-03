from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import torch

from cpgdetector.data import CpGAnnotations, CpGWindowDataset, GenomeStore, one_hot_encode
from cpgdetector.dnabert2_baseline import (
    DNABERT2WindowDataset,
    _require_transformers,
    _training_args_kwargs,
    dnabert2_training_plan,
    model_from_pretrained_kwargs,
    resolve_model_source,
    should_load_from_config,
)
from cpgdetector.interval_metrics import evaluate_intervals
from cpgdetector.losses import multitask_loss
from cpgdetector.model import MultiTaskCpGNet
from cpgdetector.postprocess import BedInterval, probabilities_to_intervals
from cpgdetector.train import composite_monitor_score, gradnorm_multitask_loss, maybe_compile_model


def write_gz_fasta(path: Path, name: str, sequence: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(f">{name}\n")
        for i in range(0, len(sequence), 60):
            handle.write(sequence[i : i + 60] + "\n")


def make_fixture(tmp_path: Path):
    genome_dir = tmp_path / "dna"
    genome_dir.mkdir()
    sequence = "ACGT" * 200
    write_gz_fasta(genome_dir / "Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz", "21", sequence)
    table = tmp_path / "cpg.tsv"
    table.write_text(
        "bin\tchrom\tchromStart\tchromEnd\tname\tlength\tcpgNum\tgcNum\tperCpg\tperGc\tobsExp\n"
        "0\tchr21\t10\t30\tCpG: 2\t20\t2\t10\t20\t50\t0.8\n",
        encoding="utf-8",
    )
    return genome_dir, table


def test_one_hot_encode_shape():
    arr = one_hot_encode("ACGT")
    assert arr.shape == (4, 4)
    assert np.allclose(arr.sum(axis=0), 1.0)
    assert arr[0, 0] == 1
    assert arr[1, 1] == 1
    assert arr[2, 2] == 1
    assert arr[3, 3] == 1


def test_mask_and_dataset(tmp_path: Path):
    genome_dir, table = make_fixture(tmp_path)
    genome = GenomeStore(genome_dir)
    annotations = CpGAnnotations(table)
    mask = annotations.mask("chr21", 0, 40)
    assert mask.sum() == 20
    assert mask[:10].sum() == 0
    assert mask[10:30].sum() == 20

    dataset = CpGWindowDataset(
        genome=genome,
        annotations=annotations,
        chroms=["chr21"],
        window_size=20,
        stride=10,
        max_windows=4,
        seed=1,
        mode="val",
    )
    item = dataset[0]
    assert item["seq_idx"].shape == (20,)
    assert item["mask"].shape == (20,)
    assert item["fraction"].shape == (1,)


def test_dnabert2_window_dataset_uses_existing_windows(tmp_path: Path):
    genome_dir, table = make_fixture(tmp_path)
    genome = GenomeStore(genome_dir)
    annotations = CpGAnnotations(table)
    source = CpGWindowDataset(
        genome=genome,
        annotations=annotations,
        chroms=["chr21"],
        window_size=20,
        stride=10,
        max_windows=4,
        seed=1,
        mode="val",
    )
    dataset = DNABERT2WindowDataset(source)
    item = dataset[0]
    assert set(item) == {"sequence", "labels", "fraction", "chrom", "start"}
    assert len(item["sequence"]) == 20
    assert item["labels"] in {0, 1}
    assert item["chrom"] == "chr21"


def test_dnabert2_local_model_path_takes_precedence(tmp_path: Path):
    local_model = tmp_path / "DNABERT-2-117M"
    local_model.mkdir()
    source = resolve_model_source({"model_name": "remote/model", "model_path": str(local_model)})
    assert source == str(local_model)


def test_dnabert2_model_load_disables_low_cpu_mem_by_default():
    kwargs = model_from_pretrained_kwargs({})
    assert kwargs["low_cpu_mem_usage"] is False
    assert kwargs["trust_remote_code"] is True
    assert kwargs["ignore_mismatched_sizes"] is True


def test_dnabert2_local_path_uses_config_load_strategy(tmp_path: Path):
    local_model = tmp_path / "DNABERT-2-117M"
    local_model.mkdir()
    assert should_load_from_config({"load_strategy": "auto"}, str(local_model))
    assert not should_load_from_config({"load_strategy": "auto"}, "zhihan1996/DNABERT-2-117M")


def test_dnabert2_flash_attention_is_disabled_in_config_by_default():
    kwargs = model_from_pretrained_kwargs({"use_triton_flash_attention": False})
    assert "use_triton_flash_attention" not in kwargs


def test_dnabert2_training_args_use_epoch_strategy(tmp_path: Path):
    hf = _require_transformers()
    cfg = {"epochs": 7, "batch_size": 4, "eval_batch_size": 8, "fp16": False}
    kwargs = _training_args_kwargs(tmp_path / "dnabert2", cfg, 1, hf["TrainingArguments"])
    assert kwargs["num_train_epochs"] == 7.0
    assert kwargs["max_steps"] == -1
    assert kwargs["save_strategy"] == "epoch"
    assert "eval_steps" not in kwargs


def test_dnabert2_training_plan_counts_epochs():
    cfg = {"epochs": 3, "batch_size": 8, "eval_batch_size": 16, "gradient_accumulation_steps": 2}
    plan = dnabert2_training_plan(100, 20, 10, cfg)
    assert plan["steps_per_epoch"] == 7
    assert plan["effective_train_steps"] == 21
    assert plan["eval_strategy"] == "epoch"


def test_model_outputs():
    model = MultiTaskCpGNet(
        channels=[8, 16],
        kernels=[7, 3],
        dilations=[1, 2],
        dropout=0.0,
        window_hidden_channels=12,
    )
    x = torch.randn(2, 4, 64)
    out = model(x)
    assert out["base_logits"].shape == (2, 64)
    assert out["window_logits"].shape == (2, 1)
    assert set(model.loss_log_vars) == {"base", "window"}
    assert set(model.gradnorm_log_weights) == {"base", "window"}


def test_uncertainty_multitask_loss_backpropagates_weights():
    model = MultiTaskCpGNet(channels=[8], kernels=[3], dilations=[1], dropout=0.0)
    x = torch.randn(2, 4, 16)
    outputs = model(x)
    mask = torch.randint(0, 2, (2, 16)).float()
    fraction = mask.mean(dim=1, keepdim=True)
    loss, parts = multitask_loss(
        outputs,
        mask,
        fraction,
        lambda_window=0.4,
        lambda_dice=0.2,
        lambda_consistency=0.1,
        mtl_method="uncertainty",
        loss_log_vars=model.loss_log_vars,
    )
    loss.backward()
    assert "consistency" in parts
    assert "base_loss_weight" in parts
    assert model.loss_log_vars["base"].grad is not None
    assert model.loss_log_vars["window"].grad is not None


def test_gradnorm_multitask_loss_backpropagates_weights():
    model = MultiTaskCpGNet(channels=[8], kernels=[3], dilations=[1], dropout=0.0)
    x = torch.randn(2, 4, 16)
    outputs = model(x)
    mask = torch.randint(0, 2, (2, 16)).float()
    fraction = mask.mean(dim=1, keepdim=True)
    config = {
        "training": {
            "mtl_method": "gradnorm",
            "lambda_window": 0.4,
            "lambda_dice": 0.2,
            "lambda_consistency": 0.1,
            "gradnorm_alpha": 1.5,
            "gradnorm_lambda": 1.0,
        }
    }
    loss, parts = gradnorm_multitask_loss(
        outputs,
        mask,
        fraction,
        model=model,
        config=config,
        base_pos_weight=None,
        gradnorm_state={},
    )
    loss.backward()
    assert "gradnorm_loss" in parts
    assert model.gradnorm_log_weights["base"].grad is not None
    assert model.gradnorm_log_weights["window"].grad is not None


def test_maybe_compile_model_returns_model_when_disabled():
    model = MultiTaskCpGNet(channels=[8], kernels=[3], dilations=[1], dropout=0.0)
    assert maybe_compile_model(model, enabled=False) is model


def test_composite_monitor_score_uses_base_and_window_metrics():
    row = {"val_base_best_f1": 0.8, "val_window_pr_auc": 0.5}
    config = {
        "training": {
            "monitor": {
                "base_metric": "val_base_best_f1",
                "window_metric": "val_window_pr_auc",
                "base_weight": 0.7,
                "window_weight": 0.3,
            }
        }
    }
    assert np.isclose(composite_monitor_score(row, config), 0.71)


def test_postprocess_and_interval_metrics():
    probs = np.zeros(100, dtype=np.float32)
    probs[10:35] = 0.9
    intervals = probabilities_to_intervals("chr1", probs, threshold=0.5, min_length=10, max_gap=2)
    assert len(intervals) == 1
    assert intervals[0].chrom == "chr1"
    assert intervals[0].start == 10
    assert intervals[0].end == 35
    assert np.isclose(intervals[0].score, 0.9)
    stats = evaluate_intervals([BedInterval("chr1", 12, 31, 1.0)], intervals, min_iou=0.)
    assert stats.precision == 1.0
    assert stats.recall == 1.0
