from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import torch

from cpgdetector.data import CpGAnnotations, CpGWindowDataset, GenomeStore, one_hot_encode
from cpgdetector.interval_metrics import evaluate_intervals
from cpgdetector.model import MultiTaskCpGNet
from cpgdetector.postprocess import BedInterval, probabilities_to_intervals


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
    assert item["x"].shape == (4, 20)
    assert item["mask"].shape == (20,)
    assert item["fraction"].shape == (1,)


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
