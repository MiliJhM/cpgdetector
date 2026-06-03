from __future__ import annotations

import argparse
import gzip
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .utils import ensembl_chrom_name, normalize_chrom


BASE_TO_INDEX = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "a": 0,
    "c": 1,
    "g": 2,
    "t": 3,
}
_ASCII_TO_INDEX = np.full(256, -1, dtype=np.int16)
for _base, _idx in BASE_TO_INDEX.items():
    _ASCII_TO_INDEX[ord(_base)] = _idx


def read_fasta(path: str | Path) -> str:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    chunks: list[str] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                continue
            chunks.append(line.strip())
    return "".join(chunks).upper()


def one_hot_encode(seq: str) -> np.ndarray:
    encoded = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    indices = _ASCII_TO_INDEX[encoded]
    if np.any(indices < 0):
        raise ValueError("Sequence contains non-ACGT bases")
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    arr[indices, np.arange(len(seq))] = 1.0
    return arr


def gc_fraction(seq: str) -> float:
    if not seq:
        return 0.0
    gc = seq.count("G") + seq.count("C")
    return gc / len(seq)


@dataclass(frozen=True)
class WindowSpec:
    chrom: str
    start: int


class GenomeStore:
    """Lazy FASTA reader keyed by UCSC-style chromosome names."""

    def __init__(self, genome_dir: str | Path):
        self.genome_dir = Path(genome_dir)
        self._cache: dict[str, str] = {}

    def find_fasta(self, chrom: str) -> Path:
        chrom = normalize_chrom(chrom)
        ensembl_name = ensembl_chrom_name(chrom)
        candidates = [
            self.genome_dir / f"Homo_sapiens.GRCh38.dna.chromosome.{ensembl_name}.fa.gz",
            self.genome_dir / f"Homo_sapiens.GRCh38.dna.chromosome.{ensembl_name}.fa",
            self.genome_dir / f"{chrom}.fa.gz",
            self.genome_dir / f"{chrom}.fa",
            self.genome_dir / f"{ensembl_name}.fa.gz",
            self.genome_dir / f"{ensembl_name}.fa",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"No FASTA found for {chrom} in {self.genome_dir}")

    def load(self, chrom: str) -> str:
        chrom = normalize_chrom(chrom)
        if chrom not in self._cache:
            self._cache[chrom] = read_fasta(self.find_fasta(chrom))
        return self._cache[chrom]

    def length(self, chrom: str) -> int:
        return len(self.load(chrom))

    def subseq(self, chrom: str, start: int, end: int) -> str:
        return self.load(chrom)[start:end]


class CpGAnnotations:
    def __init__(self, table_path: str | Path):
        self.table_path = Path(table_path)
        self.by_chrom: dict[str, np.ndarray] = {}
        self._load()

    def _load(self) -> None:
        columns = [
            "bin",
            "chrom",
            "chromStart",
            "chromEnd",
            "name",
            "length",
            "cpgNum",
            "gcNum",
            "perCpg",
            "perGc",
            "obsExp",
        ]
        required = {"chrom", "chromStart", "chromEnd"}
        df = pd.read_csv(self.table_path, sep="\t")
        if required.difference(df.columns):
            df = pd.read_csv(self.table_path, sep="\t", names=columns, skiprows=1)
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"CpG table missing columns: {sorted(missing)}")
        df = df.copy()
        df["chrom"] = df["chrom"].map(normalize_chrom)
        for chrom, sub in df.groupby("chrom", sort=False):
            arr = sub[["chromStart", "chromEnd"]].to_numpy(dtype=np.int64)
            order = np.argsort(arr[:, 0])
            self.by_chrom[chrom] = arr[order]

    def intervals(self, chrom: str) -> np.ndarray:
        return self.by_chrom.get(normalize_chrom(chrom), np.zeros((0, 2), dtype=np.int64))

    def mask(self, chrom: str, start: int, end: int) -> np.ndarray:
        intervals = self.intervals(chrom)
        mask = np.zeros(end - start, dtype=np.float32)
        if len(intervals) == 0:
            return mask
        idx = np.searchsorted(intervals[:, 1], start, side="right")
        for iv_start, iv_end in intervals[idx:]:
            if iv_start >= end:
                break
            left = max(start, int(iv_start))
            right = min(end, int(iv_end))
            if left < right:
                mask[left - start : right - start] = 1.0
        return mask

    def overlap_fraction(self, chrom: str, start: int, end: int) -> float:
        return float(self.mask(chrom, start, end).mean())


def valid_acgt(seq: str) -> bool:
    encoded = np.frombuffer(seq.encode("ascii"), dtype=np.uint8)
    return bool(np.all(_ASCII_TO_INDEX[encoded] >= 0))


class CpGWindowDataset(Dataset):
    """Sampled windows with base-level masks and window-level fractions."""

    def __init__(
        self,
        genome: GenomeStore,
        annotations: CpGAnnotations,
        chroms: Iterable[str],
        window_size: int,
        stride: int,
        max_windows: int | None,
        seed: int,
        mode: str,
        positive_fraction: float = 0.45,
        hard_negative_fraction: float = 0.25,
        min_gc_for_hard_negative: float = 0.50,
        boundary_flank: int = 2000,
    ):
        self.genome = genome
        self.annotations = annotations
        self.chroms = [normalize_chrom(c) for c in chroms]
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.max_windows = max_windows if max_windows is None else int(max_windows)
        self.seed = int(seed)
        self.mode = mode
        self.positive_fraction = float(positive_fraction)
        self.hard_negative_fraction = float(hard_negative_fraction)
        self.min_gc_for_hard_negative = float(min_gc_for_hard_negative)
        self.boundary_flank = int(boundary_flank)
        self.specs = self._build_specs()

    def _build_specs(self) -> list[WindowSpec]:
        if self.mode == "train":
            return self._sample_training_specs()
        return self._sliding_specs()

    def _sample_training_specs(self) -> list[WindowSpec]:
        rng = random.Random(self.seed)
        total = self.max_windows or 10000
        n_pos = int(total * self.positive_fraction)
        n_hard = int(total * self.hard_negative_fraction)
        n_random = total - n_pos - n_hard
        specs: list[WindowSpec] = []
        specs.extend(self._sample_positive(rng, n_pos))
        specs.extend(self._sample_hard_negative(rng, n_hard))
        specs.extend(self._sample_random_negative(rng, n_random))
        rng.shuffle(specs)
        return specs

    def _chrom_lengths(self) -> dict[str, int]:
        return {chrom: self.genome.length(chrom) for chrom in self.chroms}

    def _sample_positive(self, rng: random.Random, n: int) -> list[WindowSpec]:
        islands: list[tuple[str, int, int]] = []
        for chrom in self.chroms:
            for start, end in self.annotations.intervals(chrom):
                islands.append((chrom, int(start), int(end)))
        if not islands:
            return []
        chrom_lengths = self._chrom_lengths()
        specs: list[WindowSpec] = []
        attempts = 0
        while len(specs) < n and attempts < n * 100:
            attempts += 1
            chrom, iv_start, iv_end = rng.choice(islands)
            chrom_len = chrom_lengths[chrom]
            low = max(0, iv_start - self.window_size + 1)
            high = min(iv_end - 1, chrom_len - self.window_size)
            if high < low:
                continue
            start = rng.randint(low, high)
            seq = self.genome.subseq(chrom, start, start + self.window_size)
            if valid_acgt(seq) and self.annotations.overlap_fraction(chrom, start, start + self.window_size) > 0:
                specs.append(WindowSpec(chrom, start))
        return specs

    def _sample_hard_negative(self, rng: random.Random, n: int) -> list[WindowSpec]:
        islands: list[tuple[str, int, int]] = []
        for chrom in self.chroms:
            for start, end in self.annotations.intervals(chrom):
                islands.append((chrom, int(start), int(end)))
        if not islands:
            return self._sample_random_negative(rng, n)
        chrom_lengths = self._chrom_lengths()
        specs: list[WindowSpec] = []
        attempts = 0
        while len(specs) < n and attempts < n * 300:
            attempts += 1
            chrom, iv_start, iv_end = rng.choice(islands)
            chrom_len = chrom_lengths[chrom]
            boundary = iv_start if rng.random() < 0.5 else iv_end
            start = boundary + rng.randint(-self.boundary_flank, self.boundary_flank) - self.window_size // 2
            start = max(0, min(start, chrom_len - self.window_size))
            end = start + self.window_size
            seq = self.genome.subseq(chrom, start, end)
            if not valid_acgt(seq):
                continue
            if self.annotations.overlap_fraction(chrom, start, end) != 0:
                continue
            if gc_fraction(seq) >= self.min_gc_for_hard_negative:
                specs.append(WindowSpec(chrom, start))
        if len(specs) < n:
            specs.extend(self._sample_random_negative(rng, n - len(specs)))
        return specs

    def _sample_random_negative(self, rng: random.Random, n: int) -> list[WindowSpec]:
        chrom_lengths = self._chrom_lengths()
        specs: list[WindowSpec] = []
        attempts = 0
        while len(specs) < n and attempts < n * 200:
            attempts += 1
            chrom = rng.choice(self.chroms)
            chrom_len = chrom_lengths[chrom]
            if chrom_len <= self.window_size:
                continue
            start = rng.randint(0, chrom_len - self.window_size)
            end = start + self.window_size
            seq = self.genome.subseq(chrom, start, end)
            if valid_acgt(seq) and self.annotations.overlap_fraction(chrom, start, end) == 0:
                specs.append(WindowSpec(chrom, start))
        return specs

    def _sliding_specs(self) -> list[WindowSpec]:
        rng = random.Random(self.seed)
        specs: list[WindowSpec] = []
        seen = 0
        for chrom in self.chroms:
            chrom_len = self.genome.length(chrom)
            for start in range(0, max(0, chrom_len - self.window_size + 1), self.stride):
                seq = self.genome.subseq(chrom, start, start + self.window_size)
                if not valid_acgt(seq):
                    continue
                seen += 1
                spec = WindowSpec(chrom, start)
                if self.max_windows is None:
                    specs.append(spec)
                elif len(specs) < self.max_windows:
                    specs.append(spec)
                else:
                    j = rng.randint(0, seen - 1)
                    if j < self.max_windows:
                        specs[j] = spec
        specs.sort(key=lambda item: (item.chrom, item.start))
        return specs

    def __len__(self) -> int:
        return len(self.specs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        spec = self.specs[idx]
        end = spec.start + self.window_size
        seq = self.genome.subseq(spec.chrom, spec.start, end)
        x = one_hot_encode(seq)
        mask = self.annotations.mask(spec.chrom, spec.start, end)
        return {
            "x": torch.from_numpy(x),
            "mask": torch.from_numpy(mask),
            "fraction": torch.tensor([float(mask.mean())], dtype=torch.float32),
            "has_cpg": torch.tensor([float(mask.mean() > 0)], dtype=torch.float32),
            "chrom": spec.chrom,
            "start": spec.start,
        }


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Test CpGDetector data loading")
    parser.add_argument("--genome-dir", type=str, required=True, help="Directory containing chromosome FASTA files")
    parser.add_argument("--cpg-table", type=str, required=True, help="Tab-delimited file with CpG island annotations (UCSC-style)")
    parser.add_argument("--chrom", type=str, default="chr22", help="Chromosome to test (UCSC-style)")
    args = parser.parse_args()
    
    print('[INFO] Testing data loading...')

    genome = GenomeStore(args.genome_dir)
    annotations = CpGAnnotations(args.cpg_table)
    chrom = normalize_chrom(args.chrom)
    seq = genome.load(chrom)
    print(f'[INFO] Loaded chromosome {chrom} with length {len(seq)}')
    intervals = annotations.intervals(chrom)
    print(f'[INFO] Found {len(intervals)} CpG island intervals on {chrom}')
    if len(intervals) > 0:
        print(f'[INFO] First 5 intervals: {intervals[:5]}')
    else:
        print('[INFO] No CpG island intervals found on this chromosome.')

    print("Visualizing first 10 borders of CpG islands (each 100 bases):")
    for i, (start, end) in enumerate(intervals[:10]):
        left_flank = genome.subseq(chrom, max(0, start - 100), start)
        island = genome.subseq(chrom, start, end)
        right_flank = genome.subseq(chrom, end, min(len(seq), end + 100))
        print(f'Border {i+1}:')
        print(f'  Left flank:  {left_flank[-50:]}')
        print(f'  Island:     {island[:50]}...{island[-50:]} (length {len(island)})')
        print(f'  Right flank: {right_flank[:50]}')

    print('[INFO] Trying to instantiate CpGWindowDataset...')
    dataset = CpGWindowDataset(
        genome=genome,
        annotations=annotations,
        chroms=[chrom],
        window_size=1000,
        stride=500,
        max_windows=10,
        seed=42,
        mode='train',
        positive_fraction=0.5,
        hard_negative_fraction=0.3,
        min_gc_for_hard_negative=0.5,
        boundary_flank=2000,
    )
    print(f'[INFO] Created dataset with {len(dataset)} windows.')
    for i in range(len(dataset)):
        item = dataset[i]
        print(f'Window {i+1}: chrom={item["chrom"]}, start={item["start"]}, fraction={item["fraction"].item():.3f}, has_cpg={bool(item["has_cpg"].item())}')

    print('[INFO] Data loading test completed successfully.')