from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_yaml(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(obj, handle, sort_keys=False, allow_unicode=True)


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def normalize_chrom(chrom: str) -> str:
    chrom = str(chrom).strip()
    if chrom.lower().startswith("chr"):
        suffix = chrom[3:]
    else:
        suffix = chrom
    if suffix.upper() == "MT":
        suffix = "M"
    return f"chr{suffix}"


def ensembl_chrom_name(chrom: str) -> str:
    chrom = normalize_chrom(chrom)
    suffix = chrom[3:]
    if suffix == "M":
        return "MT"
    return suffix


def project_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()
