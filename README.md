# CpGDetector

CpGDetector implements a PyTorch pipeline for CpG island detection on GRCh38.p14 primary chromosomes. It trains a multitask 1D CNN with a base-level segmentation head and a window-level auxiliary head.

The model shares a dilated 1D convolution encoder, then uses task-specific adapters before each output head. The base head predicts one CpG island logit per base. The window head uses a 1x1 linear projection plus learned attention pooling over positions, instead of plain average pooling, so it can learn which bases in the window drive the window-level CpG island signal.

## Environment

The project is configured for the local RTX 3080 Laptop GPU and installs the CUDA 13.0 PyTorch wheel.

```powershell
conda env create -f environment.yml
conda activate cpgdetector
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

Dependencies are recorded in `requirements.txt`.

## Installation

```powershell
cd CpGDetector/src
pip install -e .
```


## Data

Expected inputs:

- `data/dna/Homo_sapiens.GRCh38.dna.chromosome.<chrom>.fa.gz`
- `data/cpgIslandExt.tsv`

The CpG table is UCSC-style with `chrom`, `chromStart`, and `chromEnd` columns. Coordinates are interpreted as 0-based half-open intervals.

## Smoke Training

```powershell
python -m cpgdetector.train --config configs/smoke.yaml
```

Main outputs:

- `runs/smoke/best_model.pt`
- `runs/smoke/metrics.csv`
- `runs/smoke/training_curves.png`
- `runs/smoke/summary.json`

## Full Training

```powershell
python -m cpgdetector.train --config configs/default.yaml
```

The default configuration uses CUDA, AMP, `pin_memory`, batch size 512, prediction batch size 2048, and chromosome-level train/validation/test splits. `num_workers` defaults to 0 on Windows to avoid duplicating cached chromosome sequences across spawned worker processes.

## One-Command Full Run

PowerShell:

```powershell
.\scripts\run_full.ps1
```

Bash:

```bash
bash scripts/run_full.sh
```

Both scripts use `configs/default.yaml`, train into `runs/default`, predict `chr19-chr22`, run interval evaluation, and generate the final report. Override defaults with environment variables in Bash or parameters in PowerShell:

```bash
RUN_DIR=runs/default_ablation PRED_CHROMS="chr22" DO_PROFILE=1 bash scripts/run_full.sh
```

```powershell
.\scripts\run_full.ps1 -RunDir runs/default_ablation -PredChroms chr22 -Profile
```

## Prediction

```powershell
python -m cpgdetector.predict --checkpoint runs/default/best_model.pt --chrom chr22
```

Default prediction output:

- `predicted_cpg_islands.bed`

The base-level signal track can be very large. Write it only when needed:

```powershell
python -m cpgdetector.predict --checkpoint runs/default/best_model.pt --chrom chr22 --out-bedgraph runs/default/predicted_cpg_signal.bedGraph
```

## Interval Evaluation

```powershell
python -m cpgdetector.evaluate --checkpoint runs/default/best_model.pt --chrom chr22
```

This computes interval precision, recall, F1, best-IoU, and boundary error against UCSC annotations.

## Performance Profiling

```powershell
python -m cpgdetector.profile --config configs/smoke.yaml --batches 30
```

The profiler reports dataset build time, DataLoader time, compute time, throughput, and peak GPU memory.

## Report

```powershell
python -m cpgdetector.report --run-dir runs/default
```

This generates `report.md` from training metrics and summary artifacts.
