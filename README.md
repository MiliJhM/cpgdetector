# CpGDetector

CpGDetector implements a PyTorch pipeline for CpG island detection on GRCh38.p14 primary chromosomes. It trains a multitask 1D CNN with a base-level segmentation head and a window-level auxiliary head.

The model shares a dilated 1D convolution encoder, then uses task-specific adapters before each output head. The base head predicts one CpG island logit per base. The window head uses a 1x1 linear projection plus learned attention pooling over positions, instead of plain average pooling, so it can learn which bases in the window drive the window-level CpG island signal.

Training uses multitask learning by default: a base-window consistency loss aligns the window probability with the mean base probability, and uncertainty weighting learns the relative weights of the base segmentation and window auxiliary tasks. Set `training.mtl_method: fixed` to recover static loss weighting, or `training.mtl_method: gradnorm` to balance base/window tasks by their shared-encoder gradient norms.

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

The default configuration is tuned for server-side GPU training: CUDA, AMP, `pin_memory`, batch size 2048, prediction batch size 2048, `num_workers: 8`, persistent workers, and prefetching. On Windows or memory-constrained machines, reduce `training.num_workers` to 0.

## DNABERT2 Window Baseline

The DNABERT2 baseline adapts the fine-tuning setup from `WeitangSun/CpG_transformer` to this project's existing windows. It trains a window-level sequence classifier with `zhihan1996/DNABERT-2-117M`; it does not generate or reuse the external repository's CSV dataset.

```powershell
python -m cpgdetector.dnabert2_baseline --config configs/default.yaml
```

Use a locally downloaded HuggingFace model directory either through config:

```yaml
dnabert2_baseline:
  model_path: models/DNABERT-2-117M
```

or through the command line:

```powershell
python -m cpgdetector.dnabert2_baseline --config configs/default.yaml --model-path models/DNABERT-2-117M
```

Configuration lives under `dnabert2_baseline` in `configs/default.yaml`. Notes on dataset differences and expected metric impact are in `docs/dnabert2_baseline_notes.md`.

To include this baseline in the full evaluation pipeline:

```powershell
.\scripts\run_full.ps1 -RunDNABERT2 -DNABERT2ModelPath models/DNABERT-2-117M
```

```bash
RUN_DNABERT2=1 DNABERT2_MODEL_PATH=models/DNABERT-2-117M bash scripts/run_full.sh
```

Alternatively, set `dnabert2_baseline.enabled: true` in the active config. The pipeline writes DNABERT2 metrics to `runs/<run>/dnabert2_baseline/dnabert2_metrics.json` and regenerates the main baseline comparison and ROC/PR plots with the DNABERT2 validation curve included.

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
