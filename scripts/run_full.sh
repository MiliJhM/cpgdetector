#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-cpgdetector}"
CONFIG="${CONFIG:-configs/default.yaml}"
RUN_DIR="${RUN_DIR:-runs/default}"
PRED_CHROMS="${PRED_CHROMS:-chr19 chr20 chr21 chr22}"
RUN_DNABERT2="${RUN_DNABERT2:-0}"
DO_PROFILE="${DO_PROFILE:-0}"
PROFILE_BATCHES="${PROFILE_BATCHES:-30}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

mkdir -p "$RUN_DIR"
LOG_FILE="$RUN_DIR/full_run.log"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

run_py() {
  conda run -n "$ENV_NAME" python "$@" 2>&1 | tee -a "$LOG_FILE"
}

log "Starting full CpGDetector run"
log "Project root: $ROOT_DIR"
log "Conda env: $ENV_NAME"
log "Config: $CONFIG"
log "Run dir: $RUN_DIR"
log "Prediction chromosomes: $PRED_CHROMS"
log "Run DNABERT2 baseline env flag: $RUN_DNABERT2"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found on PATH" >&2
  exit 1
fi

if ! conda env list | grep -qE "(^|[[:space:]])${ENV_NAME}([[:space:]]|$)"; then
  log "Conda environment '$ENV_NAME' not found; creating it from environment.yml"
  conda env create -f environment.yml 2>&1 | tee -a "$LOG_FILE"
fi

for required in "$CONFIG" "data/cpgIslandExt.tsv" "data/dna/Homo_sapiens.GRCh38.dna.chromosome.19.fa.gz"; do
  if [[ ! -e "$required" ]]; then
    echo "Required file is missing: $required" >&2
    exit 1
  fi
done

CONFIG_DNABERT2="$(conda run -n "$ENV_NAME" python -c "from cpgdetector.utils import load_yaml; import sys; cfg=load_yaml(sys.argv[1]); print('1' if cfg.get('dnabert2_baseline', {}).get('enabled', False) else '0')" "$CONFIG" | tail -n 1)"
if [[ "$RUN_DNABERT2" == "1" || "$CONFIG_DNABERT2" == "1" ]]; then
  SHOULD_RUN_DNABERT2="1"
else
  SHOULD_RUN_DNABERT2="0"
fi
log "Run DNABERT2 baseline effective: $SHOULD_RUN_DNABERT2"

log "Checking Python, PyTorch, and CUDA"
run_py -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('cuda', torch.version.cuda); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

if [[ "$DO_PROFILE" == "1" ]]; then
  log "Running throughput profile"
  run_py -m cpgdetector.profile --config "$CONFIG" --batches "$PROFILE_BATCHES"
fi

log "Training model"
run_py -m cpgdetector.train --config "$CONFIG" --run-dir "$RUN_DIR"

CHECKPOINT="$RUN_DIR/best_model.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Training did not produce checkpoint: $CHECKPOINT" >&2
  exit 1
fi

log "Predicting CpG island BED intervals"
PRED_ARGS=()
for chrom in $PRED_CHROMS; do
  PRED_ARGS+=(--chrom "$chrom")
done
run_py -m cpgdetector.predict --checkpoint "$CHECKPOINT" "${PRED_ARGS[@]}" --out-bed "$RUN_DIR/predicted_cpg_islands.bed"

log "Evaluating interval-level predictions"
run_py -m cpgdetector.evaluate --checkpoint "$CHECKPOINT" "${PRED_ARGS[@]}" --output "$RUN_DIR/interval_metrics.json"

if [[ "$SHOULD_RUN_DNABERT2" == "1" ]]; then
  log "Training DNABERT2 window baseline"
  run_py -m cpgdetector.dnabert2_baseline --config "$CONFIG" --output-dir "$RUN_DIR/dnabert2_baseline"
fi

log "Regenerating baseline comparison plots"
run_py -m cpgdetector.compare_baselines --run-dir "$RUN_DIR"

log "Generating report"
run_py -m cpgdetector.report --run-dir "$RUN_DIR" --output "$RUN_DIR/report.md"

log "Full run complete"
log "Key outputs:"
log "  $CHECKPOINT"
log "  $RUN_DIR/metrics.csv"
log "  $RUN_DIR/training_curves.png"
log "  $RUN_DIR/baseline_comparison.png"
log "  $RUN_DIR/roc_pr_curves.png"
if [[ "$SHOULD_RUN_DNABERT2" == "1" ]]; then
  log "  $RUN_DIR/dnabert2_baseline/dnabert2_metrics.json"
  log "  $RUN_DIR/dnabert2_baseline/dnabert2_roc_pr_curves.png"
fi
log "  $RUN_DIR/predicted_cpg_islands.bed"
log "  $RUN_DIR/interval_metrics.json"
log "  $RUN_DIR/report.md"
