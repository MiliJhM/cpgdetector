#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-cpgdetector}"
CONFIG="${CONFIG:-configs/default.yaml}"
RUN_DIR="${RUN_DIR:-runs/default}"
PRED_CHROMS="${PRED_CHROMS:-chr19 chr20 chr21 chr22}"
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

log "Generating report"
run_py -m cpgdetector.report --run-dir "$RUN_DIR" --output "$RUN_DIR/report.md"

log "Full run complete"
log "Key outputs:"
log "  $CHECKPOINT"
log "  $RUN_DIR/metrics.csv"
log "  $RUN_DIR/training_curves.png"
log "  $RUN_DIR/baseline_comparison.png"
log "  $RUN_DIR/roc_pr_curves.png"
log "  $RUN_DIR/predicted_cpg_islands.bed"
log "  $RUN_DIR/interval_metrics.json"
log "  $RUN_DIR/report.md"
