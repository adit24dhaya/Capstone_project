#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
REPO="$WORKSPACE/repo"
ROOT="$WORKSPACE/resolution-control"
RAW="$ROOT/raw"
DATA="$ROOT/data"
OUTPUTS="$ROOT/outputs"
LOGS="$ROOT/logs"
CHECKPOINTS="$WORKSPACE/checkpoints"
PYTHON="${PYTHON:-python}"
LEGACY_YAML="$DATA/YOLO_PCB_legacy/data.yaml"
CORRECTED_YAML="$DATA/YOLO_PCB_corrected/data.yaml"
HIGH_RESOLUTION=1280
FALLBACK_REASON=""

mkdir -p "$RAW" "$DATA" "$OUTPUTS" "$LOGS" "$CHECKPOINTS"

log_run() {
  local log_name="$1"
  shift
  "$@" 2>&1 | tee "$LOGS/$log_name"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

install_dependencies() {
  apt-get update
  apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libgl1 \
    libglib2.0-0
  "$PYTHON" -m pip install --upgrade pip
  "$PYTHON" -m pip install \
    "ultralytics==8.4.51" \
    "numpy==1.26.4" \
    "opencv-python-headless==4.10.0.84" \
    "albumentations==2.0.8" \
    "kaggle==2.1.0" \
    pandas matplotlib scikit-learn pyyaml pillow tqdm
}

prepare_kaggle_auth() {
  mkdir -p /root/.kaggle
  cp /secrets/kaggle/access_token /root/.kaggle/access_token
  chmod 600 /root/.kaggle/access_token
}

download_base_dataset() {
  if [[ -d "$RAW/PCB-DATASET-master/Annotations" ]]; then
    return
  fi
  prepare_kaggle_auth
  kaggle datasets download \
    -d aditya2402/pcb-dataset \
    -p "$RAW" \
    --unzip
}

build_datasets() {
  if [[ ! -f "$LEGACY_YAML" ]]; then
    log_run prepare_legacy_dataset.log \
      "$PYTHON" "$REPO/tools/prepare_resolution_control_dataset.py" \
      --base-root "$RAW" \
      --cache-dir "$RAW/cache" \
      --output "$DATA/YOLO_PCB_legacy" \
      --label-mode legacy
  fi
  if [[ ! -f "$CORRECTED_YAML" ]]; then
    log_run prepare_corrected_dataset.log \
      "$PYTHON" "$REPO/tools/prepare_resolution_control_dataset.py" \
      --base-root "$RAW" \
      --cache-dir "$RAW/cache" \
      --output "$DATA/YOLO_PCB_corrected" \
      --label-mode corrected
  fi
}

run_baseline_gate() {
  local gate_csv="$ROOT/legacy-eval/runs/paper_unified_eval/paper_unified_eval_metrics.csv"
  if [[ ! -f "$ROOT/legacy-eval/baseline_gate.json" ]]; then
    log_run legacy_baseline_eval.log \
      "$PYTHON" "$REPO/tools/run_nautilus_experiments.py" \
      --experiment paper_unified_eval \
      --external-data-yaml "$LEGACY_YAML" \
      --paper-eval-models-json \
        "$REPO/reports/publication/resolution_control_legacy_models.json" \
      --output-dir "$ROOT/legacy-eval" \
      --batch 1 \
      --device 0 \
      --workers 0 \
      --no-paper-eval-augment \
      --prediction-save-limit 0
    log_run baseline_gate.log \
      "$PYTHON" "$REPO/tools/verify_baseline_gate.py" \
      --expected "$REPO/reports/publication/archived_baseline_metrics.csv" \
      --actual "$gate_csv" \
      --output "$ROOT/legacy-eval/baseline_gate.json" \
      --tolerance 0.001
  fi
}

run_diagnostic() {
  local metrics="$ROOT/diagnostic-eval/runs/paper_unified_eval/paper_unified_eval_metrics.csv"
  if [[ ! -f "$metrics" ]]; then
    log_run yolo_checkpoint_at_640.log \
      "$PYTHON" "$REPO/tools/run_nautilus_experiments.py" \
      --experiment paper_unified_eval \
      --external-data-yaml "$LEGACY_YAML" \
      --paper-eval-models-json \
        "$REPO/reports/publication/resolution_control_diagnostic_models.json" \
      --output-dir "$ROOT/diagnostic-eval" \
      --device 0 \
      --workers 0 \
      --no-paper-eval-augment \
      --prediction-save-limit 0
  fi
}

train_yolo640() {
  local best="$OUTPUTS/runs/detector_train/yolo11s_640_matched/weights/best.pt"
  if [[ ! -f "$ROOT/smoke/yolo11s_640.ok" ]]; then
    log_run smoke_yolo11s_640.log \
      "$PYTHON" "$REPO/tools/run_nautilus_experiments.py" \
      --experiment detector_train \
      --external-data-yaml "$CORRECTED_YAML" \
      --yolo-model yolo11s.pt \
      --run-name smoke_yolo11s_640 \
      --imgsz 640 \
      --epochs 1 \
      --batch 12 \
      --seed 42 \
      --device 0 \
      --workers 0 \
      --output-dir "$ROOT/smoke"
    mkdir -p "$ROOT/smoke"
    touch "$ROOT/smoke/yolo11s_640.ok"
  fi
  if [[ ! -f "$best" ]]; then
    log_run train_yolo11s_640.log \
      "$PYTHON" "$REPO/tools/run_nautilus_experiments.py" \
      --experiment detector_train \
      --external-data-yaml "$CORRECTED_YAML" \
      --yolo-model yolo11s.pt \
      --run-name yolo11s_640_matched \
      --imgsz 640 \
      --epochs 50 \
      --batch 12 \
      --seed 42 \
      --device 0 \
      --workers 0 \
      --output-dir "$OUTPUTS"
  fi
}

train_rtdetr640() {
  local best="$OUTPUTS/runs/rtdetr/rtdetr_l_640_matched/weights/best.pt"
  if [[ ! -f "$ROOT/smoke/rtdetr_l_640.ok" ]]; then
    log_run smoke_rtdetr_l_640.log \
      "$PYTHON" "$REPO/tools/run_nautilus_experiments.py" \
      --experiment detector_train_rtdetr \
      --external-data-yaml "$CORRECTED_YAML" \
      --rtdetr-model rtdetr-l.pt \
      --run-name smoke_rtdetr_l_640 \
      --rtdetr-imgsz 640 \
      --epochs 1 \
      --batch 4 \
      --seed 42 \
      --device 0 \
      --workers 0 \
      --output-dir "$ROOT/smoke"
    touch "$ROOT/smoke/rtdetr_l_640.ok"
  fi
  if [[ ! -f "$best" ]]; then
    log_run train_rtdetr_l_640.log \
      "$PYTHON" "$REPO/tools/run_nautilus_experiments.py" \
      --experiment detector_train_rtdetr \
      --external-data-yaml "$CORRECTED_YAML" \
      --rtdetr-model rtdetr-l.pt \
      --run-name rtdetr_l_640_matched \
      --rtdetr-imgsz 640 \
      --epochs 10 \
      --batch 4 \
      --seed 42 \
      --device 0 \
      --workers 0 \
      --output-dir "$OUTPUTS"
  fi
}

run_rtdetr_high_resolution() {
  local resolution="$1"
  local output_dir="$2"
  local run_name="$3"
  local epochs="$4"
  log_run "${run_name}.log" \
    "$PYTHON" "$REPO/tools/run_nautilus_experiments.py" \
    --experiment detector_train_rtdetr \
    --external-data-yaml "$CORRECTED_YAML" \
    --rtdetr-model rtdetr-l.pt \
    --run-name "$run_name" \
    --rtdetr-imgsz "$resolution" \
    --epochs "$epochs" \
    --batch 1 \
    --seed 42 \
    --device 0 \
    --workers 0 \
    --output-dir "$output_dir"
}

confirmed_cuda_oom() {
  grep -Eqi "CUDA out of memory|OutOfMemoryError|CUDA error: out of memory" "$1"
}

train_rtdetr_high_resolution() {
  local best="$OUTPUTS/runs/rtdetr/rtdetr_l_high_resolution/weights/best.pt"
  if [[ -f "$ROOT/high_resolution.json" ]]; then
    HIGH_RESOLUTION="$("$PYTHON" -c \
      "import json; print(json.load(open('$ROOT/high_resolution.json'))['imgsz'])")"
    FALLBACK_REASON="$("$PYTHON" -c \
      "import json; print(json.load(open('$ROOT/high_resolution.json')).get('fallback_reason') or '')")"
  fi

  if [[ ! -f "$ROOT/smoke/rtdetr_l_high_resolution.ok" ]]; then
    set +e
    run_rtdetr_high_resolution \
      1280 "$ROOT/smoke" smoke_rtdetr_l_1280 1
    local status=$?
    set -e
    if [[ "$status" -ne 0 ]]; then
      if ! confirmed_cuda_oom "$LOGS/smoke_rtdetr_l_1280.log"; then
        echo "RT-DETR-L 1280 smoke test failed for a reason other than CUDA OOM." >&2
        exit "$status"
      fi
      HIGH_RESOLUTION=960
      FALLBACK_REASON="RT-DETR-L 1280 one-epoch smoke test produced a confirmed CUDA OOM."
      run_rtdetr_high_resolution \
        960 "$ROOT/smoke" smoke_rtdetr_l_960 1
    fi
    "$PYTHON" - "$ROOT/high_resolution.json" "$HIGH_RESOLUTION" "$FALLBACK_REASON" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "imgsz": int(sys.argv[2]),
    "preferred_imgsz": 1280,
    "fallback_imgsz": 960,
    "fallback_reason": sys.argv[3] or None,
}, indent=2) + "\n")
PY
    touch "$ROOT/smoke/rtdetr_l_high_resolution.ok"
  fi

  if [[ ! -f "$best" ]]; then
    set +e
    run_rtdetr_high_resolution \
      "$HIGH_RESOLUTION" "$OUTPUTS" rtdetr_l_high_resolution 10
    local status=$?
    set -e
    if [[ "$status" -ne 0 && "$HIGH_RESOLUTION" -eq 1280 ]]; then
      if ! confirmed_cuda_oom "$LOGS/rtdetr_l_high_resolution.log"; then
        exit "$status"
      fi
      HIGH_RESOLUTION=960
      FALLBACK_REASON="RT-DETR-L 1280 full training produced a confirmed CUDA OOM."
      rm -rf "$OUTPUTS/runs/rtdetr/rtdetr_l_high_resolution"
      run_rtdetr_high_resolution \
        960 "$OUTPUTS" rtdetr_l_high_resolution 10
      "$PYTHON" - "$ROOT/high_resolution.json" "$FALLBACK_REASON" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(json.dumps({
    "imgsz": 960,
    "preferred_imgsz": 1280,
    "fallback_imgsz": 960,
    "fallback_reason": sys.argv[2],
}, indent=2) + "\n")
PY
    elif [[ "$status" -ne 0 ]]; then
      exit "$status"
    fi
  fi
}

run_controlled_evaluation() {
  local config="$ROOT/corrected_models_runtime.json"
  "$PYTHON" - \
    "$REPO/reports/publication/resolution_control_corrected_models.json" \
    "$config" \
    "$HIGH_RESOLUTION" <<'PY'
import json
import sys
from pathlib import Path

models = json.loads(Path(sys.argv[1]).read_text())
models[-1]["imgsz"] = int(sys.argv[3])
models[-1]["notes"] = (
    f"Corrected canonical labels; 10 epochs; seed 42; imgsz={sys.argv[3]}"
)
Path(sys.argv[2]).write_text(json.dumps(models, indent=2) + "\n")
PY
  local metrics="$ROOT/controlled-eval/runs/paper_unified_eval/paper_unified_eval_metrics.csv"
  if [[ ! -f "$metrics" ]]; then
    log_run controlled_unified_eval.log \
      "$PYTHON" "$REPO/tools/run_nautilus_experiments.py" \
      --experiment paper_unified_eval \
      --external-data-yaml "$CORRECTED_YAML" \
      --paper-eval-models-json "$config" \
      --output-dir "$ROOT/controlled-eval" \
      --device 0 \
      --workers 0 \
      --no-paper-eval-augment \
      --prediction-save-limit 24
  fi
}

generate_prediction_figures() {
  local figure_dir="$ROOT/controlled-eval/paper_figures"
  if [[ ! -f "$figure_dir/representative_failure_cases.png" ]]; then
    log_run prediction_figures.log \
      "$PYTHON" "$REPO/tools/generate_resolution_control_examples.py" \
      --data-yaml "$CORRECTED_YAML" \
      --weights \
        "$OUTPUTS/runs/detector_train/yolo11s_640_matched/weights/best.pt" \
      --engine yolo \
      --imgsz 640 \
      --confidence 0.25 \
      --output-dir "$figure_dir"
  fi
}

render_manuscript_tables() {
  log_run render_manuscript_tables.log \
    "$PYTHON" "$REPO/tools/render_resolution_control_tables.py" \
    --legacy-metrics \
      "$ROOT/legacy-eval/runs/paper_unified_eval/paper_unified_eval_metrics.csv" \
    --diagnostic-metrics \
      "$ROOT/diagnostic-eval/runs/paper_unified_eval/paper_unified_eval_metrics.csv" \
    --controlled-metrics \
      "$ROOT/controlled-eval/runs/paper_unified_eval/paper_unified_eval_metrics.csv" \
    --controlled-per-class \
      "$ROOT/controlled-eval/runs/paper_unified_eval/paper_unified_eval_per_class.csv" \
    --output-dir "$ROOT/manuscript_tables"
}

build_manifest() {
  log_run build_manifest.log \
    "$PYTHON" "$REPO/tools/build_resolution_control_manifest.py" \
    --workspace "$WORKSPACE" \
    --output "$ROOT/experiment_manifest.json" \
    --high-resolution "$HIGH_RESOLUTION" \
    --fallback-reason "$FALLBACK_REASON"
}

main() {
  require_file "$CHECKPOINTS/yolo11s_1280_kaggle_fair_best.pt"
  require_file "$CHECKPOINTS/rtdetr_l_kaggle_fair_best.pt"
  install_dependencies
  nvidia-smi
  download_base_dataset
  build_datasets
  run_baseline_gate
  run_diagnostic
  train_yolo640
  train_rtdetr640
  train_rtdetr_high_resolution
  run_controlled_evaluation
  generate_prediction_figures
  render_manuscript_tables
  build_manifest
  echo "Resolution-control pipeline completed: $ROOT"
}

main "$@"
