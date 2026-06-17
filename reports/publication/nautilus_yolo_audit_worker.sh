#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
HOLD_FOR_COLLECTION="${HOLD_FOR_COLLECTION:-0}"
ROOT="$WORKSPACE/resolution-control"
REPO="${REPO:-/repo}"
PYTHON="$ROOT/python-env-v2/bin/python"
RAW="$ROOT/raw"
DATA="$ROOT/data"
OUTPUTS="$ROOT/outputs"
LOGS="$ROOT/logs"
CHECKPOINTS="$WORKSPACE/checkpoints"
LEGACY_YAML="$DATA/YOLO_PCB_legacy/data.yaml"
CORRECTED_YAML="$DATA/YOLO_PCB_corrected/data.yaml"
RUN_NAME="yolo11s_640_matched"
RUN_DIR="$OUTPUTS/runs/detector_train/$RUN_NAME"

mkdir -p "$LOGS" "$CHECKPOINTS"

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

if [[ ! -x "$PYTHON" ]]; then
  ENV_DIR="$ROOT/python-env-v2"
  rm -rf "$ENV_DIR"
  python -m venv --system-site-packages "$ENV_DIR"
  "$ENV_DIR/bin/python" -m pip install --no-cache-dir \
    "stringzilla==3.12.6" \
    "simsimd==6.2.1"
  "$ENV_DIR/bin/python" -m pip install --no-cache-dir \
    "ultralytics==8.4.51" \
    "numpy==1.26.4" \
    "opencv-python-headless==4.10.0.84" \
    "albumentations==2.0.8" \
    "kaggle==2.1.0" \
    pandas matplotlib scikit-learn pyyaml pillow tqdm
  "$ENV_DIR/bin/python" -m pip uninstall -y opencv-python || true
  "$ENV_DIR/bin/python" -m pip install --force-reinstall --no-deps \
    "opencv-python-headless==4.10.0.84"
fi
export PATH="$ROOT/python-env-v2/bin:$PATH"

require_file "$PYTHON"
require_file "$CHECKPOINTS/yolo11s_1280_kaggle_fair_best.pt"
require_file "$CHECKPOINTS/rtdetr_l_kaggle_fair_best.pt"

if ! "$PYTHON" -c "import albumentations, cv2, kaggle, numpy" >/dev/null 2>&1; then
  "$PYTHON" -m pip install --no-cache-dir \
    stringzilla==3.12.6 \
    simsimd==6.2.1
  "$PYTHON" -m pip install --no-cache-dir \
    numpy==1.26.4 \
    opencv-python-headless==4.10.0.84 \
    albumentations==2.0.8 \
    kaggle==2.1.0
fi

nvidia-smi

mkdir -p "$RAW" "$DATA" /root/.kaggle
if [[ -f /secrets/kaggle/access_token ]]; then
  cp /secrets/kaggle/access_token /root/.kaggle/access_token
  chmod 600 /root/.kaggle/access_token
fi

if [[ ! -d "$RAW/PCB-DATASET-master/Annotations" ]]; then
  kaggle datasets download \
    -d aditya2402/pcb-dataset \
    -p "$RAW" \
    --unzip
fi

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

require_file "$CORRECTED_YAML"

if [[ ! -f "$ROOT/legacy-eval/baseline_gate.json" ]]; then
  log_run legacy_baseline_eval.log \
    "$PYTHON" "$REPO/tools/run_nautilus_experiments.py" \
    --experiment paper_unified_eval \
    --external-data-yaml "$LEGACY_YAML" \
    --paper-eval-models-json \
      "$REPO/reports/publication/resolution_control_legacy_models.json" \
    --output-dir "$ROOT/legacy-eval" \
    --device 0 \
    --workers 0 \
    --no-paper-eval-augment \
    --prediction-save-limit 0
  log_run baseline_gate.log \
    "$PYTHON" "$REPO/tools/verify_baseline_gate.py" \
    --expected "$REPO/reports/publication/archived_baseline_metrics.csv" \
    --actual \
      "$ROOT/legacy-eval/runs/paper_unified_eval/paper_unified_eval_metrics.csv" \
    --output "$ROOT/legacy-eval/baseline_gate.json" \
    --tolerance 0.001
fi

if [[ ! -f "$ROOT/diagnostic-eval/runs/paper_unified_eval/paper_unified_eval_metrics.csv" ]]; then
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

if [[ ! -f "$ROOT/smoke/runs/detector_train/smoke_yolo11s_640/weights/best.pt" ]]; then
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
fi

resume_args=()
if [[ -s "$RUN_DIR/weights/last.pt" ]]; then
  resume_args=(--resume-checkpoint "$RUN_DIR/weights/last.pt")
fi
if [[ ! -f "$RUN_DIR/detector_train_summary.json" ]]; then
  log_run train_yolo11s_640.log \
    "$PYTHON" "$REPO/tools/run_nautilus_experiments.py" \
    --experiment detector_train \
    --external-data-yaml "$CORRECTED_YAML" \
    --yolo-model yolo11s.pt \
    --run-name "$RUN_NAME" \
    --imgsz 640 \
    --epochs 50 \
    --batch 12 \
    --seed 42 \
    --device 0 \
    --workers 0 \
    --output-dir "$OUTPUTS" \
    "${resume_args[@]}"
fi

WEIGHTS="$RUN_DIR/weights/best.pt"
require_file "$WEIGHTS"
cat > "$ROOT/yolo_eval_models.json" <<JSON
[
  {
    "model": "YOLO11s_640_matched",
    "weights": "$WEIGHTS",
    "imgsz": 640,
    "batch": 1,
    "engine": "yolo",
    "notes": "Corrected canonical labels; 50 epochs; seed 42; imgsz=640"
  }
]
JSON

if [[ ! -f "$ROOT/yolo-controlled-eval/runs/paper_unified_eval/paper_unified_eval_metrics.csv" ]]; then
  log_run yolo_controlled_eval.log \
    "$PYTHON" "$REPO/tools/run_nautilus_experiments.py" \
    --experiment paper_unified_eval \
    --external-data-yaml "$CORRECTED_YAML" \
    --paper-eval-models-json "$ROOT/yolo_eval_models.json" \
    --output-dir "$ROOT/yolo-controlled-eval" \
    --device 0 \
    --workers 0 \
    --no-paper-eval-augment \
    --prediction-save-limit 24
fi

log_run yolo_prediction_figures.log \
  "$PYTHON" "$REPO/tools/generate_resolution_control_examples.py" \
  --data-yaml "$CORRECTED_YAML" \
  --weights "$WEIGHTS" \
  --engine yolo \
  --imgsz 640 \
  --confidence 0.25 \
  --output-dir "$ROOT/yolo-paper-figures"

"$PYTHON" - "$ROOT/yolo_experiment_manifest.json" "$WEIGHTS" <<'PY'
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

root = Path("/workspace/resolution-control")
weights = Path(sys.argv[2])
manifest = {
    "model": "YOLO11s_640_matched",
    "imgsz": 640,
    "seed": 42,
    "epochs": 50,
    "training_batch": 12,
    "evaluation_batch": 1,
    "workers": 0,
    "tta": False,
    "gpu": subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
         "--format=csv,noheader"], text=True
    ).strip(),
    "hostname": platform.node(),
    "python": sys.version,
    "torch": version("torch"),
    "ultralytics": version("ultralytics"),
    "weights": str(weights),
    "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
    "dataset_manifest": json.loads(
        (root / "data/YOLO_PCB_corrected/dataset_manifest.json").read_text()
    ),
}
Path(sys.argv[1]).write_text(json.dumps(manifest, indent=2) + "\n")
PY

"$PYTHON" "$REPO/tools/build_resolution_control_manifest.py" \
  --workspace "$WORKSPACE" \
  --output "$ROOT/yolo_experiment_manifest.json" \
  --high-resolution 1280

EXPORT_ARCHIVE="$WORKSPACE/yolo_640_export.tar.gz"
tar -C "$ROOT" -czf "$EXPORT_ARCHIVE" \
  "data/YOLO_PCB_legacy/dataset_manifest.json" \
  "data/YOLO_PCB_corrected/dataset_manifest.json" \
  "legacy-eval" \
  "diagnostic-eval" \
  "yolo-controlled-eval" \
  "yolo-paper-figures" \
  "outputs/runs/detector_train/$RUN_NAME" \
  "yolo_experiment_manifest.json"
sha256sum "$EXPORT_ARCHIVE" > "$EXPORT_ARCHIVE.sha256"
touch "$WORKSPACE/YOLO_640_EXPORT_READY"

echo "YOLO audit worker complete."

if [[ "$HOLD_FOR_COLLECTION" == "1" ]]; then
  ACK_FILE="$WORKSPACE/YOLO_640_EXPORT_COLLECTED"
  echo "Export is ready at $EXPORT_ARCHIVE; waiting for $ACK_FILE"
  while [[ ! -f "$ACK_FILE" ]]; do
    sleep 30
  done
fi
