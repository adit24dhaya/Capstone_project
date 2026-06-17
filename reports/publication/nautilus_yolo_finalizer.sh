#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
ROOT="$WORKSPACE/resolution-control"
REPO="/repo"
PYTHON="${PYTHON:-python}"
DATA_YAML="$ROOT/data/YOLO_PCB_corrected/data.yaml"
WEIGHTS="$ROOT/outputs/runs/detector_train/yolo11s_640_matched/weights/best.pt"
EVAL_ROOT="$ROOT/yolo-controlled-eval"
FIGURE_DIR="$ROOT/yolo-paper-figures"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Required file is missing: $1" >&2
    exit 1
  fi
}

require_file "$DATA_YAML"
require_file "$WEIGHTS"

apt-get update
apt-get install -y --no-install-recommends libgl1 libglib2.0-0
"$PYTHON" -m pip install --upgrade pip
"$PYTHON" -m pip install \
  "ultralytics==8.4.51" \
  "numpy==1.26.4" \
  "opencv-python-headless==4.10.0.84" \
  pandas matplotlib pyyaml pillow

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

if [[ ! -f "$EVAL_ROOT/runs/paper_unified_eval/paper_unified_eval_metrics.csv" ]]; then
  "$PYTHON" "$REPO/tools/run_nautilus_experiments.py" \
    --experiment paper_unified_eval \
    --external-data-yaml "$DATA_YAML" \
    --paper-eval-models-json "$ROOT/yolo_eval_models.json" \
    --output-dir "$EVAL_ROOT" \
    --device 0 \
    --workers 0 \
    --no-paper-eval-augment \
    --prediction-save-limit 24
fi

"$PYTHON" "$REPO/tools/generate_resolution_control_examples.py" \
  --data-yaml "$DATA_YAML" \
  --weights "$WEIGHTS" \
  --engine yolo \
  --imgsz 640 \
  --confidence 0.25 \
  --output-dir "$FIGURE_DIR"

"$PYTHON" "$REPO/tools/build_resolution_control_manifest.py" \
  --workspace "$WORKSPACE" \
  --output "$ROOT/yolo_experiment_manifest.json" \
  --high-resolution 1280

echo "YOLO controlled evaluation and paper figures completed."
