#!/usr/bin/env bash
set -euo pipefail

MODE="${MODE:?MODE must be 640 or highres}"
WORKSPACE="${WORKSPACE:-/workspace}"
HOLD_FOR_COLLECTION="${HOLD_FOR_COLLECTION:-0}"
ROOT="$WORKSPACE/resolution-control"
RAW="$ROOT/raw"
DATA="$ROOT/data/YOLO_PCB_corrected"
OUTPUTS="$ROOT/outputs"
LOGS="$ROOT/logs"
REPO="/repo"
PYTHON="${PYTHON:-python}"
PYTHON_ENV="$ROOT/python-env-v2"
PYTHON_ENV_MARKER="$PYTHON_ENV/.resolution-control-v2"
IMG_SIZE=640
BATCH=4
RUN_NAME="rtdetr_l_640_matched"
FALLBACK_REASON=""

if [[ "$MODE" == "highres" ]]; then
  IMG_SIZE=1280
  BATCH=1
  RUN_NAME="rtdetr_l_high_resolution"
elif [[ "$MODE" != "640" ]]; then
  echo "Unsupported MODE: $MODE" >&2
  exit 2
fi

mkdir -p "$RAW" "$OUTPUTS" "$LOGS"

log_run() {
  local log_name="$1"
  shift
  "$@" 2>&1 | tee "$LOGS/$log_name"
}

if [[ ! -f "$PYTHON_ENV_MARKER" ]]; then
  echo "Persistent Python environment is not ready. Run pcb-rtdetr-highres-runtime-prep first." >&2
  exit 3
fi
echo "Using persistent Python environment from $PYTHON_ENV"
PYTHON="$PYTHON_ENV/bin/python"
export PATH="$PYTHON_ENV/bin:$PATH"

nvidia-smi
mkdir -p /root/.kaggle
cp /secrets/kaggle/access_token /root/.kaggle/access_token
chmod 600 /root/.kaggle/access_token

if [[ ! -d "$RAW/PCB-DATASET-master/Annotations" ]]; then
  kaggle datasets download \
    -d aditya2402/pcb-dataset \
    -p "$RAW" \
    --unzip
fi

if [[ ! -f "$DATA/data.yaml" ]]; then
  log_run prepare_corrected_dataset.log \
    "$PYTHON" "$REPO/tools/prepare_resolution_control_dataset.py" \
    --base-root "$RAW" \
    --cache-dir "$RAW/cache" \
    --output "$DATA" \
    --label-mode corrected
fi

run_training() {
  local image_size="$1"
  local epochs="$2"
  local output_dir="$3"
  local run_name="$4"
  local batch="$5"
  local resume_checkpoint="$output_dir/runs/rtdetr/$run_name/weights/last.pt"
  local resume_args=()
  if [[ -s "$resume_checkpoint" ]]; then
    echo "Found saved training state; resuming from $resume_checkpoint"
    resume_args=(--resume-checkpoint "$resume_checkpoint")
  fi
  log_run "${run_name}.log" \
    "$PYTHON" "$REPO/tools/run_nautilus_experiments.py" \
    --experiment detector_train_rtdetr \
    --external-data-yaml "$DATA/data.yaml" \
    --rtdetr-model rtdetr-l.pt \
    --run-name "$run_name" \
    --rtdetr-imgsz "$image_size" \
    --epochs "$epochs" \
    --batch "$batch" \
    --seed 42 \
    --device 0 \
    --workers 0 \
    --output-dir "$output_dir" \
    "${resume_args[@]}"
}

confirmed_cuda_oom() {
  grep -Eqi "CUDA out of memory|OutOfMemoryError|CUDA error: out of memory" "$1"
}

if [[ "$MODE" == "highres" ]]; then
  if [[ -s "$ROOT/smoke/runs/rtdetr/smoke_rtdetr_l_1280/weights/best.pt" ]]; then
    echo "RT-DETR-L 1280 smoke test already passed; reusing its saved checkpoint."
  else
    set +e
    run_training 1280 1 "$ROOT/smoke" smoke_rtdetr_l_1280 1
    status=$?
    set -e
    if [[ "$status" -ne 0 ]]; then
      if ! confirmed_cuda_oom "$LOGS/smoke_rtdetr_l_1280.log"; then
        exit "$status"
      fi
      IMG_SIZE=960
      FALLBACK_REASON="RT-DETR-L 1280 smoke test produced a confirmed CUDA OOM."
      run_training 960 1 "$ROOT/smoke" smoke_rtdetr_l_960 1
    fi
  fi
else
  if [[ -s "$ROOT/smoke/runs/rtdetr/smoke_rtdetr_l_640/weights/best.pt" ]]; then
    echo "RT-DETR-L 640 smoke test already passed; reusing its saved checkpoint."
  else
    run_training 640 1 "$ROOT/smoke" smoke_rtdetr_l_640 4
  fi
fi

set +e
run_training "$IMG_SIZE" 10 "$OUTPUTS" "$RUN_NAME" "$BATCH"
status=$?
set -e
if [[ "$status" -ne 0 && "$MODE" == "highres" && "$IMG_SIZE" -eq 1280 ]]; then
  if ! confirmed_cuda_oom "$LOGS/$RUN_NAME.log"; then
    exit "$status"
  fi
  IMG_SIZE=960
  FALLBACK_REASON="RT-DETR-L 1280 full training produced a confirmed CUDA OOM."
  rm -rf "$OUTPUTS/runs/rtdetr/$RUN_NAME"
  run_training 960 10 "$OUTPUTS" "$RUN_NAME" 1
elif [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

"$PYTHON" - "$ROOT/eval_models.json" "$RUN_NAME" "$IMG_SIZE" "$MODE" <<'PY'
import json
import sys
from pathlib import Path

model_id = (
    "RTDETR_L_640_matched"
    if sys.argv[4] == "640"
    else "RTDETR_L_high_resolution"
)
config = [{
    "model": model_id,
    "weights": f"/workspace/resolution-control/outputs/runs/rtdetr/{sys.argv[2]}/weights/best.pt",
    "imgsz": int(sys.argv[3]),
    "batch": 1,
    "engine": "rtdetr",
    "notes": f"Corrected labels; seed 42; imgsz={sys.argv[3]}",
}]
Path(sys.argv[1]).write_text(json.dumps(config, indent=2) + "\n")
PY

log_run unified_eval.log \
  "$PYTHON" "$REPO/tools/run_nautilus_experiments.py" \
  --experiment paper_unified_eval \
  --external-data-yaml "$DATA/data.yaml" \
  --paper-eval-models-json "$ROOT/eval_models.json" \
  --output-dir "$ROOT/eval" \
  --device 0 \
  --workers 0 \
  --no-paper-eval-augment \
  --prediction-save-limit 0

"$PYTHON" - "$ROOT/worker_manifest.json" "$MODE" "$IMG_SIZE" "$FALLBACK_REASON" <<'PY'
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

root = Path("/workspace/resolution-control")
run_name = "rtdetr_l_640_matched" if sys.argv[2] == "640" else "rtdetr_l_high_resolution"
weights = root / "outputs/runs/rtdetr" / run_name / "weights/best.pt"
digest = hashlib.sha256(weights.read_bytes()).hexdigest()
gpu = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
    text=True,
).strip()
manifest = {
    "mode": sys.argv[2],
    "imgsz": int(sys.argv[3]),
    "preferred_imgsz": 1280 if sys.argv[2] == "highres" else 640,
    "fallback_reason": sys.argv[4] or None,
    "seed": 42,
    "epochs": 10,
    "training_batch": 1 if sys.argv[2] == "highres" else 4,
    "evaluation_batch": 1,
    "workers": 0,
    "tta": False,
    "gpu": gpu,
    "hostname": platform.node(),
    "python": sys.version,
    "torch": version("torch"),
    "ultralytics": version("ultralytics"),
    "weights": str(weights),
    "weights_sha256": digest,
    "dataset_manifest": json.loads((root / "data/YOLO_PCB_corrected/dataset_manifest.json").read_text()),
}
Path(sys.argv[1]).write_text(json.dumps(manifest, indent=2) + "\n")
PY

EXPORT_ARCHIVE="$WORKSPACE/rtdetr_${MODE}_export.tar.gz"
tar -C "$ROOT" -czf "$EXPORT_ARCHIVE" \
  "data/YOLO_PCB_corrected/dataset_manifest.json" \
  "eval" \
  "logs" \
  "outputs/runs/rtdetr/$RUN_NAME" \
  "worker_manifest.json"
sha256sum "$EXPORT_ARCHIVE" > "$EXPORT_ARCHIVE.sha256"
touch "$WORKSPACE/RTDETR_${MODE}_EXPORT_READY"

echo "RT-DETR worker complete: mode=$MODE imgsz=$IMG_SIZE"

if [[ "$HOLD_FOR_COLLECTION" == "1" ]]; then
  ACK_FILE="$WORKSPACE/RTDETR_${MODE}_EXPORT_COLLECTED"
  echo "Export is ready at $EXPORT_ARCHIVE; waiting for $ACK_FILE"
  while [[ ! -f "$ACK_FILE" ]]; do
    sleep 30
  done
fi
