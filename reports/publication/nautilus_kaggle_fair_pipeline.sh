#!/usr/bin/env bash
# Fair Table 5 baselines on the Kaggle-prepared YOLO_PCB split (~4751/1016/1016).
# Prerequisite: prepared dataset on the pod (see header comments below).
set -euo pipefail

REPO="${REPO:-$HOME/Capstone_project}"
DATA_ROOT="${DATA_ROOT:-$HOME/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/outputs/nautilus}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"

# Point this at the data.yaml from the Kaggle notebook (YOLO_PCB).
KAGGLE_YAML="${KAGGLE_YAML:-$DATA_ROOT/YOLO_PCB/data.yaml}"

if [[ ! -f "$KAGGLE_YAML" ]]; then
  echo "Missing KAGGLE_YAML: $KAGGLE_YAML"
  echo "Prepare YOLO_PCB on Kaggle (project.ipynb) and copy to the pod, e.g.:"
  echo "  ~/data/YOLO_PCB/{train,val,test}/{images,labels} + data.yaml"
  exit 1
fi

cd "$REPO"
git pull
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

echo "=== 1) YOLO11s on Kaggle split (v16 settings: 1280, 50 ep, batch 12) ==="
python tools/run_nautilus_experiments.py \
  --experiment detector_train \
  --external-data-yaml "$KAGGLE_YAML" \
  --yolo-model yolo11s.pt \
  --imgsz 1280 \
  --epochs 50 \
  --batch 12 \
  --device 0 \
  --workers 0 \
  --output-dir "$OUTPUT_DIR" \
  --run-name yolo11s_1280_kaggle_fair \
  2>&1 | tee "$LOG_DIR/yolo11s_kaggle_fair.log"

YOLO11S_WTS="$OUTPUT_DIR/runs/detector_train/yolo11s_1280_kaggle_fair/weights/best.pt"

echo "=== 2) RT-DETR-L on Kaggle split (v16: 640, 10 ep, batch 4) ==="
python - <<PY 2>&1 | tee "$LOG_DIR/rtdetr_kaggle_fair.log"
from pathlib import Path
from ultralytics import RTDETR

data_yaml = Path("${KAGGLE_YAML}")
out = Path("${OUTPUT_DIR}") / "runs" / "rtdetr" / "rtdetr_l_kaggle_fair"
out.parent.mkdir(parents=True, exist_ok=True)
model = RTDETR("rtdetr-l.pt")
model.train(
    data=str(data_yaml),
    imgsz=640,
    epochs=10,
    batch=4,
    device=0,
    workers=0,
    project=str(out.parent),
    name=out.name,
    exist_ok=True,
)
print("best:", out / "weights" / "best.pt")
PY

RTDETR_WTS="$OUTPUT_DIR/runs/rtdetr/rtdetr_l_kaggle_fair/weights/best.pt"

echo "=== 3) Adaptive fusion on Kaggle test split ==="
python tools/run_nautilus_experiments.py \
  --experiment adaptive_fusion \
  --external-data-yaml "$KAGGLE_YAML" \
  --yolo-weights "$YOLO11S_WTS" \
  --rtdetr-weights "$RTDETR_WTS" \
  --output-dir "$OUTPUT_DIR" \
  --device 0 \
  --workers 0 \
  2>&1 | tee "$LOG_DIR/adaptive_fusion_kaggle_fair.log"

echo "=== 4) Zip Kaggle-fair artifacts ==="
cd "$HOME"
zip -r "backups/kaggle_fair_$(date +%Y%m%d_%H%M%S).zip" \
  "$OUTPUT_DIR/runs/detector_train/yolo11s_1280_kaggle_fair" \
  "$OUTPUT_DIR/runs/rtdetr/rtdetr_l_kaggle_fair" \
  "$OUTPUT_DIR/runs/adaptive_fusion" \
  "$LOG_DIR/yolo11s_kaggle_fair.log" \
  "$LOG_DIR/rtdetr_kaggle_fair.log" \
  "$LOG_DIR/adaptive_fusion_kaggle_fair.log"

echo "Done. On Mac: unzip to local_artifacts/ and refresh publication tables."
