#!/usr/bin/env bash
# Primary-benchmark pipeline on Nautilus current_pcb_yolo (same split as YOLO11l champion).
# 1) batch=1 unified eval (existing checkpoints)
# 2) YOLO11s train (Kaggle v16-aligned hyperparameters, documented caveats)
# 3) RT-DETR-L train
# 4) Adaptive fusion (no --external-data-yaml; uses conversion summary)
# 5) batch=1 unified eval including new baselines
#
# Prereqs: OUTPUT_DIR has current_pcb_conversion_summary.json + four YOLO11l/m checkpoints.
set -euo pipefail

REPO="${REPO:-$HOME/Capstone_project}"
DATA_ROOT="${DATA_ROOT:-$HOME/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/outputs/nautilus}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"

YOLO11S_RUN="${YOLO11S_RUN:-yolo11s_1280_current_pcb}"
RTDETR_RUN="${RTDETR_RUN:-rtdetr_l_current_pcb}"

cd "$REPO"
git pull
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

if [[ ! -f "$OUTPUT_DIR/current_pcb_conversion_summary.json" ]]; then
  echo "Missing conversion summary. Run smoke/convert first."
  exit 1
fi

echo "=== 0) Backup prior unified eval (if any) ==="
if [[ -d "$OUTPUT_DIR/runs/paper_unified_eval" ]]; then
  cp -a "$OUTPUT_DIR/runs/paper_unified_eval" \
    "$OUTPUT_DIR/runs/paper_unified_eval_backup_$(date +%Y%m%d_%H%M%S)"
fi

echo "=== 1) batch=1 unified eval — existing YOLO checkpoints (Table 2 latency) ==="
python tools/run_nautilus_experiments.py \
  --experiment paper_unified_eval \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --workers 0 \
  --device 0 \
  --prediction-save-limit 6 \
  --paper-eval-models-json reports/publication/paper_unified_eval_batch1_models.json \
  2>&1 | tee "$LOG_DIR/paper_unified_eval_batch1_existing.log"

mv "$OUTPUT_DIR/runs/paper_unified_eval" "$OUTPUT_DIR/runs/paper_unified_eval_batch1_existing"
mkdir -p "$OUTPUT_DIR/runs/paper_unified_eval"

echo "=== 2) Train YOLO11s on current_pcb_yolo (v16-aligned: 1280, 50 epochs, batch 12) ==="
python tools/run_nautilus_experiments.py \
  --experiment detector_train \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --yolo-model yolo11s.pt \
  --run-name "$YOLO11S_RUN" \
  --imgsz 1280 \
  --epochs 50 \
  --batch 12 \
  --patience 25 \
  --close-mosaic 15 \
  --cos-lr \
  --seed 42 \
  --device 0 \
  --workers 0 \
  2>&1 | tee "$LOG_DIR/${YOLO11S_RUN}_train.log"

echo "=== 3) Train RT-DETR-L on same data.yaml (v16-aligned: 640, 10 epochs, batch 4) ==="
python tools/run_nautilus_experiments.py \
  --experiment detector_train_rtdetr \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --rtdetr-model rtdetr-l.pt \
  --run-name "$RTDETR_RUN" \
  --rtdetr-imgsz 640 \
  --epochs 10 \
  --batch 4 \
  --patience 10 \
  --seed 42 \
  --device 0 \
  --workers 0 \
  2>&1 | tee "$LOG_DIR/${RTDETR_RUN}_train.log"

YOLO11S_WTS="$OUTPUT_DIR/runs/detector_train/${YOLO11S_RUN}/weights/best.pt"
RTDETR_WTS="$OUTPUT_DIR/runs/rtdetr/${RTDETR_RUN}/weights/best.pt"

echo "=== 4) Adaptive fusion on current_pcb_yolo (default yaml, not Kaggle external) ==="
python tools/run_nautilus_experiments.py \
  --experiment adaptive_fusion \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --yolo-weights "$YOLO11S_WTS" \
  --rtdetr-weights "$RTDETR_WTS" \
  --imgsz 1280 \
  --yolo-imgsz 1280 \
  --rtdetr-imgsz 640 \
  --device 0 \
  --workers 0 \
  2>&1 | tee "$LOG_DIR/adaptive_fusion_current_pcb.log"

echo "=== 5) Write batch=1 unified eval JSON (all models) and run ==="
EVAL_JSON="$OUTPUT_DIR/runs/paper_unified_eval_batch1_all_models.json"
python - <<PY
import json
from pathlib import Path

out = Path("${OUTPUT_DIR}")
models = [
    ("YOLO11l_1280_champion", out / "runs/detector_train/yolo11l_1280_publication/weights/best.pt", 1280, "yolo"),
    ("YOLO11l_1280_v100_sensitivity", out / "runs/detector_train/yolo11l_1280_publication_v100_w0/weights/best.pt", 1280, "yolo"),
    ("YOLO11l_1280_refine_lowaug_v1", out / "runs/detector_train/yolo11l_1280_refine_lowaug_v1/weights/best.pt", 1280, "yolo"),
    ("YOLO11m_960_ablation", out / "runs/detector_train/yolo11m_960_publication/weights/best.pt", 960, "yolo"),
    ("YOLO11s_1280_current_pcb", out / "runs/detector_train/${YOLO11S_RUN}/weights/best.pt", 1280, "yolo"),
    ("RT-DETR-L_current_pcb", out / "runs/rtdetr/${RTDETR_RUN}/weights/best.pt", 640, "rtdetr"),
]
rows = []
for name, weights, imgsz, engine in models:
    rows.append({
        "model": name,
        "weights": str(weights),
        "imgsz": imgsz,
        "batch": 1,
        "engine": engine,
        "notes": "batch=1 fair latency; current_pcb_yolo split",
    })
Path("${EVAL_JSON}").write_text(json.dumps(rows, indent=2) + "\n")
print("Wrote", "${EVAL_JSON}")
PY

python tools/run_nautilus_experiments.py \
  --experiment paper_unified_eval \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --workers 0 \
  --device 0 \
  --prediction-save-limit 6 \
  --paper-eval-models-json "$EVAL_JSON" \
  2>&1 | tee "$LOG_DIR/paper_unified_eval_batch1_all.log"

mv "$OUTPUT_DIR/runs/paper_unified_eval" "$OUTPUT_DIR/runs/paper_unified_eval_batch1_all"

echo "=== 6) Zip artifacts ==="
cd "$HOME"
zip -r "backups/primary_baselines_$(date +%Y%m%d_%H%M%S).zip" \
  "$OUTPUT_DIR/runs/detector_train/${YOLO11S_RUN}" \
  "$OUTPUT_DIR/runs/rtdetr/${RTDETR_RUN}" \
  "$OUTPUT_DIR/runs/adaptive_fusion" \
  "$OUTPUT_DIR/runs/paper_unified_eval_batch1_existing" \
  "$OUTPUT_DIR/runs/paper_unified_eval_batch1_all" \
  "$EVAL_JSON" \
  "$LOG_DIR/paper_unified_eval_batch1_existing.log" \
  "$LOG_DIR/${YOLO11S_RUN}_train.log" \
  "$LOG_DIR/${RTDETR_RUN}_train.log" \
  "$LOG_DIR/adaptive_fusion_current_pcb.log" \
  "$LOG_DIR/paper_unified_eval_batch1_all.log"

echo "Done."
echo "  Table 2 (batch=1): $OUTPUT_DIR/runs/paper_unified_eval_batch1_existing/"
echo "  Full comparison:   $OUTPUT_DIR/runs/paper_unified_eval_batch1_all/"
echo "  Fusion metrics:    $OUTPUT_DIR/runs/adaptive_fusion/"
