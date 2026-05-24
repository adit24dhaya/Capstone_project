#!/usr/bin/env bash
# Run on Nautilus after restoring detector_train checkpoints and current_pcb_yolo.
set -euo pipefail

REPO="${REPO:-$HOME/Capstone_project}"
DATA_ROOT="${DATA_ROOT:-$HOME/data}"
OUTPUT_DIR="${OUTPUT_DIR:-$HOME/outputs/nautilus}"
LOG_DIR="${LOG_DIR:-$HOME/logs}"

cd "$REPO"
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

echo "=== 1) Optional: Faster R-CNN baseline (5 epochs) ==="
python tools/run_nautilus_experiments.py \
  --experiment faster_rcnn \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --frcnn-epochs 5 \
  --frcnn-batch 2 \
  --workers 0 \
  2>&1 | tee "$LOG_DIR/faster_rcnn.log"

echo "=== 2) Re-run unified eval with batch=1 for fair latency ==="
python tools/run_nautilus_experiments.py \
  --experiment paper_unified_eval \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --workers 0 \
  --device 0 \
  --prediction-save-limit 6 \
  --paper-eval-models-json reports/publication/paper_unified_eval_batch1_models.json \
  2>&1 | tee "$LOG_DIR/paper_unified_eval_batch1.log"

echo "=== 3) Zip artifacts for local machine ==="
cd "$HOME"
zip -r "backups/paper_remaining_$(date +%Y%m%d_%H%M%S).zip" \
  "$OUTPUT_DIR/runs/faster_rcnn" \
  "$OUTPUT_DIR/runs/paper_unified_eval" \
  "$LOG_DIR/faster_rcnn.log" \
  "$LOG_DIR/paper_unified_eval_batch1.log"

echo "=== 4) Champion inference sweep (current_pcb_yolo champion only) ==="
CHAMPION="${CHAMPION:-$OUTPUT_DIR/runs/detector_train/yolo11l_1280_publication/weights/best.pt}"
PCB_YAML="${PCB_YAML:-$OUTPUT_DIR/datasets/current_pcb_yolo/data.yaml}"
python - <<PY 2>&1 | tee "$LOG_DIR/champion_eval_sweep.log"
import csv
from pathlib import Path
from ultralytics import YOLO

champion = Path("${CHAMPION}")
data_yaml = Path("${PCB_YAML}")
out_csv = Path("${OUTPUT_DIR}") / "runs" / "champion_eval_sweep.csv"
rows = []
model = YOLO(str(champion))
for imgsz in (1280, 1536):
    for augment in (False, True):
        metrics = model.val(
            data=str(data_yaml),
            imgsz=imgsz,
            augment=augment,
            batch=1,
            device=0,
            workers=0,
            verbose=False,
        )
        rows.append(
            {
                "imgsz": imgsz,
                "augment": augment,
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
            }
        )
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print("Wrote", out_csv)
for r in rows:
    print(r)
PY

echo "Done. Copy zip to local_artifacts/ and rerun:"
echo "  python tools/build_publication_package.py"
echo ""
echo "For fair YOLO11s / RT-DETR / fusion on Kaggle split, run:"
echo "  KAGGLE_YAML=~/data/YOLO_PCB/data.yaml bash reports/publication/nautilus_kaggle_fair_pipeline.sh"
