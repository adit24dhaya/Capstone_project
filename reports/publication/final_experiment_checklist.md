# Final Experiment Checklist

## Completed Evidence

- YOLO11s baseline from Kaggle v16.
- RT-DETR-L transformer-style baseline from Kaggle v16.
- Hybrid YOLO11s + RT-DETR-L fusion experiments from Kaggle v16.
- Adaptive defect-aware fusion, defect-size, calibration, robustness, and inspection-cost artifacts from Kaggle v16.
- YOLO11m 960 Nautilus/RTX 2080 Ti result with saved `best.pt`, logs, curves, and metrics.
- YOLO11l 1280 Nautilus/RTX 2080 Ti result in `local_artifacts/yolo11l_1280_publication_outputs_20260522_092005/`.
- YOLO11l 1280 ONNX export in `local_artifacts/yolo11l_1280_onnx_export_20260522_102315/`.
- Six selected prediction examples in `local_artifacts/outputs/nautilus/runs/paper_figures/yolo11l_1280_selected_examples/`.

## Still Worth Running

1. Faster R-CNN baseline on the current split.
2. Optional cross-dataset test if DeepPCB/DsPCBSD+/Mendeley YOLO data is ready.
3. Optional YOLO11l fine-tune only if more time/GPU is available.

## Repeatable Export Command

```bash
python tools/run_nautilus_experiments.py \
  --experiment detector_export \
  --data-root ~/data \
  --output-dir ~/outputs/nautilus \
  --yolo-weights ~/outputs/nautilus/runs/detector_train/yolo11l_1280_publication/weights/best.pt \
  --imgsz 1280 \
  --export-format onnx
```

## Repeatable Visual Examples Command

```bash
python tools/run_nautilus_experiments.py \
  --experiment visual_examples \
  --data-root ~/data \
  --output-dir ~/outputs/nautilus \
  --yolo-weights ~/outputs/nautilus/runs/detector_train/yolo11l_1280_publication/weights/best.pt \
  --run-name yolo11l_1280_selected_examples \
  --imgsz 1280 \
  --split test \
  --prediction-save-limit 6
```

## Faster R-CNN Command

```bash
cd ~/Capstone_project
mkdir -p ~/logs ~/backups

nohup bash -lc '
python tools/run_nautilus_experiments.py \
  --experiment faster_rcnn \
  --data-root ~/data \
  --output-dir ~/outputs/nautilus \
  --frcnn-epochs 5 \
  --frcnn-batch 2 \
  --workers 0

EXIT_CODE=$?

cd ~
zip -r backups/faster_rcnn_outputs_$(date +%Y%m%d_%H%M%S).zip \
  outputs/nautilus/runs/faster_rcnn \
  logs/faster_rcnn.log

echo "EXIT_CODE=$EXIT_CODE"
echo "Backup finished at $(date)"
exit $EXIT_CODE
' > ~/logs/faster_rcnn.log 2>&1 &
```

## Paper Priority

Start writing now using YOLO11l @ 1280 as the main detector. Use YOLO11m-960 and Kaggle fusion as ablations. Treat Faster R-CNN and cross-dataset as optional before ESCS (May 27).
