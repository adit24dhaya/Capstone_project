# Final Experiment Checklist

## Completed Evidence

- YOLO11s baseline from Kaggle v16.
- RT-DETR-L transformer-style baseline from Kaggle v16.
- Hybrid YOLO11s + RT-DETR-L fusion experiments from Kaggle v16.
- Adaptive defect-aware fusion, defect-size, calibration, robustness, and inspection-cost artifacts from Kaggle v16.
- YOLO11m 960 Nautilus/RTX 2080 Ti result with saved `best.pt`, logs, curves, and metrics.

## Still Worth Running

1. Faster R-CNN baseline on the current split.
2. Optional YOLO11m 960 ONNX export for deployment evidence.
3. Optional final visualization set using YOLO11m 960 predictions.
4. Optional cross-dataset test if DeepPCB/DsPCBSD+/Mendeley YOLO data is ready.

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

Start writing now using YOLO11m 960 as the main model and fusion/cost analysis as the novelty layer. Treat Faster R-CNN and cross-dataset results as additions if they finish before submission.
