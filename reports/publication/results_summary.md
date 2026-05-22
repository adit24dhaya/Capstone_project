# ESCS'26 Publication Results Summary

## Current Main Result

The strongest completed model is **YOLO11m 960** on the test split with precision `0.9819`, recall `0.9648`, mAP50 `0.9787`, and mAP50-95 `0.5347`.

Compared with the Kaggle YOLO11s baseline, YOLO11m 960 improves test mAP50-95 from `0.5068` to `0.5347` for an absolute gain of `0.0279`.

## Paper Claim To Use

A medium YOLO detector trained at 960-pixel resolution improves PCB defect detection accuracy while remaining real-time on a commodity RTX 2080 Ti-class GPU. The broader system adds transformer-style and adaptive fusion experiments for embedded inspection tradeoff analysis.

## YOLO11m 960 Training Note

The Nautilus run used `imgsz=960`, `batch=2`, `workers=0`, and early stopped after 51 epochs.
The best epoch in `results.csv` by validation mAP50-95 was epoch `31` with mAP50-95 `0.5338`.

## Important Caveat

Kaggle v16 and Nautilus YOLO11m runs were executed on different GPU environments. Use accuracy metrics for model comparison, and discuss latency as hardware-specific.

## Evidence Files

- `reports/publication/model_comparison.csv`
- `reports/publication/yolo11m_960_training_summary.csv`
- Local artifact backup: `local_artifacts/yolo11m_960_publication_outputs_20260522_061024/`
