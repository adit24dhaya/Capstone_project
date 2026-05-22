# ESCS'26 Publication Results Summary

## Current Main Result

The strongest completed model is **YOLO11l 1280** on the test split with precision `0.9832`, recall `0.9881`, mAP50 `0.9898`, and mAP50-95 `0.5769`.

Compared with the Kaggle YOLO11s baseline (test mAP50-95 `0.5068`), the gain is `0.0701` absolute.
Compared with YOLO11m-960 (test mAP50-95 `0.5347`), the gain is `0.0422` absolute.

## Paper Claim To Use

A large YOLO detector at 1280px achieves strong six-class PCB defect detection on a commodity RTX 2080 Ti (test mAP50 near 0.99, mAP50-95 near 0.58). YOLO11m-960 and Kaggle fusion experiments support ablation and deployment tradeoffs.

## YOLO11l 1280 Training Note

Nautilus RTX 2080 Ti; see `results.csv` and overnight log in local artifact backup.
The best epoch in `results.csv` by validation mAP50-95 was epoch `64` with mAP50-95 `0.5491`.

## Important Caveat

Kaggle v16 and Nautilus YOLO11 runs were executed on different GPU environments. Use accuracy metrics for model comparison, and discuss latency as hardware-specific.

## Evidence Files

- `reports/publication/model_comparison.csv`
- `reports/publication/yolo11l_1280_training_summary.csv`
- `reports/publication/yolo11m_960_training_summary.csv`
- Local artifact backup: `local_artifacts/yolo11l_1280_publication_outputs_20260522_092005/`
- ONNX deployment artifact: `local_artifacts/yolo11l_1280_onnx_export_20260522_102315/outputs/nautilus/runs/detector_train/yolo11l_1280_publication/weights/best.onnx`
- Selected prediction figures: `local_artifacts/outputs/nautilus/runs/paper_figures/yolo11l_1280_selected_examples/`
- Prior ablation: `local_artifacts/yolo11m_960_publication_outputs_20260522_061024/`
