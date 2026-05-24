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

## Champion Evaluation Sweep

On May 24, 2026, the restored YOLO11l 1280 champion checkpoint was re-evaluated on Nautilus V100 at 1280 and 1536 image sizes, with and without Ultralytics test-time augmentation (`augment=True`). The best test mAP50-95 remained the original 1280/no-TTA setting: precision `0.9832`, recall `0.9881`, mAP50 `0.9898`, mAP50-95 `0.5769`.

The sweep is useful for the paper because it shows that simply increasing inference size to 1536 or enabling TTA did not improve strict localization. Use 1280/no-TTA as the headline configuration, and mention the sweep as a negative ablation supporting the chosen deployment setting.

## Important Caveat

Kaggle v16 and Nautilus YOLO11 runs were executed on different GPU environments. Use accuracy metrics for model comparison, and discuss latency as hardware-specific.

## Evidence Files

- `reports/publication/model_comparison.csv`
- `reports/publication/champion_eval_sweep.csv`
- `reports/publication/yolo11l_1280_training_summary.csv`
- `reports/publication/yolo11m_960_training_summary.csv`
- Local artifact backup: `local_artifacts/yolo11l_1280_publication_outputs_20260522_092005/`
- ONNX deployment artifact: `local_artifacts/yolo11l_1280_onnx_export_20260522_102315/outputs/nautilus/runs/detector_train/yolo11l_1280_publication/weights/best.onnx`
- Selected prediction figures: `local_artifacts/outputs/nautilus/runs/paper_figures/yolo11l_1280_selected_examples/`
- Prior ablation: `local_artifacts/yolo11m_960_publication_outputs_20260522_061024/`
