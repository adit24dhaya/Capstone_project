# Generated Resolution-Control Results

## Accepted Practical Configurations - Test

| Model | Precision | Recall | F1 | mAP50 | mAP50-95 | Inference (ms) | Total (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11s 1280 | 0.880 | 0.866 | 0.873 | 0.902 | 0.502 | 12.8 | 15.2 |
| RT-DETR-L 640 | 0.886 | 0.839 | 0.862 | 0.887 | 0.470 | 48.8 | 49.8 |

## Accepted Practical Configurations - Validation

| Model | Precision | Recall | F1 | mAP50 | mAP50-95 | Inference (ms) | Total (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11s 1280 | 0.874 | 0.874 | 0.874 | 0.908 | 0.505 | 13.1 | 15.7 |
| RT-DETR-L 640 | 0.884 | 0.855 | 0.869 | 0.905 | 0.472 | 48.6 | 49.7 |

## Evaluation-Only 640 Diagnostic - Test

| Model | Precision | Recall | F1 | mAP50 | mAP50-95 | Inference (ms) | Total (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11s checkpoint diagnostic 640 | 0.874 | 0.810 | 0.841 | 0.855 | 0.453 | 12.1 | 13.8 |

## Controlled and High-Resolution Runs - Test

| Model | Precision | Recall | F1 | mAP50 | mAP50-95 | Inference (ms) | Total (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR-L 640 | 0.867 | 0.839 | 0.853 | 0.881 | 0.462 | 48.0 | 49.0 |
| RT-DETR-L 1280 | 0.766 | 0.534 | 0.630 | 0.604 | 0.295 | 72.4 | 74.4 |
| YOLO11s 640 | 0.870 | 0.852 | 0.861 | 0.892 | 0.482 | 11.4 | 13.2 |

## Controlled and High-Resolution Runs - Validation

| Model | Precision | Recall | F1 | mAP50 | mAP50-95 | Inference (ms) | Total (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| RT-DETR-L 640 | 0.861 | 0.848 | 0.854 | 0.893 | 0.470 | 48.6 | 49.7 |
| RT-DETR-L 1280 | 0.724 | 0.545 | 0.622 | 0.615 | 0.307 | 72.8 | 74.8 |
| YOLO11s 640 | 0.888 | 0.860 | 0.874 | 0.902 | 0.488 | 11.9 | 13.7 |

## Matched-Resolution Per-Class Results - Test

| Class | Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| Missing_hole | YOLO11s 640 | 0.990 | 0.986 | 0.994 | 0.569 |
| Missing_hole | RT-DETR-L 640 | 0.962 | 1.000 | 0.995 | 0.567 |
| Mouse_bite | YOLO11s 640 | 0.821 | 0.763 | 0.835 | 0.403 |
| Mouse_bite | RT-DETR-L 640 | 0.818 | 0.765 | 0.829 | 0.403 |
| Open_circuit | YOLO11s 640 | 0.848 | 0.850 | 0.890 | 0.525 |
| Open_circuit | RT-DETR-L 640 | 0.882 | 0.880 | 0.917 | 0.513 |
| Short | YOLO11s 640 | 0.880 | 0.892 | 0.942 | 0.551 |
| Short | RT-DETR-L 640 | 0.881 | 0.865 | 0.900 | 0.505 |
| Spur | YOLO11s 640 | 0.869 | 0.767 | 0.839 | 0.381 |
| Spur | RT-DETR-L 640 | 0.818 | 0.729 | 0.794 | 0.342 |
| Spurious_copper | YOLO11s 640 | 0.815 | 0.853 | 0.852 | 0.462 |
| Spurious_copper | RT-DETR-L 640 | 0.844 | 0.795 | 0.852 | 0.443 |
