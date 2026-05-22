# ESCS'26 Paper Outline

Working title:
Cost-Aware Adaptive Detector Fusion for Real-Time Embedded PCB Defect Inspection

## Abstract

Write 100-120 words. Mention PCB defect detection, embedded inspection, YOLO11m 960, RT-DETR, Faster R-CNN baseline, adaptive fusion, real-time latency, and inspection-cost metrics.

## 1. Introduction

- PCB manufacturing needs fast, reliable surface-defect inspection.
- False negatives are costly because missed defects can propagate downstream.
- Embedded deployment requires balancing accuracy, latency, and inspection burden.
- Contribution summary:
  - Strong YOLO11m 960 real-time detector baseline.
  - Comparison against YOLO11s, RT-DETR-L, and fusion variants.
  - Cost-aware and adaptive fusion analysis for industrial inspection.
  - Deployment-oriented discussion with ONNX/TensorRT readiness.

## 2. Related Work

- Classical PCB visual inspection.
- YOLO-family PCB defect detectors.
- Transformer-based object detectors for industrial inspection.
- Ensemble/fusion and calibration for dependable inspection.

## 3. Method

- Dataset and six-class taxonomy.
- YOLO conversion and train/val/test split.
- Detector baselines: YOLO11s, YOLO11m 960, RT-DETR-L, Faster R-CNN.
- Adaptive fusion policy and inspection-cost metric.
- Evaluation metrics: precision, recall, mAP50, mAP50-95, FP/image, latency.

## 4. Experiments

- Hardware and environment table.
- Training settings table.
- Model comparison table from `model_comparison.csv`.
- Per-class table for YOLO11m 960 and fusion models.
- Latency and deployment analysis.

## 5. Results and Discussion

- YOLO11m 960 is the strongest current accuracy result.
- Fusion improves the precision/recall tradeoff discussion but is not always the highest mAP model.
- Missing_hole remains easiest; Short/Spur localization is harder under stricter mAP50-95.
- Discuss 2080 Ti feasibility and why A100 is requested for final high-resolution ablations.

## 6. Limitations

- Single primary PCB dataset unless cross-dataset conversion is completed.
- Pseudo segmentation masks are box-derived, not true pixel masks.
- Latency differs across Kaggle and Nautilus hardware.

## 7. Conclusion

- Summarize real-time embedded inspection result and practical cost-aware evaluation.
