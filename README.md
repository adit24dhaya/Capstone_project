# Automated PCB Defect Detection Using Deep Learning

This project implements an end-to-end PCB defect detection workflow aligned with the capstone proposal and survey paper. It uses Kaggle GPU to prepare the PCB defect dataset, train an object detector, evaluate defect localization/classification metrics, measure inference latency, and export deployment artifacts.

## Project Goals

- Detect six PCB defect classes: mouse bite, spur, open circuit, short, missing hole, and spurious copper.
- Train a deep learning detector using GPU acceleration.
- Evaluate precision, recall, mAP50, mAP50-95, and inference latency.
- Benchmark a transformer-style RT-DETR detector against the YOLOv8 CNN baseline.
- Test robustness with Albumentations image transformations.
- Target real-time inference below 100 ms per image.
- Export ONNX artifacts, with an optional TensorRT path for Jetson Orin deployment.
- Provide visual prediction outputs for inspection and reporting.

## Main Files

- `project.ipynb` - Kaggle GPU-ready notebook.
- `kaggle_kernel/project.ipynb` - uploadable Kaggle kernel copy.
- `kaggle_kernel/kernel-metadata.json` - Kaggle kernel metadata with GPU and dataset settings.
- `tools/make_gpu_notebook.py` - generator for rebuilding the Kaggle notebook.
- `PROJECT_ALIGNMENT.md` - mapping between proposal requirements and implementation.

## Dataset

The notebook uses the Kaggle dataset:

```text
aditya2402/pcb-dataset
```

It also downloads DsPCBSD+ from Figshare during the Kaggle run when internet is enabled:

```text
DOI: 10.6084/m9.figshare.24970329.v1
File: DsPCBSD+.zip
```

DsPCBSD+ is merged in overlap mode so it strengthens the five classes that match the current six-class project taxonomy:

- `SH` -> `Short`
- `SP` -> `Spur`
- `SC` -> `Spurious_copper`
- `OP` -> `Open_circuit`
- `MB` -> `Mouse_bite`

The remaining DsPCBSD+ categories are skipped by default because they do not directly match the proposal's current classes. The original Kaggle PCB dataset still provides the `Missing_hole` class.

Local dataset downloads are intentionally ignored by git because they are large.

## Kaggle Outputs

After a successful Kaggle run, the important generated files are expected in `/kaggle/working`:

- `project_metrics_summary.csv`
- `architecture_comparison.csv` (YOLOv8n and RT-DETR-L, val and test rows when RT-DETR benchmark runs)
- `hybrid_fusion_metrics.csv` (Hybrid YOLOv8n + RT-DETR-L late-fusion val/test metrics)
- `hybrid_per_class_metrics.csv` (hybrid per-class precision/recall/mAP)
- `robustness_metrics.csv`
- `per_class_metrics.csv`
- `latency_summary.json` (keys `yolov8n`, and `rtdetr_l` when the RT-DETR benchmark cell ran in the same session)
- `latency_comparison.csv`
- `final_results_summary.csv`
- `deployment_exports.json`
- `requirements_traceability.csv`
- prediction visualization folders

## Current Benchmark Notes

- The implemented system uses `YOLOv8n` as a CNN real-time baseline and `RT-DETR-L` as a transformer-style benchmark to evaluate CNN-vs-transformer tradeoffs.
- The notebook also includes a **Hybrid YOLO-Transformer late-fusion pipeline** (decision-level fusion), combining YOLO and RT-DETR predictions class-wise with IoU-guided fusion and class-aware NMS.
- In the latest benchmark run, RT-DETR-L improved test mAP50 from 0.853 to 0.878 and mAP50-95 from 0.442 to 0.479, while remaining under the 100 ms mean latency target.
- Gaussian noise remains the weakest robustness condition and is currently treated as a known limitation for future training-time augmentation improvements.

## Security Note

Do not commit Kaggle credentials or API tokens. This repository ignores common credential filenames.
