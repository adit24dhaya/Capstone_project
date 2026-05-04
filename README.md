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
- `architecture_comparison.csv`
- `robustness_metrics.csv`
- `latency_summary.json`
- `deployment_exports.json`
- `requirements_traceability.csv`
- prediction visualization folders

## Security Note

Do not commit Kaggle credentials or API tokens. This repository ignores common credential filenames.
