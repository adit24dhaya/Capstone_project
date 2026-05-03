# Automated PCB Defect Detection Using Deep Learning

This project implements an end-to-end PCB defect detection workflow aligned with the capstone proposal and survey paper. It uses Kaggle GPU to prepare the PCB defect dataset, train an object detector, evaluate defect localization/classification metrics, measure inference latency, and export deployment artifacts.

## Project Goals

- Detect six PCB defect classes: mouse bite, spur, open circuit, short, missing hole, and spurious copper.
- Train a deep learning detector using GPU acceleration.
- Evaluate precision, recall, mAP50, mAP50-95, and inference latency.
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

Local dataset downloads are intentionally ignored by git because they are large.

## Kaggle Outputs

After a successful Kaggle run, the important generated files are expected in `/kaggle/working`:

- `project_metrics_summary.csv`
- `latency_summary.json`
- `deployment_exports.json`
- `requirements_traceability.csv`
- prediction visualization folders

## Security Note

Do not commit Kaggle credentials or API tokens. This repository ignores common credential filenames.
