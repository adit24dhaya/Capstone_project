# Automated PCB Defect Detection Using Deep Learning

This project implements an end-to-end PCB defect detection workflow aligned with the capstone proposal and survey paper. It uses Kaggle GPU to prepare the PCB defect dataset, train an object detector, evaluate defect localization/classification metrics, measure inference latency, and export deployment artifacts.

## Project Goals

- Detect six PCB defect classes: mouse bite, spur, open circuit, short, missing hole, and spurious copper.
- Train a deep learning detector using GPU acceleration.
- Evaluate precision, recall, mAP50, mAP50-95, and inference latency.
- Benchmark a transformer-style RT-DETR detector against the YOLOv8 CNN baseline.
- Add synthetic defect augmentation for rare/weak classes as a practical TransGAN-style class-balancing surrogate.
- Evaluate a custom CNN-Transformer feature refiner on top of the hybrid detector.
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
- `synthetic_augmentation_summary.csv` (generated synthetic defect counts by class)
- `hybrid_tuning_grid.csv` (validation-only sweep over confidence, fusion IoU, fusion mode, and class-aware NMS)
- `hybrid_selected_config.json` (selected tuned hybrid configuration)
- `hybrid_selected_test_metrics.csv` (test metrics for the selected validation-tuned hybrid)
- `hybrid_selected_per_class_metrics.csv` (per-class test metrics for the selected validation-tuned hybrid)
- `hybrid_fusion_metrics.csv` (Hybrid YOLOv8n + RT-DETR-L late-fusion val/test metrics)
- `hybrid_per_class_metrics.csv` (hybrid per-class precision/recall/mAP)
- `hybrid_robustness_metrics.csv` (selected hybrid robustness under clean, lighting, noise, blur, and rotation conditions)
- `hybrid_robustness_per_class_metrics.csv` (hybrid robustness broken down by class)
- `hybrid_error_analysis.csv`, `hybrid_error_examples.csv`, and `hybrid_class_delta.csv`
- `cnn_transformer_refiner_training.csv` (custom CNN-Transformer feature refiner training history)
- `cnn_transformer_refined_hybrid_metrics.csv` (refined hybrid val/test metrics)
- `cnn_transformer_refined_hybrid_per_class.csv` (refined hybrid per-class metrics)
- `robustness_metrics.csv`
- `per_class_metrics.csv`
- `latency_summary.json` (keys `yolov8n`, `rtdetr_l`, `hybrid_yolo_rtdetr_fusion`, and `cnn_transformer_refined_hybrid` when the corresponding phases run)
- `latency_comparison.csv`
- `final_results_summary.csv`
- `deployment_exports.json`
- `jetson_deployment_status.json` (ONNX/TensorRT status and Jetson build note)
- `requirements_traceability.csv`
- prediction visualization folders, including `hybrid_visual_evidence`

## Current Benchmark Notes

- The implemented system uses `YOLOv8n` as a CNN real-time baseline and `RT-DETR-L` as a transformer-style benchmark to evaluate CNN-vs-transformer tradeoffs.
- The notebook also includes a tuned **Hybrid YOLO-Transformer late-fusion pipeline** (decision-level fusion), combining YOLO and RT-DETR predictions class-wise with validation-selected confidence, IoU, fusion mode, and class-aware NMS settings.
- The next run adds synthetic copy-paste defect augmentation and a custom CNN-Transformer patch refiner to strengthen the survey-paper claims around augmentation and feature-level CNN-Transformer modeling.
- TensorRT export is documented as a Jetson-target deployment step because TensorRT engines are hardware/runtime specific; ONNX export is completed in Kaggle.
- In the latest completed benchmark run, RT-DETR-L improved test mAP50 from 0.853 to 0.879 and mAP50-95 from 0.442 to 0.488, while remaining under the 100 ms mean latency target.
- Gaussian noise remains the weakest robustness condition and is currently treated as a known limitation for future training-time augmentation improvements.

## Security Note

Do not commit Kaggle credentials or API tokens. This repository ignores common credential filenames.
