# Project Alignment

Project: Automated PCB Defect Detection Using Deep Learning

The proposal and survey describe an end-to-end PCB inspection system with these major goals:

- Detect and localize PCB defects including mouse bite, spur, open circuit, short, missing hole, and spurious copper.
- Improve small-defect performance using augmentation and multi-scale object detection.
- Use GPU training with PyTorch/Ultralytics YOLOv8.
- Evaluate with precision, recall, mAP, and inference latency.
- Target real-time inference below 100 ms per image.
- Export a deployment artifact using ONNX, with a TensorRT path for Jetson Orin deployment.
- Include a comparison path for CNN-only and transformer-style detectors.

Current implementation status:

- `kaggle_kernel/project.ipynb` prepares the Kaggle PCB dataset from Pascal VOC XML into YOLO format.
- It downloads and verifies DsPCBSD+ from Figshare DOI `10.6084/m9.figshare.24970329.v1`, then merges the overlapping real-image categories into the project taxonomy.
- It creates stratified `train`, `val`, and `test` splits by defect folder.
- It trains YOLOv8 on Kaggle GPU with small-defect-oriented augmentation.
- It evaluates both validation and held-out test metrics.
- It runs Albumentations-based robustness tests for brightness/contrast, Gaussian noise, motion blur, and rotation.
- It writes `project_metrics_summary.csv`, `architecture_comparison.csv` (YOLOv8n and RT-DETR-L rows for **val** and **test**), `robustness_metrics.csv`, `latency_summary.json`, `deployment_exports.json`, and `requirements_traceability.csv`.
- It exports the trained model to ONNX.
- It enables a `RUN_RTDETR_BENCHMARK` switch for a transformer-style RT-DETR comparison phase, addressing the CNN-vs-Transformer requirement from the survey.
- It includes an optional `RUN_TENSORRT_EXPORT` switch, with final Jetson TensorRT export expected to be rebuilt on the Jetson target.

Remaining high-value milestones:

- Run the RT-DETR/robustness version to completion and compare its `architecture_comparison.csv` and `robustness_metrics.csv` results against YOLOv8.
- Rotate the Kaggle access token because it was visible during setup.
- Build the TensorRT engine directly on Jetson Orin for the final deployment demo.
- Add final report plots/tables from the generated CSV/JSON artifacts.

Dataset-improvement note:

- DsPCBSD+ has nine categories. The current notebook uses overlap mode to add `SH`, `SP`, `SC`, `OP`, and `MB` as `Short`, `Spur`, `Spurious_copper`, `Open_circuit`, and `Mouse_bite`.
- DsPCBSD+ categories `HB`, `CS`, `CFO`, and `BMFO` are skipped for now because they are not one-to-one matches for the six classes listed in the proposal.
- `Missing_hole` remains sourced from the original Kaggle PCB dataset.
