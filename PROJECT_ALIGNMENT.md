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
- It creates stratified `train`, `val`, and `test` splits by defect folder.
- It trains YOLOv8 on Kaggle GPU with small-defect-oriented augmentation.
- It evaluates both validation and held-out test metrics.
- It writes `project_metrics_summary.csv`, `latency_summary.json`, `deployment_exports.json`, and `requirements_traceability.csv`.
- It exports the trained model to ONNX.
- It includes an optional `RUN_RTDETR_BENCHMARK` switch for a transformer-style RT-DETR comparison phase.
- It includes an optional `RUN_TENSORRT_EXPORT` switch, with final Jetson TensorRT export expected to be rebuilt on the Jetson target.

Remaining high-value milestones:

- Run the Kaggle GPU notebook to completion and capture final metrics.
- If time allows, enable `RUN_RTDETR_BENCHMARK=True` for the transformer comparison.
- Rotate the Kaggle access token because it was visible during setup.
- Build the TensorRT engine directly on Jetson Orin for the final deployment demo.
- Add final report plots/tables from the generated CSV/JSON artifacts.
