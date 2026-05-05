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
- It generates copy-paste synthetic defect images for weak/rare classes as a practical TransGAN-style class-balancing surrogate, while leaving validation and test data untouched.
- It trains YOLOv8 on Kaggle GPU with small-defect-oriented augmentation.
- It evaluates both validation and held-out test metrics.
- It runs Albumentations-based robustness tests for brightness/contrast, Gaussian noise, motion blur, and rotation.
- It writes `project_metrics_summary.csv`, `architecture_comparison.csv` (YOLOv8n, RT-DETR-L, Hybrid, and CNN-Transformer refined hybrid rows for **val** and **test**), `synthetic_augmentation_summary.csv`, `robustness_metrics.csv`, `per_class_metrics.csv`, `hybrid_tuning_grid.csv`, `hybrid_tuning_grid_v2.csv`, `hybrid_selected_config.json`, `hybrid_selected_config_v2.json`, `hybrid_selected_test_metrics.csv`, `hybrid_selected_test_metrics_v2.csv`, `hybrid_selected_per_class_metrics.csv`, `hybrid_fusion_metrics.csv`, `hybrid_per_class_metrics.csv`, `hybrid_robustness_metrics.csv`, `hybrid_robustness_per_class_metrics.csv`, `hybrid_error_analysis.csv`, `hybrid_error_analysis_v2.csv`, `hybrid_error_examples.csv`, `hybrid_class_delta.csv`, `cnn_transformer_refiner_training.csv`, `cnn_transformer_refined_hybrid_metrics.csv`, `cnn_transformer_refined_hybrid_per_class.csv`, `latency_summary.json`, `latency_comparison.csv`, `final_results_summary.csv`, `deployment_exports.json`, `jetson_deployment_status.json`, and `requirements_traceability.csv`.
- It exports the trained model to ONNX.
- It enables a `RUN_RTDETR_BENCHMARK` switch for a transformer-style RT-DETR comparison phase, addressing the CNN-vs-Transformer requirement from the survey.
- It tunes hybrid fusion on validation data only, including a stricter v2 sweep over F0.5, precision floors, false positives per image, single-model high-confidence fallback, per-class confidence thresholds, and class-weighted YOLO/RT-DETR fusion; it evaluates the selected configuration once on test data, runs selected-hybrid robustness tests, and saves visual/error-analysis artifacts.
- It includes a custom CNN-Transformer patch-level feature refiner that learns from defect/background crops and filters/refines fused detector outputs.
- It includes an optional `RUN_TENSORRT_EXPORT` switch, with final Jetson TensorRT export expected to be rebuilt on the Jetson target.

Remaining high-value milestones:

- Run the RT-DETR/robustness version to completion and compare its `architecture_comparison.csv` and `robustness_metrics.csv` results against YOLOv8.
- Run the precision-focused hybrid v2 version to completion and compare `hybrid_selected_test_metrics_v2.csv`, `hybrid_robustness_metrics.csv`, and `hybrid_error_analysis_v2.csv` against the version 9 late-fusion hybrid.
- Rotate the Kaggle access token because it was visible during setup.
- Build the TensorRT engine directly on Jetson Orin for the final deployment demo, or report `jetson_deployment_status.json` as evidence that the deployment path is prepared and hardware-specific TensorRT build remains target-side.
- Add final report plots/tables from the generated CSV/JSON artifacts.

Dataset-improvement note:

- DsPCBSD+ has nine categories. The current notebook uses overlap mode to add `SH`, `SP`, `SC`, `OP`, and `MB` as `Short`, `Spur`, `Spurious_copper`, `Open_circuit`, and `Mouse_bite`.
- DsPCBSD+ categories `HB`, `CS`, `CFO`, and `BMFO` are skipped for now because they are not one-to-one matches for the six classes listed in the proposal.
- `Missing_hole` remains sourced from the original Kaggle PCB dataset.

Architecture wording note:

- The implementation includes three architecture levels: *YOLOv8n CNN baseline*, *RT-DETR-L transformer-style benchmark*, and *YOLO-Transformer late fusion*.
- The current phase keeps the late-fusion hybrid as the final candidate and treats the custom feature-level CNN-Transformer patch refiner as an experimental learned module unless its metrics beat the tuned hybrid. This is still not a fully custom end-to-end detector backbone.
- In the latest completed benchmark run, the tuned hybrid reached test precision 0.625, recall 0.928, mAP50 0.883, and mAP50-95 0.480. RT-DETR-L remains the high-precision reference, so the v2 sweep focuses on reducing hybrid false positives.
- Gaussian noise remains the weakest robustness condition and is tracked as a current limitation.
