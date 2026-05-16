# Automated PCB Defect Detection Using Deep Learning

This project implements an end-to-end PCB defect detection workflow aligned with the capstone proposal and survey paper. It uses Kaggle GPU to prepare the PCB defect dataset, train an object detector, evaluate defect localization/classification metrics, measure inference latency, and export deployment artifacts.

## Project Goals

- Detect six PCB defect classes: mouse bite, spur, open circuit, short, missing hole, and spurious copper.
- Train a deep learning detector using GPU acceleration.
- Evaluate precision, recall, mAP50, mAP50-95, and inference latency.
- Benchmark a transformer-style RT-DETR detector against the YOLO11s CNN baseline.
- Add publication-oriented analyses: defect size performance, adaptive defect-aware hybrid fusion, industrial FP/FN inspection burden, and calibration/reliability.
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
- `tools/run_nautilus_experiments.py` - portable Nautilus/Kubernetes runner for publication experiments.
- `k8s/` - Nautilus Kubernetes Job templates for long GPU runs.
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
- `architecture_comparison.csv` (YOLO11s and RT-DETR-L, val and test rows when RT-DETR benchmark runs)
- `synthetic_augmentation_summary.csv` (generated synthetic defect counts by class)
- `hybrid_tuning_grid.csv` (validation-only sweep over confidence, fusion IoU, fusion mode, and class-aware NMS)
- `hybrid_tuning_grid_v2.csv` (stricter validation-only sweep with F0.5, false-positive rate, single-model fallback confidence, per-class thresholds, and class-weighted fusion)
- `hybrid_pareto_frontier.csv` (v11: non-dominated tuning configs on precision, recall, mAP50-95, false positives per image)
- `hybrid_selected_profiles_config.json` (v11: three validation-only profiles — high_recall, balanced, high_precision — plus full hybrid configs)
- `hybrid_selected_profiles_test_metrics.csv` (v11: one test evaluation per profile; v10 `hybrid_selected_*` paths unchanged)
- `hybrid_selected_config.json` (selected tuned hybrid configuration)
- `hybrid_selected_config_v2.json` (selected precision-focused hybrid configuration)
- `hybrid_selected_test_metrics.csv` (test metrics for the selected validation-tuned hybrid)
- `hybrid_selected_test_metrics_v2.csv` (test metrics for the selected precision-focused hybrid)
- `hybrid_selected_per_class_metrics.csv` (per-class test metrics for the selected validation-tuned hybrid)
- `hybrid_fusion_metrics.csv` (Hybrid YOLO11s + RT-DETR-L late-fusion val/test metrics)
- `hybrid_per_class_metrics.csv` (hybrid per-class precision/recall/mAP)
- `hybrid_robustness_metrics.csv` (selected hybrid robustness under clean, lighting, noise, blur, and rotation conditions)
- `hybrid_robustness_per_class_metrics.csv` (hybrid robustness broken down by class)
- `hybrid_error_analysis.csv`, `hybrid_error_analysis_v2.csv`, `hybrid_error_examples.csv`, and `hybrid_class_delta.csv`
- `cnn_transformer_refiner_training.csv` (custom CNN-Transformer feature refiner training history)
- `cnn_transformer_refined_hybrid_metrics.csv` (refined hybrid val/test metrics)
- `cnn_transformer_refined_hybrid_per_class.csv` (refined hybrid per-class metrics)
- `hybrid_final_balanced_test_metrics.csv` and `hybrid_final_balanced_per_class_metrics.csv`
- `defect_size_analysis.csv` (tiny/small/medium/large defect metrics)
- `adaptive_defect_aware_policy.json` and `adaptive_defect_aware_policy.csv`
- `adaptive_defect_aware_hybrid_test_metrics.csv` and `adaptive_defect_aware_hybrid_per_class_metrics.csv`
- `industrial_inspection_cost_metrics.csv` (FP/image + missed-defect penalty * FN/image)
- `calibration_metrics.csv`, `calibration_reliability_bins.csv`, and `calibration_reliability_plot.png`
- `robustness_metrics.csv`
- `per_class_metrics.csv`
- `latency_summary.json` (keys `yolo11s`, `rtdetr_l`, `hybrid_yolo_rtdetr_fusion`, and `cnn_transformer_refined_hybrid` when the corresponding phases run)
- `latency_comparison.csv`
- `final_results_summary.csv`
- `deployment_exports.json`
- `jetson_deployment_status.json` (ONNX/TensorRT status and Jetson build note)
- `requirements_traceability.csv`
- prediction visualization folders, including `hybrid_visual_evidence`

## Current Benchmark Notes

- The implemented system uses `YOLO11s` as the CNN baseline and `RT-DETR-L` as the transformer-style benchmark.
- The notebook includes tuned **Hybrid YOLO-Transformer late-fusion** profiles plus a final WBF-based balanced hybrid.
- The latest completed benchmark run produced: YOLO11s test mAP50-95 `0.4996`, RT-DETR-L test mAP50-95 `0.4901`, agreement-only hybrid precision `0.9074` with `0.216` FP/image, and final balanced WBF hybrid recall `0.9370` with mAP50-95 `0.5077`.
- The final publication-analysis path adds adaptive defect-aware fusion, defect-size metrics, inspection-burden metrics, and confidence calibration outputs.
- Synthetic copy-paste defect augmentation and the custom CNN-Transformer patch refiner are implemented to support the survey-paper discussion around augmentation and feature-level CNN-Transformer modeling. The refiner is treated as an experimental extension because it underperforms the detector-level hybrid.
- TensorRT export is documented as a Jetson-target deployment step because TensorRT engines are hardware/runtime specific; ONNX export is completed in Kaggle.

## Publication Experiment Path

The capstone notebook remains the stable final workflow. For publication work, use the Nautilus runner to add stronger baselines and generalization evidence.

Recommended next experiments:

1. Train stronger YOLO baselines: `YOLO11m` and, on larger GPUs, `YOLO11l`.
2. Evaluate stronger transformer baselines such as `RT-DETR-X` when GPU memory allows.
3. Keep Faster R-CNN as a two-stage high-recall baseline.
4. Convert external COCO-style datasets such as DsPCBSD+ into the six-class project taxonomy.
5. Run defect-size, calibration, and industrial FP/FN cost analyses.
6. Run adaptive defect-aware fusion when both YOLO and RT-DETR weights are available.

Example Nautilus commands:

```bash
python tools/run_nautilus_experiments.py \
  --experiment detector_train \
  --data-root ~/data \
  --output-dir ~/outputs/nautilus \
  --yolo-model yolo11m.pt \
  --run-name yolo11m_publication \
  --imgsz 1280 \
  --batch 4 \
  --epochs 100 \
  --workers 0

python tools/run_nautilus_experiments.py \
  --experiment detector_eval \
  --data-root ~/data \
  --output-dir ~/outputs/nautilus \
  --yolo-weights ~/outputs/nautilus/runs/detector_train/yolo11m_publication/weights/best.pt \
  --imgsz 1280 \
  --split test \
  --workers 0
```

For long runs, prefer Kubernetes Jobs from `k8s/` rather than browser-based `nohup`.

## Security Note

Do not commit Kaggle credentials or API tokens. This repository ignores common credential filenames.
