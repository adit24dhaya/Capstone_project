# Pre-Hybrid Architecture Comparison Report

> Note: This document captures the **pre-hybrid** baseline comparison. The active notebook now includes a **Hybrid YOLO-Transformer late-fusion pipeline** with dedicated outputs (`hybrid_fusion_metrics.csv`, `hybrid_per_class_metrics.csv`) for the next-phase evaluation.

This section compares the two detector architectures evaluated before adding the hybrid fusion stage. The purpose of this comparison is to establish a clear baseline between a real-time CNN-based detector and a transformer-style detector on the same PCB defect dataset, train/validation/test split, and six-class taxonomy.

## Models Compared

| Model | Role in Project | Architecture Type | Purpose |
|---|---|---|---|
| YOLOv8n | Baseline detector | CNN / one-stage YOLO detector | Fast real-time PCB defect localization |
| RT-DETR-L | Transformer benchmark | Transformer-style detector | Higher-capacity comparison model for accuracy tradeoff analysis |

YOLOv8n was selected as the real-time baseline because it is lightweight and suitable for deployment-oriented inspection systems. RT-DETR-L was added as the transformer-style benchmark to evaluate whether a transformer detector improves PCB defect detection accuracy, especially for small or visually subtle defect categories.

## Dataset Used

The experiment used a merged PCB defect dataset containing images from the original Kaggle PCB dataset and overlapping categories from DsPCBSD+. The final dataset included six project classes:

| Class ID | Defect Class |
|---:|---|
| 0 | Mouse_bite |
| 1 | Spur |
| 2 | Open_circuit |
| 3 | Short |
| 4 | Missing_hole |
| 5 | Spurious_copper |

The final split used in Kaggle version 7 was:

| Split | Images |
|---|---:|
| Train | 4,751 |
| Validation | 1,016 |
| Test | 1,016 |
| Total | 6,783 |

## Overall Validation and Test Results

| Model | Split | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| YOLOv8n | Validation | 0.861 | 0.808 | 0.871 | 0.457 |
| YOLOv8n | Test | 0.836 | 0.796 | 0.853 | 0.442 |
| RT-DETR-L | Validation | 0.885 | 0.860 | 0.883 | 0.472 |
| RT-DETR-L | Test | 0.886 | 0.849 | 0.881 | 0.475 |

RT-DETR-L achieved better accuracy than YOLOv8n on both validation and test splits. On the held-out test set, RT-DETR-L improved mAP50 from 0.853 to 0.881 and mAP50-95 from 0.442 to 0.475. This shows that the transformer-style detector provides stronger localization performance, particularly under the stricter mAP50-95 metric.

## Standard Deep Learning Outputs for Comparison

For a complete deep learning model comparison, the report should include both numerical metrics and visual training/evaluation outputs. The Kaggle version 7 run produces the core outputs normally expected in object detection experiments.

| Output Type | Purpose in Comparison | Current Project Artifact / Source |
|---|---|---|
| Dataset split summary | Shows that both models were evaluated on the same train, validation, and test split | Kaggle log and `YOLO_PCB/data.yaml` |
| Training curves | Shows convergence behavior and overfitting/underfitting trends | `runs/detect/train/results.csv` and generated loss/mAP plots |
| Precision and recall | Measures detection correctness and missed defects | `project_metrics_summary.csv`, `architecture_comparison.csv` |
| mAP50 | Measures object detection accuracy at IoU 0.50 | `project_metrics_summary.csv`, `architecture_comparison.csv` |
| mAP50-95 | Measures stricter localization quality across IoU thresholds | `project_metrics_summary.csv`, `architecture_comparison.csv` |
| Per-class metrics | Identifies which defect categories are strong or weak | `per_class_metrics.csv` |
| Confusion matrix | Shows class-level misclassification patterns | YOLO training outputs in `runs/detect/train` |
| Precision-recall / F1 curves | Shows confidence-threshold behavior | YOLO training outputs in `runs/detect/train` |
| Sample predictions | Provides visual proof of localization quality | `vis_predictions/val_preds` |
| Robustness tests | Shows performance under imaging variation | `robustness_metrics.csv` |
| Latency / FPS | Shows real-time deployment suitability | `latency_summary.json`, `latency_comparison.csv` |
| Deployment export | Shows deployability of the trained model | `deployment_exports.json`, `best.onnx` |

These outputs are important because accuracy alone is not enough for a PCB inspection system. A practical defect detector must also show stable training, class-wise behavior, robustness under visual variation, inference speed, and deployability. Therefore, the comparison uses both quantitative results and visual inspection artifacts.

## Test Set Improvement From YOLOv8n to RT-DETR-L

| Metric | YOLOv8n Test | RT-DETR-L Test | Absolute Improvement |
|---|---:|---:|---:|
| Precision | 0.836 | 0.886 | +0.050 |
| Recall | 0.796 | 0.849 | +0.053 |
| mAP50 | 0.853 | 0.881 | +0.028 |
| mAP50-95 | 0.442 | 0.475 | +0.033 |

The transformer-style model improved both precision and recall, meaning it produced fewer false detections while also detecting more true defects. The improvement in mAP50-95 is especially important because it evaluates localization quality across stricter IoU thresholds.

## Per-Class Test Results

| Model | Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|
| YOLOv8n | Mouse_bite | 0.828 | 0.715 | 0.799 | 0.373 |
| YOLOv8n | Spur | 0.839 | 0.729 | 0.825 | 0.368 |
| YOLOv8n | Open_circuit | 0.793 | 0.774 | 0.820 | 0.460 |
| YOLOv8n | Short | 0.770 | 0.842 | 0.868 | 0.477 |
| YOLOv8n | Missing_hole | 1.000 | 0.948 | 0.987 | 0.550 |
| YOLOv8n | Spurious_copper | 0.783 | 0.765 | 0.819 | 0.428 |
| RT-DETR-L | Mouse_bite | 0.827 | 0.802 | 0.841 | 0.410 |
| RT-DETR-L | Spur | 0.860 | 0.729 | 0.821 | 0.362 |
| RT-DETR-L | Open_circuit | 0.888 | 0.894 | 0.894 | 0.521 |
| RT-DETR-L | Short | 0.885 | 0.919 | 0.913 | 0.534 |
| RT-DETR-L | Missing_hole | 0.992 | 1.000 | 0.995 | 0.570 |
| RT-DETR-L | Spurious_copper | 0.862 | 0.752 | 0.821 | 0.453 |

RT-DETR-L showed clear gains for Mouse_bite, Open_circuit, Short, Missing_hole, and Spurious_copper. Spur remained similar between the two models, with YOLOv8n slightly higher in mAP50 and mAP50-95 for that class. Missing_hole achieved very high scores for both models, but this should be interpreted carefully because it has fewer samples than the other classes.

## Latency Comparison

| Model | Images Tested | Mean Latency (ms) | P50 Latency (ms) | P95 Latency (ms) | FPS | Under 100 ms Target |
|---|---:|---:|---:|---:|---:|---|
| YOLOv8n | 80 | 14.03 | 8.67 | 52.81 | 71.27 | Yes |
| RT-DETR-L | 80 | 50.27 | 44.79 | 91.74 | 19.89 | Yes |

YOLOv8n was significantly faster than RT-DETR-L. Its mean latency was 14.03 ms per image, compared with 50.27 ms for RT-DETR-L. However, both models remained below the project target of 100 ms per image. This means YOLOv8n is more suitable for strict real-time deployment, while RT-DETR-L is more suitable when accuracy is prioritized over speed.

## Robustness Summary for YOLOv8n Baseline

| Condition | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Clean subset | 0.827 | 0.795 | 0.856 | 0.457 |
| Brightness/contrast | 0.804 | 0.763 | 0.820 | 0.424 |
| Gaussian noise | 0.749 | 0.633 | 0.691 | 0.354 |
| Motion blur | 0.804 | 0.796 | 0.839 | 0.419 |
| Rotation 10 degrees | 0.831 | 0.786 | 0.855 | 0.456 |

The robustness test shows that YOLOv8n remained stable under brightness/contrast changes, motion blur, and small rotation. Gaussian noise caused the largest performance drop, reducing mAP50 to 0.691 and mAP50-95 to 0.354. This identifies noise robustness as the main limitation to address in future augmentation or preprocessing work.

## Interpretation

The comparison shows a clear accuracy-speed tradeoff. YOLOv8n provides fast inference and remains suitable for real-time PCB inspection. RT-DETR-L provides stronger accuracy, recall, and localization quality, but with higher latency. Since both models satisfy the 100 ms target, RT-DETR-L is a valid transformer-style benchmark for improving defect detection accuracy, while YOLOv8n remains the more deployment-friendly baseline.

## Transition Toward the Hybrid YOLO-Transformer Model

The survey paper motivates hybrid deep learning systems because PCB defects are often small, visually subtle, and affected by noise, lighting, and background texture. CNN-based detectors such as YOLO are efficient and strong for real-time localization, but they can struggle when fine contextual reasoning is needed. Transformer-based detectors improve global context modeling and can better capture relationships between defect regions and surrounding PCB patterns, but they are usually slower and more computationally expensive.

The pre-hybrid comparison supports this survey direction. YOLOv8n achieved fast inference at 14.03 ms per image, making it suitable for real-time use. RT-DETR-L achieved stronger test accuracy, improving mAP50 from 0.853 to 0.881 and mAP50-95 from 0.442 to 0.475. This means the two architectures have complementary strengths:

| Component | Strength | Limitation | Role in Hybrid System |
|---|---|---|---|
| YOLOv8n | Fast real-time detection | Lower accuracy than RT-DETR-L | Generate fast defect proposals |
| RT-DETR-L | Stronger contextual detection and localization | Higher latency | Validate or refine YOLO proposals |
| Hybrid YOLO-Transformer | Combines speed and contextual reasoning | More complex pipeline | Improve detection reliability while preserving real-time feasibility |

Based on these results, the next phase should implement a hybrid YOLO-Transformer strategy rather than only reporting separate model benchmarks. A defensible project implementation is a two-stage or late-fusion hybrid pipeline:

1. YOLOv8n performs fast initial defect detection and proposes candidate bounding boxes.
2. RT-DETR-L performs transformer-style contextual validation or refinement on the same image or proposed regions.
3. The predictions are fused using confidence, class agreement, and IoU overlap.
4. The fused output is evaluated on the same validation/test split using precision, recall, mAP50, mAP50-95, per-class metrics, and latency.

This keeps the hybrid claim aligned with the survey paper: the project is no longer only comparing CNN and transformer models, but using them together in one detection pipeline. The expected contribution of the hybrid model is not just higher accuracy, but a better tradeoff between YOLO's speed and the transformer's contextual reasoning.

## Recommended Report Flow

Use the following order in the final report:

1. Dataset preparation and class distribution.
2. YOLOv8n baseline training outputs: loss curves, mAP curves, confusion matrix, prediction samples.
3. YOLOv8n validation/test metrics and robustness results.
4. RT-DETR-L transformer benchmark metrics.
5. YOLOv8n vs RT-DETR-L comparison table.
6. Accuracy-latency tradeoff discussion.
7. Motivation for hybrid YOLO-Transformer model based on the comparison.
8. Hybrid model implementation and results.

