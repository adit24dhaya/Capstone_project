# Project Research Briefing: PCB Defect Detection

Use this as a read-through document before meetings, presentation Q&A, or advisor discussions.

## One-Minute Explanation

This project studies automated printed circuit board defect detection using object detection. The goal is to detect and localize six PCB defect classes: Missing hole, Mouse bite, Open circuit, Short, Spur, and Spurious copper. The paper compares YOLO11s and RT-DETR-L on the same dataset split, reports aggregate and per-class metrics, addresses the resolution fairness issue raised by reviewers, and demonstrates a web-deployed YOLO11s inference workflow through Hugging Face Spaces.

The main practical finding is that YOLO11s is the stronger deployment-oriented choice in the tested configurations. In the accepted practical comparison, YOLO11s at 1280 pixels achieved higher recall, higher mAP50-95, and lower latency than RT-DETR-L at 640 pixels. The camera-ready revision adds a matched 640-pixel comparison to address reviewer concerns, and YOLO11s remains slightly better while being much faster.

## Problem Motivation

PCBs are used in embedded and cyber-physical products. A small manufacturing defect on an unpopulated board can become expensive after assembly because it may cause failed tests, electrical issues, rework, or reliability problems.

Traditional inspection systems can use reference images, templates, or hand-engineered rules, but those workflows can be difficult to adapt when board layouts, lighting, or defect appearances change. Deep object detectors are useful because they can return both a defect class and a bounding box in one pass.

For manufacturing inspection, the detector needs more than high classification accuracy. It must:

- Detect small and subtle defects.
- Localize defects precisely enough for review or repair.
- Run fast enough for an inspection workflow.
- Provide understandable outputs such as boxes, confidences, class counts, and downloadable results.

## Dataset

The prepared dataset is referred to as YOLO_PCB. It uses YOLO-format labels, where each image has a text file containing class IDs and normalized bounding-box coordinates.

Dataset split:

- Training images: 5,551
- Validation images: 1,016
- Test images: 1,016
- Validation defect instances: 2,106
- Test defect instances: 2,179

Defect classes:

| ID | Class | Plain-English meaning |
|---:|---|---|
| 0 | Missing hole | A drilled hole or pad opening is absent. |
| 1 | Mouse bite | Small edge erosion or notch in copper. |
| 2 | Open circuit | A trace is broken or interrupted. |
| 3 | Short | Unwanted bridge between traces. |
| 4 | Spur | Thin copper protrusion from a trace. |
| 5 | Spurious copper | Extra copper not expected in the design. |

Training uses class-balancing copy-paste augmentation for underrepresented and small-defect categories. Validation and test sets are not augmented for metric reporting.

Important note for the web demo: upload clean raw PCB images. Do not upload screenshots, validation mosaics, or images that already contain text labels, because the model may detect artifacts or labels as visual patterns.

## Models

### YOLO11s

YOLO11s is a compact one-stage YOLO-family detector. It is designed for fast inference and predicts bounding boxes and class probabilities in one forward pass. It is a strong fit for deployment-oriented inspection because it is relatively lightweight and fast.

Main configurations:

- Accepted practical configuration: YOLO11s at 1280 pixels, 50 epochs, batch size 12.
- Matched-resolution configuration: YOLO11s at 640 pixels, 50 epochs, batch size 12.

### RT-DETR-L

RT-DETR-L is a real-time detection transformer. It uses transformer-style object queries rather than the same one-stage convolutional structure as YOLO. It is included as a contrasting detector family.

Main configurations:

- Accepted practical configuration: RT-DETR-L at 640 pixels, 10 epochs, batch size 4.
- Matched-resolution configuration: RT-DETR-L at 640 pixels, 10 epochs, batch size 4.
- High-resolution analysis: RT-DETR-L at 1280 pixels, 10 epochs, batch size 1.

## Evaluation Protocol

Evaluation uses the same split and the same Ultralytics validation workflow.

Protocol:

- Hardware: NVIDIA Tesla V100-SXM2-32GB.
- Evaluation batch size: 1.
- Workers: 0.
- Test-time augmentation: disabled.
- Metrics: precision, recall, F1, mAP50, mAP50-95, and latency.
- Latency: reported as preprocess, inference, postprocess, and total per-image time.

Batch size 1 is used because it better represents a single-image inspection workflow than large batched offline scoring.

## Metric Meanings

Precision: Of the detections predicted by the model, how many are correct?

Recall: Of the real labeled defects, how many did the model find?

F1: Harmonic mean of precision and recall.

mAP50: Mean average precision at IoU 0.50. This is a more forgiving localization metric.

mAP50-95: Mean average precision averaged over IoU thresholds from 0.50 to 0.95. This is stricter and better reflects precise localization quality.

Latency: Time per image. In the paper, benchmark latency is measured on V100. The Hugging Face Space runs on shared CPU infrastructure, so its latency is only a demo latency.

## Main Results

### Accepted practical comparison on the test split

| Model | Precision | Recall | F1 | mAP50 | mAP50-95 | Total latency |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11s 1280 | 0.880 | 0.866 | 0.873 | 0.902 | 0.502 | 15.2 ms |
| RT-DETR-L 640 | 0.886 | 0.839 | 0.862 | 0.887 | 0.470 | 49.8 ms |

Interpretation: YOLO11s-1280 has better recall, stricter localization, and latency. RT-DETR-L-640 has a small precision advantage but is slower.

### Matched-resolution comparison on the test split

| Model | Precision | Recall | F1 | mAP50 | mAP50-95 | Total latency |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11s 640 | 0.870 | 0.852 | 0.861 | 0.892 | 0.482 | 13.2 ms |
| RT-DETR-L 640 | 0.867 | 0.839 | 0.853 | 0.881 | 0.462 | 49.0 ms |

Interpretation: Under matched resolution, YOLO11s remains slightly better on recall and mAP while being much faster.

### RT-DETR-L high-resolution analysis

| Model | Precision | Recall | F1 | mAP50 | mAP50-95 | Total latency |
|---|---:|---:|---:|---:|---:|---:|
| RT-DETR-L 1280 | 0.766 | 0.534 | 0.630 | 0.604 | 0.295 | 74.4 ms |

Interpretation: Increasing RT-DETR-L to 1280 pixels did not improve this measured configuration. It became slower and less accurate, likely because higher resolution alone does not guarantee better transformer optimization under the same schedule and resource constraints.

## Per-Class Findings

The matched 640-pixel per-class results show that Missing hole is strong for both models. Open circuit and Short are also relatively strong. Mouse bite and Spur are harder because they are small, subtle, and spatially narrow.

YOLO11s-640 test examples:

- Missing hole recall: 0.986
- Mouse bite recall: 0.763
- Open circuit recall: 0.850
- Short recall: 0.892
- Spur recall: 0.767
- Spurious copper recall: 0.853

This matters because aggregate mAP can hide weaker classes. For inspection, a class with lower recall can be more important than a strong average if missed defects are costly.

## Reviewer Concern and How It Was Addressed

Reviewer concern: The original paper compared YOLO11s at 1280 pixels with RT-DETR-L at 640 pixels, so the comparison was not fully resolution-controlled.

Response:

- The original comparison is now described as a practical completed-configuration comparison.
- A matched 640-pixel comparison was added.
- RT-DETR-L was also tested at 1280 pixels.
- The paper states that even the matched-resolution comparison is not fully resource-normalized because the models differ in size, training schedule, memory use, and computational cost.

Reviewer concern: V100-only evaluation may not represent embedded deployment.

Response:

- The paper states that V100 latency is not edge-device latency.
- Jetson/TensorRT benchmarking is future work only.
- The Hugging Face Space is described as an online deployment artifact, not an embedded benchmark.

## Hugging Face Deployment Demo

The project is live at:

https://adiivd-pcb-defect-detection.hf.space

The demo uses the trained YOLO11s model with 1280-pixel input resolution through a Gradio web interface. A user can upload a clean PCB image or choose a sample image. The interface returns:

- Annotated output image.
- Detection count.
- Class breakdown.
- Confidence scores.
- Bounding-box coordinates.
- Before/after comparison.
- Downloadable annotated image.
- Downloadable JSON and CSV results.

Important clarification: the demo shows YOLO11s only. The paper provides the controlled YOLO11s versus RT-DETR-L comparison. The web app demonstrates practical deployment of the selected YOLO11s configuration.

## Limitations

The project is strong but not unlimited. The key limitations are:

- The matched-resolution experiment is not fully resource-normalized.
- RT-DETR-L and YOLO11s use different model capacities and training schedules.
- Evaluation is on an in-distribution dataset, not real factory video.
- Latency is benchmarked on V100, not Jetson or another embedded device.
- The Hugging Face demo runs on CPU and should not be treated as paper benchmark latency.
- The dataset may not cover all real-world lighting, camera, manufacturing, or board-layout variations.

## Future Work

Strong future directions:

- Edge-device benchmarking on Jetson/TensorRT or another embedded target.
- Resource-normalized comparison using controlled parameter count, FLOPs, memory, training budget, and latency budget.
- Real factory data collection with lighting and board-layout variation.
- High-resolution crop reinspection for small defects.
- Active visual inspection: detect uncertain areas at low resolution, crop and reinfer high-risk regions at higher resolution.
- Class-specific augmentation or hard-example mining for Mouse bite and Spur.

## What To Say If Asked For The Main Takeaway

The main takeaway is that YOLO11s is the more practical detector in this study. It performs strongly in the original 1280-pixel deployment configuration, remains competitive in the matched 640-pixel comparison, and is much faster than RT-DETR-L on the same V100 evaluation protocol. The work also shows why deployment-oriented papers must separate practical configuration comparisons from controlled fairness experiments.
