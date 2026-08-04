# Capstone Coursework Progress Update - July 2026

## Project

**Title:** Real-Time PCB Defect Detection for Embedded Visual Inspection
**Student:** Aditya Varun Dhayapulay
**Project Type:** Master's capstone / final semester project
**Current status:** Core implementation, evaluation, online deployment, report, paper, and presentation artifacts are substantially complete.

## Original Timeline Summary

The original project timeline targeted completion by **July 10, 2026** and included:

1. Dataset expansion and standardization
2. Baseline model training and benchmarking
3. Advanced model development
4. Hybrid/reference-based pipeline
5. Final evaluation and optimization
6. Deployment through ONNX/TensorRT
7. Final documentation and presentation

## Updated Progress Against Timeline

| Phase | Original Goal | Current Status |
|---|---|---|
| Dataset expansion and standardization | Integrate additional PCB datasets and standardize annotations | Completed as a unified YOLO-style PCB defect dataset with deterministic train/validation/test split |
| Baseline model benchmarking | Train YOLO models and evaluate mAP, precision, recall | Completed using YOLO11s with aggregate and per-class metrics |
| Advanced model development | Improve model and compare against baseline | Completed through YOLO11s vs RT-DETR-L comparison, matched-resolution experiments, and high-resolution RT-DETR-L analysis |
| Hybrid/reference-based pipeline | Explore reference/template-based comparison | Partially shifted in scope; publication-quality detector comparison and resolution-control experiments became the main focus |
| Final evaluation and optimization | Compare configurations and optimize accuracy/latency | Completed with V100 batch-1 evaluation, no TTA, latency reporting, per-class analysis, and success/failure figures |
| Deployment | Export model and optimize inference | ONNX export and Hugging Face online deployment completed; TensorRT/Jetson benchmarking remains future work due to lack of target hardware access |
| Documentation and presentation | Final report and slides | Completed/ongoing: capstone report, ESCS camera-ready paper, presentation slides, and voice-over script |

## Major Completed Work

- Built and standardized a six-class PCB defect detection dataset.
- Trained and evaluated YOLO11s and RT-DETR-L models.
- Completed reviewer-requested resolution-control experiments:
  - YOLO11s at 1280 px practical configuration
  - RT-DETR-L at 640 px practical configuration
  - YOLO11s at 640 px matched-resolution comparison
  - RT-DETR-L at 640 px matched-resolution comparison
  - RT-DETR-L at 1280 px high-resolution analysis
- Generated aggregate metrics, per-class metrics, latency measurements, and representative success/failure examples.
- Exported the trained YOLO model to ONNX.
- Deployed the model as a Hugging Face Space for online PCB defect detection.
- Prepared the final capstone report and ESCS camera-ready paper.
- Prepared a 20-minute ESCS presentation deck and slide-by-slide script.

## Publication and Presentation Update

The project resulted in an accepted regular research paper:

**Paper ID:** ESC3007
**Title:** Real-Time PCB Defect Detection for Embedded Visual Inspection: A Same-Split Comparison of YOLO11s and RT-DETR-L
**Authors:** Aditya Varun Dhayapulay and Paul Salvador Inventado
**Venue:** ESCS'26 / CSCE'26
**Status:** Accepted for publication and presentation

The presentation mode selected is **remote interactive / Zoom**.

## Honest Scope Notes

- The original plan included TensorRT optimization, but full TensorRT/Jetson benchmarking was not completed because target hardware access was not available.
- The project should only claim ONNX export and online Hugging Face deployment, not embedded-device performance.
- The reference-based/hybrid pipeline was de-emphasized because the final project shifted toward publication-quality detector benchmarking and reviewer-requested resolution-control analysis.
- All reported latency values are from NVIDIA V100 evaluation, not from embedded hardware.

## Remaining Work Before Final Capstone Closure

- Confirm final ESCS presentation schedule and Zoom instructions.
- Review presentation slides with the project advisor.
- Prepare a short demo walkthrough of the Hugging Face Space.
- Keep the Springer camera-ready upload ready when the final upload link is provided.
- Optionally document future work on active visual inspection, high-resolution crop reinspection, and true edge-device benchmarking.

## Suggested Summary for Advisor

The capstone project has met the main academic goals: dataset preparation, model training, benchmarking, evaluation, deployment, documentation, and presentation. The scope evolved from the initial timeline because the project produced an accepted ESCS'26 paper and required additional resolution-control experiments. The only major original item not fully completed is target-device TensorRT/Jetson benchmarking, which is documented as future work due to lack of hardware access.
