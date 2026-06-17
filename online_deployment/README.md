---
title: Automated PCB Defect Detection
colorFrom: blue
colorTo: green
sdk: gradio
python_version: 3.11
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
---

# Automated PCB Defect Detection

This Hugging Face Space deploys the capstone project's PCB defect detector as an online demo.
Users can upload a PCB image and receive:

- Annotated output with color-coded defect boxes
- Metric cards (detections, latency, input size, classes hit)
- Inspection verdict banner (pass vs review recommended)
- Per-class count breakdown and filterable detection table
- Before/after image comparison gallery
- Downloadable annotated image, JSON, and CSV results
- Clean raw PCB sample buttons for each defect class
- About tab with capstone deployment context and reference offline metrics

## Model File

The trained model artifacts are included in:

- `models/best.pt`
- `models/best.onnx`

The current Space uses the paper-backed YOLO11s-1280 checkpoint from the accepted ESCS'26
experiments. The app loads `models/best.pt` by default. It also supports setting a custom
model path with the `MODEL_PATH` environment variable.

Reference offline test metrics for this deployed checkpoint:

- mAP50: `0.902`
- mAP50-95: `0.502`
- recall: `0.866`
- V100 batch-1 inference latency: `12.8 ms/image`

## Classes

- Missing hole
- Mouse bite
- Open circuit
- Short
- Spur
- Spurious copper

## Deployment Note

This online demo satisfies the web-deployment demonstration requirement. It does not claim
Jetson/TensorRT deployment. The project report correctly treats TensorRT benchmarking as future
work because target hardware/runtime access was unavailable.

## Input Image Note

Use clean PCB images for the demo. The bundled sample buttons use raw images from
`PCB-DATASET-master/images/<class>/`. You can also browse the public
[PCB-DATASET image folders](https://github.com/Ironbrotherstyle/PCB-DATASET/tree/master/images)
and upload a clean raw `.jpg` from an `images/<class>/` folder. Do not upload YOLO
training/validation batch mosaics, screenshots, or images that already contain filenames,
class labels, or drawn boxes. Text and overlay labels are out-of-distribution visual noise
and can create false positives near letters.
