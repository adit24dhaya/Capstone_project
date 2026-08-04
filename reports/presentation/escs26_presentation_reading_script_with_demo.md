# ESCS'26 Presentation Reading Script With Demo

Paper: Real-Time PCB Defect Detection for Embedded Visual Inspection: A Same-Split Comparison of YOLO11s and RT-DETR-L
Presenter: Aditya Varun Dhayapulay
Recommended pacing: the slot is 20 minutes including Q&A. The spoken script runs about 12-13 minutes at a normal reading pace, including the 20-30 second demo video, which leaves about 6-7 minutes for questions. There is no need to rush.

## Slide 1 - Title

Good afternoon, everyone. My name is Aditya Varun Dhayapulay, and I am presenting our work titled "Real-Time PCB Defect Detection for Embedded Visual Inspection: A Same-Split Comparison of YOLO11s and RT-DETR-L." This work was completed with Dr. Paul Salvador Inventado at California State University, Fullerton.

The main goal of this project is to evaluate practical object detection models for printed circuit board defect inspection. We focus on six common PCB defect categories and compare a compact YOLO-family detector with a transformer-based RT-DETR detector. The emphasis is not on proposing a new detector architecture, but on building a traceable and reproducible benchmark that can support deployment-oriented inspection decisions.

## Slide 2 - Motivation

Printed circuit boards are used in many embedded and cyber-physical systems. Small visual defects on an unpopulated board can become much more expensive after assembly, testing, or field deployment. Examples include missing holes, shorts, spurs, mouse bites, open circuits, and spurious copper.

Automated optical inspection is important, but this problem is not only classification. A useful detector must localize small defects, produce bounding boxes that support review or repair, and run fast enough for an inspection workflow. That is why this project studies both accuracy and latency.

## Slide 3 - Goal

The research question is: under one controlled PCB defect dataset split, how do YOLO11s and RT-DETR-L compare for real-time PCB defect detection?

YOLO11s is a compact one-stage convolutional detector built for fast inference. RT-DETR-L is a transformer-style detector designed for real-time end-to-end detection. Comparing them is useful because PCB defects are small, sparse, and often require precise localization.

The study is deployment-oriented. We want to know which completed configuration is more practical, while also separating that practical comparison from a cleaner resolution-controlled comparison.

## Slide 4 - Contributions

There are five main contributions in this camera-ready version.

First, we preserve the accepted practical comparison between YOLO11s trained and evaluated at 1280 pixels and RT-DETR-L trained and evaluated at 640 pixels.

Second, we add a matched-resolution comparison where both YOLO11s and RT-DETR-L are trained and evaluated at 640 pixels on the same split.

Third, we report a high-resolution RT-DETR-L experiment at 1280 pixels using batch size 1.

Fourth, we correct the canonical class mapping for semantic per-class analysis and provide class-balanced success examples and representative failure cases.

Finally, we provide an online Hugging Face deployment artifact, while keeping Jetson and TensorRT benchmarking as future work.

## Slide 5 - Dataset

The dataset has six defect classes: Missing hole, Mouse bite, Open circuit, Short, Spur, and Spurious copper.

The prepared YOLO_PCB split contains 5,551 training images, 1,016 validation images, and 1,016 test images. The validation set contains 2,106 labeled defect instances, and the test set contains 2,179 labeled instances.

The training split also includes class-balancing copy-paste augmentation for underrepresented and small-defect categories. Validation and test images are kept unchanged, so the reported metrics reflect the fixed evaluation split.

## Slide 6 - Evaluation Protocol

Every reported model is evaluated under the same protocol. The experiments use Ultralytics and PyTorch on an NVIDIA Tesla V100-SXM2-32GB GPU. Evaluation uses batch size 1, zero dataloader workers, the same validation and test images, and no test-time augmentation.

We report precision, recall, F1, mAP50, mAP50-95, and latency. Batch size 1 is important because it avoids inflated throughput from batched offline scoring and is closer to an inline inspection setting.

Before comparing new experiments, the archived baseline checkpoints were evaluated on the rebuilt split. The pipeline stopped if the archived values differed by more than 0.001, which helped verify that the split and evaluator were reproduced correctly.

## Slide 7 - Models and Training Settings

The accepted practical YOLO11s checkpoint was trained for 50 epochs at 1280 pixels with batch size 12. The accepted practical RT-DETR-L checkpoint was trained for 10 epochs at 640 pixels with batch size 4.

For the matched-resolution study, YOLO11s was trained for 50 epochs at 640 pixels, and RT-DETR-L was trained for 10 epochs at 640 pixels. Both use the same split and seed-controlled workflow.

We also trained RT-DETR-L at 1280 pixels with batch size 1. That run completed successfully, so the planned 960-pixel fallback was not needed.

## Slide 8 - Experiment Design

The key fairness issue is resolution. The accepted comparison was useful, but YOLO11s used 1280-pixel input while RT-DETR-L used 640-pixel input. That means the original table should be interpreted as a practical comparison of completed project configurations, not as a fully resolution-controlled architecture comparison.

This matters because PCB defects can be very small. A higher input resolution may preserve more spatial detail and improve localization. To address this reviewer concern, we added the matched 640-pixel comparison and the high-resolution RT-DETR-L experiment.

## Slide 9 - Practical Comparison

This table shows the accepted practical test comparison. YOLO11s at 1280 pixels achieved 0.880 precision, 0.866 recall, 0.902 mAP50, and 0.502 mAP50-95, with 15.2 milliseconds total latency.

RT-DETR-L at 640 pixels achieved 0.886 precision, 0.839 recall, 0.887 mAP50, and 0.470 mAP50-95, with 49.8 milliseconds total latency.

So in the completed practical configurations, YOLO11s had higher recall, higher mAP50-95, and substantially lower latency. RT-DETR-L had a small precision advantage, but it was slower in this setup.

## Slide 10 - Matched-Resolution Results

This slide shows the controlled 640-pixel comparison. YOLO11s at 640 pixels achieved 0.870 precision, 0.852 recall, 0.892 mAP50, and 0.482 mAP50-95, with 13.2 milliseconds total latency.

RT-DETR-L at 640 pixels achieved 0.867 precision, 0.839 recall, 0.881 mAP50, and 0.462 mAP50-95, with 49.0 milliseconds total latency.

Under matched resolution, YOLO11s still performed slightly better on recall and strict localization while remaining much faster. This directly addresses the concern that the original comparison was not fully resolution-controlled.

## Slide 11 - High-Resolution RT-DETR-L

We also tested whether RT-DETR-L benefited from higher 1280-pixel input. In this run, RT-DETR-L at 1280 pixels achieved 0.766 precision, 0.534 recall, 0.604 mAP50, and 0.295 mAP50-95, with 74.4 milliseconds total latency.

This did not improve over RT-DETR-L at 640 pixels. The interpretation is that simply increasing input resolution is not enough. The model also has different optimization behavior, memory use, training schedule, and computational cost.

Because of that, this result is reported separately as high-resolution analysis, not as a replacement for the matched 640-pixel comparison.

## Slide 12 - Per-Class Behavior

The per-class table gives more detail about the matched 640-pixel models. Missing hole is detected strongly by both models, and Open circuit and Short are also relatively strong.

The harder categories are Mouse bite and Spur. These are small, subtle defects. Mouse bite is a small edge notch, and Spur is a thin copper protrusion. Their small spatial extent and low contrast make them harder to localize precisely.

This per-class analysis is important because aggregate mAP can hide the classes that most need additional inspection or second-stage refinement.

## Slide 13 - Representative Successes

This figure shows class-balanced representative detections from the matched YOLO11s-640 test run. The goal is not only to show attractive examples, but to show that the detector can localize different defect types across the six-class taxonomy.

In practical inspection, the bounding box is important because it tells a reviewer where to look. A correct class label without useful localization would be less helpful for repair or quality-control workflows.

## Slide 14 - Representative Failure Cases

This figure shows representative failure cases selected using a miss-weighted error score. The score is used only to select failure examples for manual review. It is not the training loss and it is not a headline evaluation metric.

The failures show why mAP50-95 matters. A detector may identify the general defect region at IoU 0.50 but still produce a loose box at stricter thresholds. Some defects are also very small or visually subtle, especially Mouse bite and Spur.

These results motivate future work on high-resolution crop reinspection, class-specific augmentation, and possibly a second-stage refinement model.

## Slide 15 - Deployment Demo

Before playing the video, say:

This slide shows the deployed browser workflow. The paper contains the controlled comparison between YOLO11s and RT-DETR-L, but this web demo focuses on the selected YOLO11s deployment configuration.

While the video plays, say:

This video demonstrates the deployed version of our PCB defect detection system. The application uses our trained YOLO11s model with an input resolution of 1280 pixels and supports all six PCB defect classes. Here, I select a sample containing spur defects. The image is processed by the model, and the detected regions are displayed as bounding boxes. In this example, the system identifies three spur defect candidates with confidence scores between approximately 77 and 79 percent. The interface also reports inference latency, class-level counts, bounding-box coordinates, and allows the annotated image and structured JSON or CSV results to be downloaded.

After the video, say:

One important clarification is that the online demo runs on shared CPU infrastructure. Its latency is useful for showing the browser workflow, but it should not be treated as the V100 benchmark latency reported in the experimental section.

## Slide 16 - Limitations and Future Work

There are several limitations. First, the matched-resolution comparison is not fully resource-normalized because YOLO11s and RT-DETR-L differ in model capacity, training schedule, memory use, and computational cost.

Second, this is an in-distribution benchmark. Real factory deployment would introduce lighting changes, registration errors, product variation, and manufacturing noise.

Third, we did not benchmark on edge hardware. Jetson and TensorRT remain future work until they can be measured directly.

Future work will include real factory captures, resource-normalized training, high-resolution crop reinspection, and actual embedded-device deployment.

## Slide 17 - Takeaways and Close

To summarize, the accepted practical comparison favored YOLO11s-1280 for recall, strict localization, and latency. The matched 640-pixel comparison confirms that YOLO11s remains competitive under a controlled resolution setting. Increasing RT-DETR-L to 1280 pixels did not improve the measured configuration and increased latency.

The larger lesson is that deployment-oriented comparisons need to separate practical completed configurations from controlled ablations. For PCB inspection, resolution, latency, per-class behavior, and failure analysis all matter.

Thank you for listening. I would be happy to take questions.

If there is time, I can also show the Hugging Face deployment link or answer questions about the dataset, resolution-control experiments, or future edge-device benchmarking.

Stage direction: stay on slide 17 during Q&A. Slide 18 is the Q&A backup appendix divider - do not advance to it while closing. Jump into the appendix only when a question calls for it, by typing the slide number and pressing Enter: 19 what changed from the accepted draft, 20 per-class label correction, 21 resize-only diagnostic, 22 validation results, 23 full per-class table, 24 fusion and inspection cost, 25 latency details, 26 dataset provenance, 27 environment and reproducibility.
