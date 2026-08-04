# ESCS'26 Presentation Script

**Paper:** Real-Time PCB Defect Detection for Embedded Visual Inspection: A Same-Split Comparison of YOLO11s and RT-DETR-L
**Presenter:** Aditya Varun Dhayapulay
**Target format:** Remote interactive Zoom presentation, 20 minutes including Q&A
**Recommended pacing:** 16-17 minutes for the talk, 3-4 minutes for questions

## Slide 1 - Title

Good morning, everyone. My name is Aditya Varun Dhayapulay, and I am presenting our work titled "Real-Time PCB Defect Detection for Embedded Visual Inspection: A Same-Split Comparison of YOLO11s and RT-DETR-L." This work was completed with Dr. Paul Salvador Inventado at California State University, Fullerton.

The main goal of this project is to evaluate practical object detection models for printed circuit board defect inspection. We focus on six common PCB defect categories and compare a compact YOLO-family detector with a transformer-based RT-DETR detector. The emphasis is not on proposing a new detector architecture, but on building a traceable and reproducible benchmark that can support deployment-oriented inspection decisions.

## Slide 2 - Why PCB Inspection Is Hard

Printed circuit boards are used in almost every embedded and cyber-physical system. Small visual defects on an unpopulated board can become much more expensive after assembly, testing, or field deployment. For example, a missing hole, a short, a spur, or spurious copper can affect manufacturing quality and product reliability.

This makes automated optical inspection important, but the problem is not only about classification accuracy. A useful inspection detector must localize small defects, provide bounding boxes that can support review or repair, and run with latency compatible with an inspection workflow. That is why this project studies both detection accuracy and inference timing.

## Slide 3 - Research Question

The research question is: under one controlled PCB defect dataset split, how do YOLO11s and RT-DETR-L compare for real-time PCB defect detection?

YOLO11s represents a compact, one-stage convolutional detector that is widely used when latency matters. RT-DETR-L represents a transformer-style detector designed for real-time end-to-end detection. Comparing these models is useful because PCB defects are small, sparse, and often require precise localization.

The study is deployment-oriented. We want to know which completed configuration is more practical, but we also want to separate that practical comparison from a cleaner resolution-controlled comparison.

## Slide 4 - What the Camera-Ready Version Adds

There are five main contributions in this camera-ready version.

First, we preserve the accepted practical comparison between YOLO11s trained and evaluated at 1280 pixels and RT-DETR-L trained and evaluated at 640 pixels.

Second, we add a matched-resolution comparison where both YOLO11s and RT-DETR-L are trained and evaluated at 640 pixels on the same split.

Third, we report a high-resolution RT-DETR-L experiment at 1280 pixels using batch size 1.

Fourth, we correct the canonical class mapping for semantic per-class analysis and provide class-balanced success examples and representative failure cases.

Finally, we provide an online Hugging Face deployment artifact, while keeping Jetson and TensorRT benchmarking as future work.

## Slide 5 - Dataset and Defect Taxonomy

The study uses six PCB defect classes: Missing hole, Mouse bite, Open circuit, Short, Spur, and Spurious copper.

The dataset is called YOLO_PCB. It combines the project PCB dataset with overlapping classes from DsPCBSD+. The split is deterministic using seed 42. It contains 5,551 training images, 1,016 validation images, and 1,016 test images. The validation set contains 2,106 labeled instances, and the test set contains 2,179 labeled instances.

The training split also includes class-balancing copy-paste augmentation: 320 Missing hole, 240 Mouse bite, and 240 Spur examples. This was used to help with underrepresented and small-defect categories.

## Slide 6 - Reproducible Evaluation Protocol

For reproducibility, every reported model is evaluated under the same protocol. The experiments use Ultralytics 8.4.51 and PyTorch 2.5.1 on an NVIDIA Tesla V100-SXM2-32GB GPU. Evaluation uses batch size 1, zero dataloader workers, the same validation and test images, and no test-time augmentation.

We report precision, recall, F1, mAP50, mAP50-95, and latency. The latency includes preprocess, inference, postprocess, and total per-image time. Before running the new controlled experiments, the archived checkpoints were evaluated on the rebuilt split, and the pipeline stopped if the archived metrics differed by more than 0.001. The observed differences were approximately 10 to the negative 8, confirming that the split and evaluator were reproduced correctly.

## Slide 7 - Models and Training Settings

The original YOLO11s checkpoint was trained for 50 epochs at 1280 pixels with batch size 12. The original RT-DETR-L checkpoint was trained for 10 epochs at 640 pixels with batch size 4. These are the accepted practical configurations.

For the matched-resolution study, YOLO11s was trained for 50 epochs at 640 pixels with batch size 12 and seed 42. RT-DETR-L was trained for 10 epochs at 640 pixels with batch size 4 and seed 42.

We also trained RT-DETR-L at 1280 pixels for 10 epochs with batch size 1. That run completed successfully, so the planned 960-pixel fallback was not needed.

## Slide 8 - The Key Fairness Issue: Resolution

The accepted comparison was useful, but it had one important limitation: YOLO11s used 1280-pixel input while RT-DETR-L used 640-pixel input. That means the original table should be interpreted as a practical comparison of completed project configurations, not as a fully resolution-controlled architecture comparison.

This matters because PCB defects can be very small. Higher input resolution may preserve more spatial detail and can improve localization. To address this reviewer concern, we added the matched 640-pixel comparison and the high-resolution RT-DETR-L experiment. This lets us distinguish practical deployment results from resolution-controlled evidence.

## Slide 9 - Accepted Practical Comparison

This table shows the accepted practical test comparison. YOLO11s at 1280 pixels achieved 0.880 precision, 0.866 recall, 0.902 mAP50, and 0.502 mAP50-95. Its total latency was 15.2 milliseconds per image.

RT-DETR-L at 640 pixels achieved 0.886 precision, 0.839 recall, 0.887 mAP50, and 0.470 mAP50-95. Its total latency was 49.8 milliseconds.

So in the practical completed configurations, YOLO11s had higher recall, higher mAP50-95, and substantially lower latency. RT-DETR-L had a small precision advantage, but it was slower in this setup.

## Slide 10 - Matched-Resolution Results

This slide shows the controlled 640-pixel comparison. YOLO11s at 640 pixels achieved 0.870 precision, 0.852 recall, 0.892 mAP50, and 0.482 mAP50-95, with 13.2 milliseconds total latency.

RT-DETR-L at 640 pixels achieved 0.867 precision, 0.839 recall, 0.881 mAP50, and 0.462 mAP50-95, with 49.0 milliseconds total latency.

Under matched resolution, YOLO11s still performed slightly better on recall and strict localization, while remaining much faster. This directly addresses the review concern that the original comparison was not fully resolution-controlled.

## Slide 11 - High-Resolution RT-DETR-L Finding

We also tested whether RT-DETR-L benefited from the higher 1280-pixel input size. In this run, RT-DETR-L at 1280 pixels achieved 0.766 precision, 0.534 recall, 0.604 mAP50, and 0.295 mAP50-95, with 74.4 milliseconds total latency.

This result did not improve over RT-DETR-L at 640 pixels. The likely interpretation is that simply increasing input resolution is not enough. The model also has a different training schedule, memory requirement, optimization behavior, and computational cost. Therefore, this row is reported separately as a high-resolution analysis, not as a replacement for the matched 640-pixel comparison.

## Slide 12 - Per-Class Behavior at 640 px

The per-class table gives more detail about the matched 640-pixel models. Missing hole is detected very strongly by both models, with near-perfect mAP50. Open circuit and Short also perform relatively well.

The more difficult categories are Mouse bite and Spur. For YOLO11s at 640 pixels, Mouse bite recall is 0.763 and Spur recall is 0.767. For RT-DETR-L at 640 pixels, Mouse bite recall is 0.765 and Spur recall is 0.729.

This suggests that small or subtle edge defects remain challenging. The per-class analysis is important because aggregate mAP can hide the classes that are most likely to need additional inspection or second-stage refinement.

## Slide 13 - Representative Successes

This figure shows class-balanced representative detections from the matched YOLO11s-640 test run. The goal of this slide is not just to show attractive examples, but to confirm that the detector can localize different defect types across the six-class taxonomy.

In practical inspection, the bounding box is important because it tells a reviewer where to look. A correct class label without useful localization would be less helpful for repair or quality-control workflows. These examples show that the model can often identify the correct defect region and produce usable boxes.

## Slide 14 - Representative Failure Cases

This figure shows representative failure cases selected using a miss-weighted error score. The selection prioritizes false negatives because missed defects are usually more costly than false alarms in an inspection setting.

The failures show why mAP50-95 matters. A detector may identify the general defect region at IoU 0.50 but still produce a loose box at stricter thresholds. Some defects are also very small or visually subtle, especially Mouse bite and Spur. These results motivate future work on high-resolution crops, class-specific augmentation, and possibly a second-stage refinement model.

## Slide 15 - Online Deployment Artifact

The trained YOLO checkpoint is served through a public Hugging Face Space. The browser workflow allows a user to upload a PCB image and returns annotated detections, class counts, confidence values, and downloadable results.

This is an online deployment artifact and usability demonstration. It shows that the model can be packaged into an accessible inference workflow. However, it is not an embedded-device benchmark. All latency values in the paper are measured on the V100 GPU, and we do not claim Jetson, TensorRT, RK3588, FPGA, or edge-device performance.

## Slide 16 - Limitations and Future Work

There are several limitations. First, the matched-resolution comparison is not fully resource-normalized because YOLO11s and RT-DETR-L differ in model capacity, training schedule, memory use, and computational cost.

Second, the evaluation is an in-distribution benchmark. Real factory deployment would introduce lighting changes, registration errors, product variation, and manufacturing noise.

Third, we did not benchmark on edge hardware. Jetson and TensorRT remain future work until they can be measured directly.

Future work will include real factory captures, resource-normalized training, high-resolution crop reinspection, and actual embedded-device deployment.

## Slide 17 - Takeaways

To summarize, the accepted practical comparison favored YOLO11s-1280 for recall, strict localization, and latency. The matched 640-pixel comparison confirms that YOLO11s remains competitive under a controlled resolution setting. Increasing RT-DETR-L to 1280 pixels did not improve the measured configuration, and it increased latency.

The larger lesson is that deployment-oriented comparisons need to separate practical completed configurations from controlled ablations. For PCB inspection, resolution, latency, per-class behavior, and failure analysis all matter.

Thank you for listening. I would be happy to take questions.
