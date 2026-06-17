# Real-Time PCB Defect Detection for Embedded Visual Inspection: A Same-Split Comparison of YOLO11s and RT-DETR-L

**Aditya Varun Dhayapulay** (contact author), **Paul Salvador Inventado**
California State University, Fullerton, Fullerton, California, USA
Emails: adivd@csu.fullerton.edu; pinventado@fullerton.edu

**Abstract.** PCB inspection demands accurate localization of small defects at frame rates compatible with conveyor-paced manufacturing. We compare two detectors -- YOLO11s at 1280 px and RT-DETR-L at 640 px -- on a 5,551 / 1,016 / 1,016 train, validation, and test split spanning six defect classes. Both checkpoints are scored with the Ultralytics validation API at batch size one on an NVIDIA V100. YOLO11s reaches 0.880 precision, 0.866 recall, 0.902 mAP50, and 0.502 mAP50-95 with 12.6 ms inference time per image. RT-DETR-L is slightly more precise but trails on recall, mAP, and latency. Per-class analysis identifies Mouse_bite and Missing_hole as the lowest-localization categories; supporting fusion results quantify how recall-oriented configurations reduce missed defects while increasing false-positive review burden.

**Keywords:** PCB defect detection, embedded vision, YOLO11, RT-DETR, real-time object detection, industrial inspection

**Submission category:** Regular Research Paper

## 1. Introduction

Printed circuit boards (PCBs) sit at the core of embedded and cyber-physical products, and a visual defect on an unpopulated board -- a missing hole, a hairline short, a stray copper spur -- can become much more expensive after downstream assembly. Optical inspection is therefore a practical embedded-vision workload in electronics manufacturing, and the bar it has to clear is not just accuracy: the detector has to be fast enough for an inspection line, reliable enough to reduce missed defects, and portable enough to move beyond an HPC training environment.

This paper looks at a narrow slice of that problem. We take a six-class PCB defect dataset -- Missing_hole, Mouse_bite, Open_circuit, Short, Spur, Spurious_copper -- and ask which of two pretrained detector families, restored from saved project checkpoints, is the better starting point for an embedded inspection deployment. The two candidates are YOLO11s, a compact one-stage detector from the You Only Look Once family run at 1280 pixels because the defects of interest are tiny relative to the board, and RT-DETR-L, a real-time detection transformer evaluated in the saved 640-pixel configuration. Both were evaluated on the same `YOLO_PCB` split using the same Ultralytics validation workflow. Ultralytics provides the Python API and command-line tooling used here for model loading, validation, metric calculation, and timing, so the comparison is controlled at the split and evaluator level.

Beyond the headline detector choice, we report a per-class breakdown that pinpoints which defects still resist high-overlap box localization, and we keep fusion configurations as supporting results so that we can discuss the recall-versus-false-positive trade that an inspection line has to make. We are deliberately not claiming a new architecture or a state-of-the-art number; the contribution is a traceable baseline on a reproducible PCB split, with a deployment lens.

The remainder of the paper is organized as follows. Section 2 places the work in context. Section 3 describes the dataset and the evaluation environment. Section 4 reports the unified test and validation results, the per-class breakdown, and the fusion and inspection-cost numbers. Sections 5 and 6 discuss what the numbers do and do not support, and Section 7 closes with what we would do next.

## 2. Related Work

PCB defect detection has cycled through reference-comparison methods, classical machine-vision pipelines, and, more recently, end-to-end deep object detectors. Reference and rule-based inspection can be efficient when board layout, lighting, and registration are tightly controlled, but these pipelines often require manual thresholds and feature engineering for each product family. Learned object detectors are useful when the inspection system needs a single pass that both classifies a defect and returns a bounding box for downstream review or repair guidance. The six-class synthetic dataset released by Huang and Wei [1] is a useful reference point because it offers labeled examples for each of the defect categories addressed here. The underlying inspection problem -- small, sparse, visually subtle defects on a high-contrast copper substrate -- makes localization as important as classification.

On the detector side, YOLO-family models are commonly used when latency matters because they predict object boxes and classes in one forward pass. We use Ultralytics YOLO11 [2] because its s/m/l variants and its training and validation tooling support reproducible high-resolution training without bespoke code. As a contrasting design point we include RT-DETR-L, the real-time end-to-end detection transformer proposed by Zhao et al. [3]. This gives a practical comparison between a convolutional one-stage detector and a transformer-style detector, two families that are both relevant to current real-time object detection work.

Detector ensembling is also relevant for inspection because the cost of a missed defect is usually not symmetric with the cost of a false alarm. We adopt the Weighted Boxes Fusion (WBF) formulation of Solovyev et al. [4], with a class-wise calibrated policy, to obtain a supporting operating point that emphasizes recall.

## 3. Dataset and Experimental Setup

### 3.1 Dataset

The primary dataset is a PCB defect corpus that we refer to as `YOLO_PCB`. The starting files were accessed in the project through the Kaggle dataset `aditya2402/pcb-dataset` and prepared into YOLO label format. In YOLO format, each image is paired with a text label file whose rows store a class identifier and normalized bounding-box center, width, and height. The prepared split contains the six defect classes listed in Table 1 and is divided into 5,551 training, 1,016 validation, and 1,016 test images. The validation split contains 2,106 labeled defect instances; the test split contains 2,179. All headline metrics in this paper come from the validation and test partitions of that single split. The reported workflow uses separate train, validation, and test directories, and the test metrics were not used for hyperparameter tuning.

**Table 1.** Six defect categories used in `YOLO_PCB`.

| ID | Class           |
|---:|-----------------|
| 0  | Missing_hole    |
| 1  | Mouse_bite      |
| 2  | Open_circuit    |
| 3  | Short           |
| 4  | Spur            |
| 5  | Spurious_copper |

### 3.2 Models

Two checkpoints are evaluated. `YOLO11s_1280_kaggle_fair` is a YOLO11s detector fine-tuned at an input size of 1280 px to preserve the spatial detail that small PCB defects depend on. `RTDETR_L_kaggle_fair` is an RT-DETR-L transformer fine-tuned and evaluated in the saved 640 px configuration. Both checkpoint names are experiment identifiers from the project artifact folders, not separate model families. Both checkpoints are reloaded for evaluation through the same Ultralytics interface; neither is re-tuned between the runs reported in Tables 2 and 3.

### 3.3 Evaluation environment

All numbers come from a single run of Ultralytics' `model.val` evaluator at batch size one. Batch size one avoids throughput inflation from larger batches and is closer to an inline inspection setting than batched offline scoring. The host configuration is summarized in Table 2.

**Table 2.** Evaluation host.

| Component      | Value                |
|----------------|----------------------|
| GPU            | NVIDIA Tesla V100-SXM2-32GB |
| Python         | 3.12.8               |
| PyTorch        | 2.5.1 + CUDA 12.4    |
| Ultralytics    | 8.4.51               |
| OpenCV         | 4.11.0               |
| NumPy          | 1.26.4               |

Latency numbers here are V100 numbers, not embedded-device numbers; Section 6 returns to that point.

### 3.4 Metrics

We report precision, recall, F1 score, mean average precision (mAP), and per-image preprocess, inference, and postprocess times. Precision measures how many predicted defects are correct; recall measures how many labeled defects are found; and F1 is the harmonic mean of precision and recall. mAP summarizes area under the precision-recall curve across classes at a given intersection-over-union (IoU) threshold. mAP50 uses IoU 0.50 and is a forgiving overlap metric; mAP50-95 averages over IoU 0.50:0.05:0.95 and is stricter. The stricter metric is more informative when an inspection workflow has to crop the defect region for repair guidance or for a second human-in-the-loop pass. Reporting both side by side makes the localization gap visible rather than hidden.

## 4. Results

### 4.1 Same-split test performance

Table 3 is the headline. YOLO11s at 1280 px outperforms RT-DETR-L at 640 px on every metric except precision, and it does so at roughly 2.5x the speed.

**Table 3.** Unified batch-1 evaluation on the `YOLO_PCB` test split (1,016 images, 2,179 instances).

| Model         | Precision | Recall | F1    | mAP50  | mAP50-95 | Inference (ms) | Total (ms) |
|---------------|----------:|-------:|------:|-------:|---------:|---------------:|-----------:|
| YOLO11s 1280  | 0.880     | 0.866  | 0.873 | 0.902  | 0.502    | 12.6           | 14.6       |
| RT-DETR-L 640 | 0.886     | 0.839  | 0.862 | 0.887  | 0.470    | 31.1           | 31.9       |

The mAP50-95 gap of 0.032 in absolute terms is small but consistent: YOLO11s produces box coordinates that keep higher overlap with the ground-truth boxes across stricter IoU thresholds. The recall difference of 2.8 points is, for an inspection workflow, the more practically important one, because a defect that the detector never proposes cannot be recovered downstream.

![Figure 1. Test mAP50-95 by model](figures/test_mAP50_95_by_model.png)

![Figure 2. Test inference time by model](figures/test_inference_ms_by_model.png)

### 4.2 Validation performance

Validation numbers (Table 4) are consistent with the test pattern. RT-DETR-L is fractionally ahead on precision, YOLO11s is ahead on recall, mAP50, mAP50-95, and inference time.

**Table 4.** Unified batch-1 evaluation on the `YOLO_PCB` validation split (1,016 images, 2,106 instances).

| Model         | Precision | Recall | F1    | mAP50  | mAP50-95 | Inference (ms) | Total (ms) |
|---------------|----------:|-------:|------:|-------:|---------:|---------------:|-----------:|
| YOLO11s 1280  | 0.874     | 0.874  | 0.874 | 0.908  | 0.505    | 12.8           | 14.8       |
| RT-DETR-L 640 | 0.884     | 0.855  | 0.869 | 0.905  | 0.472    | 30.8           | 31.5       |

### 4.3 Per-class behavior

Aggregate mAP hides which defect categories are easy and which are not. Table 5 reports YOLO11s per-class numbers on the test split. Spur has the highest mAP50-95 score at 0.631, while Mouse_bite (0.390) and Missing_hole (0.423) have the lowest strict-localization scores. The gap between mAP50 and mAP50-95 within each class -- often more than 0.4 in absolute terms -- shows that the detector is usually finding the right region but is not always matching the defect boundary tightly enough at high IoU thresholds.

**Table 5.** Per-class YOLO11s test results.

| Class           | Precision | Recall | mAP50  | mAP50-95 |
|-----------------|----------:|-------:|-------:|---------:|
| Missing_hole    | 0.824     | 0.802  | 0.860  | 0.423    |
| Mouse_bite      | 0.860     | 0.742  | 0.844  | 0.390    |
| Open_circuit    | 0.866     | 0.891  | 0.904  | 0.534    |
| Short           | 0.881     | 0.928  | 0.940  | 0.553    |
| Spur            | 0.993     | 1.000  | 0.995  | 0.631    |
| Spurious_copper | 0.857     | 0.835  | 0.868  | 0.481    |

![Figure 3. Per-class mAP50-95 heatmap](figures/test_per_class_mAP50_95_heatmap.png)

For an inspection deployment, this per-class picture matters more than the aggregate. If missed defects are weighted by class-specific risk, Mouse_bite and Missing_hole should receive additional error analysis even though the aggregate YOLO11s mAP50 is high.

### 4.4 Precision-recall trade-off

Figure 4 plots the precision-recall position of the two detectors on the test split. The transformer sits slightly above on precision; YOLO11s sits clearly to its right on recall. For a screening workflow whose downstream stage is a human reviewer, the YOLO11s point is the more attractive one: higher recall lowers the chance that a true defect enters the next manufacturing step unflagged, while false positives mainly increase review workload.

![Figure 4. Test precision-recall scatter](figures/test_precision_recall_scatter.png)

### 4.5 Adaptive fusion and inspection cost

The detectors above are single-model baselines. As supporting analysis we also evaluate two ensembling variants that combine the YOLO11s and RT-DETR-L outputs through weighted boxes fusion. The adaptive defect-aware variant applies the project's saved class-wise fusion policy to emphasize recall on defect categories where misses are costly. The balanced WBF variant uses a more permissive fusion setting to test how much recall can be gained before false positives become burdensome. Both come from the project's Kaggle v16 supporting bundle and are not part of the headline batch-1 comparison, so they appear separately.

**Table 6.** Supporting fusion variants.

| Variant                                              | Precision | Recall | mAP50  | mAP50-95 | FP / image |
|------------------------------------------------------|----------:|-------:|-------:|---------:|-----------:|
| Adaptive defect-aware YOLO11s + RT-DETR-L            | 0.850     | 0.903  | 0.865  | 0.493    | 0.402      |
| Hybrid YOLO11s + RT-DETR-L, balanced WBF             | 0.757     | 0.943  | 0.903  | 0.509    | 0.852      |

The adaptive fusion lifts recall to 0.903 at a moderate cost in precision; the balanced WBF lifts it further to 0.943 but more than doubles the false-positive rate per image, from 0.40 to 0.85. Whether either trade is worth taking depends on what a missed defect costs relative to a human-reviewed false alarm. To make that explicit we use a simple inspection-cost score

  cost  =  FP / image  +  λ  ·  FN / image

at four values of the missed-defect penalty λ. Table 7 summarises.

**Table 7.** Inspection-cost ranking at four missed-defect penalties (lower is better).

| λ  | Best variant                                 | Cost   |
|---:|----------------------------------------------|-------:|
| 1  | YOLO11s                                      | 0.672  |
| 2  | Hybrid balanced WBF                          | 0.897  |
| 5  | Hybrid balanced WBF                          | 1.505  |
| 10 | Hybrid balanced WBF                          | 2.519  |

YOLO11s is the cost-minimising choice when false positives and missed defects are weighted equally. In the saved supporting cost table, balanced WBF becomes the lowest-cost option once missed defects are weighted more heavily than false positives. This is exactly the operating-point selection problem an embedded inspection line has to solve at integration time.

## 5. Discussion

The result is conservative and useful. YOLO11s at 1280 px is the better single-model deployment baseline for this `YOLO_PCB` split: it is more accurate by recall, F1, and mAP, and it runs 2.5x faster on the same GPU. RT-DETR-L's precision edge is genuine, but in this configuration it is not enough to pay for the recall and latency cost. A likely reason is scale: many PCB defects occupy only a small fraction of the image, and the YOLO11s configuration preserves more input detail by evaluating at 1280 px.

Two points are worth flagging. First, the mAP50 versus mAP50-95 gap is much larger than the gap between the two detectors. Both models exceed 0.88 at IoU 0.5 but neither breaks 0.55 at the stricter range; boundary-level localization, not coarse defect discovery, is the main constraint on this dataset. That suggests the next accuracy gains may come from higher-resolution crops, class-specific augmentation, or label-quality review rather than simply changing the detector family. Second, no single operating point wins across all cost regimes in the supporting cost table. The single-model detector is best when the cost asymmetry between missed defects and false alarms is mild, but an ensemble can become preferable as the line's tolerance for escapes drops. In a production setting this trade-off maps directly to labor and quality cost: false positives require additional human review, while false negatives become escaped defects that may be discovered only after more expensive downstream processing.

The comparison with classical inspection is therefore not only a question of accuracy. A learned detector can reduce manual threshold tuning and return defect boxes directly, which is useful when boards vary or when the inspection output feeds a repair station. The cost is that the model must be validated on the actual imaging setup and product mix, and it may still need a conservative recall-oriented operating point if escaped defects are expensive.

## 6. Limitations

The evaluation rests on a single `YOLO_PCB` split. We have not run cross-dataset tests on alternative PCB collections or on real factory captures, so the headline numbers should be read as in-distribution numbers. We have also not benchmarked either detector on the actual embedded targets an inspection station would use (Jetson Orin, RK3588, FPGA accelerators); the 12.6 ms YOLO11s number is a V100 inference number, so it should not be presented as embedded-device latency. The two detectors were also evaluated at the input sizes that were available from completed training runs (1280 vs 640), so the comparison is best read as one between two practical configurations rather than as an input-size-controlled architectural ablation. Finally, the fusion and inspection-cost numbers in Section 4.5 come from a separate supporting bundle and were not re-run inside the same batch-1 unified evaluator; they are presented as supporting context, not as a head-to-head extension of Table 3.

## 7. Conclusion

For an embedded PCB inspection deployment built on the six-class defect taxonomy, YOLO11s at 1280 px is the stronger single-model starting point against RT-DETR-L at 640 px: higher recall, higher F1, higher mAP, and about 2.5x faster inference on the same GPU. The localization gap between mAP50 and mAP50-95 is the most useful place to spend the next iteration of effort, and supporting inspection-cost analysis shows that a balanced WBF ensemble can become preferable when missed defects are weighted more heavily than false alarms. Future work should benchmark the model on embedded targets such as Jetson-class devices, add cross-dataset validation on real factory captures, compare against classical reference-based inspection baselines, and improve Mouse_bite and Missing_hole localization through targeted augmentation and high-resolution defect crops.

## Acknowledgments

We thank Professor Ryu for support with Nautilus / NRP access and research computing resources. This work used resources available through the National Research Platform (NRP) at the University of California, San Diego. NRP is supported in part by funding from the National Science Foundation through awards 1730158, 1540112, 1541349, 1826967, 2112167, 2100237, and 2120019, as well as additional funding from community partners. The CSUF Titan Supercomputing Center is one of the collaborative partners that contribute to NRP resources [5]. Cursor and OpenAI Codex were used as programming assistants for code navigation, refactoring suggestions, artifact consistency checks, and manuscript proofreading; the authors reviewed all suggestions and remain responsible for the code, experiments, analysis, figures, and text.

## References

1. Huang, W., Wei, P.: A PCB Dataset for Defects Detection and Classification. arXiv:1901.08204 (2019)
2. Ultralytics: YOLO11 Models. https://docs.ultralytics.com/models/yolo11/ (accessed May 2026)
3. Zhao, Y., Lv, W., Xu, S., Wei, J., Wang, G., Dang, Q., Liu, Y., Chen, J.: DETRs Beat YOLOs on Real-time Object Detection. arXiv:2304.08069 (2023)
4. Solovyev, R., Wang, W., Gabruseva, T.: Weighted Boxes Fusion: Ensembling Boxes from Different Object Detection Models. arXiv:1910.13302 (2019)
5. Weitzel, D., et al.: The National Research Platform: Stretched, Multi-Tenant, Scientific Kubernetes Cluster. arXiv:2505.22864 (2025)
