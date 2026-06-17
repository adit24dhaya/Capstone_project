# ESCS'26 Camera-Ready Compliance Audit

Audit date: June 16, 2026

Paper: ESC3007, *Real-Time PCB Defect Detection for Embedded Visual Inspection: A Same-Split Comparison of YOLO11s and RT-DETR-L*

## Overall Status

**Paper-side requirements complete; administrative items remain external.** The reviewer and advisor requests are represented in the camera-ready source, final matched-resolution artifacts are retained locally, generated tables and figures are rebuilt from saved CSV files, and the camera-ready PDF builds successfully at 9 pages.

## Request-by-Request Check

| Source | Request | Status | Evidence |
|---|---|---:|---|
| Referee A | Clarify that YOLO11s-1280 versus RT-DETR-L-640 is not fully resolution controlled | PASS | Abstract, Introduction, Results, and Discussion in `escs26_camera_ready.tex` |
| Referee B | Explain why RT-DETR-L originally used 640 pixels | PASS | Models/Training and Discussion sections |
| Referee B | Consider matched-resolution or resource-normalized comparison | PASS / LIMITED | Matched 640-pixel runs completed; manuscript explicitly states that training compute and capacity are not normalized |
| Referee B | Treat edge-hardware evaluation as future work | PASS | Deployment Scope, Limitations, and Conclusion contain no Jetson/TensorRT performance claim |
| Dr. Inventado | Reduce YOLO11s resolution or raise RT-DETR-L resolution | PASS | YOLO11s-640 and RT-DETR-L-640 completed; RT-DETR-L-1280 completed without CUDA OOM |
| Dr. Inventado | Keep edge benchmarking as a short future-work item | PASS | Jetson/TensorRT are future work only |
| ESCS'26 | Use Springer LNCS format and remain within 15 pages | PASS | Source uses `llncs`; final PDF builds at 9 pages |
| ESCS'26 | Improve organization and English before final upload | PASS | Structure and wording revised in the camera-ready source; final advisor proofread is still recommended before upload |
| ESCS'26 | Select presentation mode by June 21 and register by June 23 | EXTERNAL | Email drafts exist, but sending, conference confirmation, and payment/funding cannot be verified from repository files |

## Metric Evidence

The accepted aggregate values match `archived_baseline_metrics.csv`:

- YOLO11s-1280 test: precision 0.8802885, recall 0.8663570, mAP50 0.9018129, mAP50-95 0.5021455.
- RT-DETR-L-640 test: precision 0.8862587, recall 0.8387827, mAP50 0.8869201, mAP50-95 0.4701086.

The completed matched and high-resolution results are retained locally in `resolution_control_results/`:

- YOLO11s-640 test: precision 0.8704202, recall 0.8519801, F1 0.8611015.
- YOLO11s-640 test mAP50 0.8920552 and mAP50-95 0.4816808.
- YOLO11s-640 total batch-1 latency 13.1528 ms on Tesla V100-SXM2-32GB.
- RT-DETR-L-640 test: precision 0.8674367, recall 0.8390429, F1 0.8530036.
- RT-DETR-L-640 test mAP50 0.8812886 and mAP50-95 0.4621181.
- RT-DETR-L-640 total batch-1 latency 49.0268 ms on Tesla V100-SXM2-32GB.

- RT-DETR-L-1280 test: precision 0.7660249, recall 0.5343558, F1 0.6295539.
- RT-DETR-L-1280 test mAP50 0.6039372 and mAP50-95 0.2952862.
- RT-DETR-L-1280 total batch-1 latency 74.3823 ms on Tesla V100-SXM2-32GB.
- Checkpoint SHA-256: `3f1b0e362305ba7c8a7f137675e7e1a20cb0e26d0f7034b0e5556096c5050e42`.
- Dataset: 5,551 train, 1,016 validation, and 1,016 test images; 2,106 validation and 2,179 test instances.
- Training resumed after an interruption; the retained `results.csv` contains epochs 6-10, while the final checkpoint, configuration, evaluation CSV files, and environment manifest are retained. This does not invalidate the evaluation, but the partial training-history record should not be described as a complete epoch-by-epoch log.

These high-resolution results are valid but substantially weaker and slower than the completed 640-pixel RT-DETR-L run. The paper reports that outcome without implying that higher resolution improved RT-DETR-L.

## Metric Definitions and References

- Precision, recall, F1, AP, mAP50, and mAP50-95 are computed by the pinned Ultralytics evaluator.
- mAP50-95 is described as mean AP over IoU thresholds 0.50 through 0.95 in increments of 0.05, consistent with the COCO convention.
- The camera-ready bibliography now includes the COCO paper and the official Ultralytics metric documentation.
- DOI metadata was cross-checked for RT-DETR, Weighted Boxes Fusion, DsPCBSD+, COCO, and the National Research Platform paper.
- All citation keys used by the manuscript exist in the bibliography; no bibliography entry is unused.
- The final camera-ready build completes at 9 pages with real generated tables and figures. Placeholder tables/macros are absent.

## Integrity Corrections

- The accepted aggregate metrics remain valid because class-ID permutation does not change class-agnostic aggregate precision, recall, or mAP.
- The accepted semantic per-class labels were vulnerable to a legacy class-order mismatch. New per-class tables and figures use the canonical mapping only:
  `Missing_hole`, `Mouse_bite`, `Open_circuit`, `Short`, `Spur`, `Spurious_copper`.
- The matched 640-pixel comparison controls split, resolution, evaluator, hardware class, evaluation batch size, and TTA. It does not control epoch count, training batch size, parameters, FLOPs, memory, or training compute.
- The Hugging Face Space is described only as an online deployment artifact.

## Remaining Submission Items

1. Confirm that the virtual-presentation email was sent to `cs@american-cse.org` and acknowledged before June 21, 2026.
2. Confirm registration funding/payment with Dr. Inventado before June 23, 2026.
3. When the Springer upload link arrives around September 10, 2026, upload the final camera-ready paper within the two-week window.
4. Do one final advisor proofread before upload.

The paper artifact currently builds as `reports/publication/escs26_camera_ready.pdf`.
