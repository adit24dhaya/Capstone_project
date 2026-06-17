# Response to Reviewers

Paper ESC3007: *Real-Time PCB Defect Detection for Embedded Visual Inspection: A Same-Split Comparison of YOLO11s and RT-DETR-L*

## Referee A

**Comment:** The YOLO11s-1280 and RT-DETR-L-640 comparison is not fully resolution controlled, and the higher YOLO11s resolution may benefit small-defect localization.

**Response:** We agree. The original table is retained and explicitly described as a comparison of the completed practical configurations rather than a controlled architecture comparison. We add a separate experiment in which YOLO11s and RT-DETR-L are both trained and evaluated at 640 pixels using the same data split, hardware class, batch-1 evaluation, evaluator, and no test-time augmentation. The Discussion is revised to separate resolution effects from architectural effects.

## Referee B

**Comment:** The paper does not sufficiently explain why RT-DETR-L was restricted to 640 pixels. Matched-resolution or resource-normalized comparisons would strengthen the conclusions.

**Response:** We clarify the original computational rationale and add a matched 640-pixel comparison. We also completed RT-DETR-L training and evaluation at 1280 pixels with batch size 1 on a Tesla V100-SXM2-32GB; no 960-pixel fallback was required. The outcome and fallback decision are recorded in the experiment manifest. We explicitly state that the matched-resolution experiment is not resource normalized because the models retain different epoch counts, batch sizes, capacities, and computational costs.

**Comment:** V100-only evaluation may not represent embedded deployment conditions.

**Response:** We agree and now state this limitation more directly. The Hugging Face Space is described only as an online deployment artifact. Jetson and TensorRT measurements remain future work; the manuscript makes no claim of embedded-device performance.

## Additional Integrity Correction

During camera-ready reproducibility work, we found that an earlier dataset export changed the displayed class-name order without remapping label IDs. This can misname the per-class rows for Missing hole, Mouse bite, and Spur while leaving class-agnostic aggregate detection metrics unchanged. The resolution-control pipeline now uses a canonical semantic mapping and validates labels before training. Camera-ready per-class claims and figures are generated only from corrected data. The accepted aggregate results remain archived as the original practical-configuration baseline.

## Verification Checklist

- [x] Original aggregate baseline reproduced within 0.001 on the legacy reconstruction.
- [x] Corrected dataset counts and semantic class mapping verified.
- [x] YOLO11s-640 one-epoch smoke test completed.
- [x] RT-DETR-L-640 one-epoch smoke test completed.
- [x] RT-DETR-L-1280 one-epoch smoke test completed without CUDA OOM.
- [x] YOLO11s-640 full 50-epoch run completed.
- [x] RT-DETR-L-640 full 10-epoch run completed.
- [x] Matched-resolution artifacts re-collected after the original temporary PVCs were removed.
- [x] RT-DETR-L-1280 full 10-epoch run completed without CUDA OOM.
- [x] RT-DETR-L-1280 aggregate, per-class, training, dataset, and environment records saved locally.
- [x] Final combined aggregate and per-class CSV files saved locally.
- [x] Success and failure figures regenerated from the retained final YOLO11s-640 predictions.
- [x] Camera-ready claims checked against the experiment manifest.
