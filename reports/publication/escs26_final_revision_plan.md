# ESCS'26 Final Revision Plan

Paper ID: ESC3007

Title: Real-Time PCB Defect Detection for Embedded Visual Inspection: A Same-Split Comparison of YOLO11s and RT-DETR-L

Status: Accepted as a Regular Research Paper for ESCS'26 / CSCE'26.

## Immediate Admin Items

| Item | Date | Action |
|---|---:|---|
| Hotel reservation, if attending in person | June 19, 2026 | Optional; coordinate with advisor/funding first. |
| Presentation option | June 21, 2026 | Choose physical or virtual and email `cs@american-cse.org`. |
| Registration | June 23, 2026 | Confirm with Dr. Inventado before paying. |
| Conference | July 20-23, 2026 | Present at CSCE'26 / ESCS'26. |
| Final paper upload | Around September 10, 2026 | Springer sends upload link; upload final version within about two weeks. |

## Current PDF Audit

- PDF file reviewed: `/Users/adiiii/Desktop/escs26_yolo_pcb_submission.pdf`
- Current page count: 7 pages.
- Regular paper limit from acceptance email: maximum 15 pages.
- The paper already contains the key limitation in Section 6: YOLO11s is evaluated at 1280 px and RT-DETR-L at 640 px, so the comparison is a practical-configuration comparison rather than a controlled architecture-only ablation.
- The paper already states V100 latency is not embedded-device latency.
- The main final-version task is to strengthen these caveats so the reviewer request is unmistakably addressed.

## Must-Fix Before Final Upload

1. Confirm contact email.
   - Current PDF lists `adivd@csu.fullerton.edu`.
   - If this is not correct, replace it with `aditdhayapulay@gmail.com` or the correct institutional address.

2. Add one explicit sentence to the abstract.
   - Current abstract says both models are compared on the same split, but the reviewer may still expect the input-size caveat up front.
   - Suggested sentence:

   `Because YOLO11s and RT-DETR-L were evaluated using the completed project configurations available at 1280 px and 640 px respectively, the comparison should be interpreted as a same-split practical-configuration benchmark rather than an input-resolution-controlled architectural ablation.`

3. Strengthen Section 5 Discussion after the paragraph that mentions scale.
   - Suggested paragraph:

   `The resolution difference is an important interpretation caveat. YOLO11s was evaluated at 1280 px to preserve small PCB defect detail, whereas RT-DETR-L was evaluated at 640 px because that was the completed saved training/evaluation configuration available for the transformer-style benchmark. Higher input resolution can improve localization of small objects, so part of YOLO11s' advantage may reflect the resolution setting as well as the detector family. The result therefore supports YOLO11s-1280 as the stronger completed deployment-oriented configuration in this study, but it should not be read as a fully resolution-matched proof that the YOLO architecture is always superior to RT-DETR for PCB inspection.`

4. Strengthen Section 6 Limitations.
   - Suggested replacement/addition:

   `A resolution-controlled comparison remains future work. The present study intentionally reports completed project configurations: YOLO11s at 1280 px and RT-DETR-L at 640 px. A stricter architectural ablation would train and evaluate both detectors across matched input sizes, or compare them under a resource-normalized budget such as equal latency, equal memory, or equal FLOPs. Such an experiment would separate architecture effects from the benefits of higher spatial resolution.`

5. Strengthen future work on embedded deployment without overclaiming.
   - Suggested sentence for Conclusion:

   `Because the authors did not have access to Jetson/TensorRT hardware during this revision, target-device benchmarking is left as future work; the current latency results should be reported only as V100 batch-1 measurements.`

## Optional Improvements

- Convert the final version to a Springer LNCS template before final upload if the portal requests it.
- Add a short "Response to reviewer comments" note for the advisor:
  - We clarified that YOLO11s-1280 vs RT-DETR-L-640 is not input-size-controlled.
  - We explained why RT-DETR-L used 640 px.
  - We added matched-resolution/resource-normalized comparisons as future work.
  - We clarified that embedded target benchmarking remains future work.
