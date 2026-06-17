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

## Current Manuscript Audit

- PDF file reviewed: `/Users/adiiii/Desktop/escs26_yolo_pcb_submission.pdf`
- Accepted PDF page count: 7 pages.
- Regular paper limit from acceptance email: maximum 15 pages.
- The LNCS camera-ready source preserves the accepted practical comparison and explicitly identifies YOLO11s-1280 versus RT-DETR-L-640 as not resolution controlled.
- Separate matched-resolution YOLO11s-640 and RT-DETR-L-640 experiments were completed.
- RT-DETR-L-1280 completed for 10 epochs at batch size 1 on a V100-SXM2-32GB; the 960-pixel fallback was not required.
- The manuscript states that matched resolution is not the same as equal training compute, memory, latency, FLOPs, or model capacity.
- The manuscript states that V100 latency is not embedded-device latency and makes no Jetson or TensorRT performance claim.

## Completed Camera-Ready Work

1. The matched 640-pixel experiments were collected into CSV records and merged into `reports/publication/resolution_control_results/merged_controlled/`.
2. The success and failure figures were copied into `reports/publication/camera_ready_figures/`.
3. Manuscript tables and numeric macros are rendered from saved CSV files through `tools/render_resolution_control_tables.py`.
4. The LNCS PDF build is checked by `tools/build_escs26_camera_ready.sh`, including the 15-page limit, placeholder detection, and unverified embedded-claim guard.
5. The reviewer response and compliance audit document how the manuscript addresses the resolution-control and embedded-benchmarking comments.
6. The source retains the active institutional address `adivd@csu.fullerton.edu`; change it only if Dr. Inventado requests a different publication contact.

## Remaining Admin Items

- Email `cs@american-cse.org` with the virtual presentation selection by June 21, 2026, if confirmation has not already been sent.
- Coordinate registration funding with Dr. Inventado before the June 23, 2026 deadline.
- Upload the Springer camera-ready paper only after the official Springer link arrives, expected around September 10, 2026.

## Implemented Reviewer Changes

- The camera-ready source uses the Springer LNCS class.
- The abstract, Methods, Results, Discussion, and Limitations distinguish the accepted practical configurations from the matched-resolution comparison.
- The paper adds the requested matched-resolution experiment rather than leaving it only as future work.
- RT-DETR-L was also tested at 1280 pixels to answer the advisor's question directly.
- Resource-normalized comparison remains future work and is stated as a limitation.
- Embedded target benchmarking remains future work.
