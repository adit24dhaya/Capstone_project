# ESCS'26 Acceptance and Review Record

## Paper

- Paper ID: ESC3007
- Title: Real-Time PCB Defect Detection for Embedded Visual Inspection: A Same-Split Comparison of YOLO11s and RT-DETR-L
- Authors: Aditya Varun Dhayapulay and Paul Salvador Inventado
- Category: Regular Research Paper
- Maximum length: 15 pages
- Conference dates: July 20-23, 2026
- Presentation selection deadline: June 21, 2026
- Registration deadline: June 23, 2026
- Final-paper upload link expected: approximately September 10, 2026

## Compiled Scores

| Criterion | Score |
|---|---:|
| Originality | 16/20 |
| Technical quality | 15/20 |
| Significance | 16/20 |
| Readability and organization | 15/20 |
| Relevance | 20/20 |
| Overall recommendation | 16/20 |
| Reviewer expertise | 18/20 |

## Referee A

Recommendation: Accept, with high confidence.

The reviewer described the study as a solid, reproducible, deployment-oriented comparison of YOLO11s and RT-DETR-L for six PCB defect classes. The reviewer noted that YOLO11s was evaluated at 1280 pixels while RT-DETR-L was evaluated at 640 pixels. Because higher resolution can improve small-defect localization, the observed advantage of YOLO11s may partly reflect resolution rather than architecture. The reviewer requested clarification of this limitation in the final manuscript.

## Referee B

Recommendation: Accept.

The reviewer found the benchmark practically relevant and valued its deployment perspective and per-class analysis. The reviewer requested a clearer explanation for restricting RT-DETR-L to 640 pixels and suggested matched-resolution or resource-normalized comparisons. The reviewer also noted that V100-only evaluation does not represent embedded deployment and suggested edge-hardware experiments as future work.

## Camera-Ready Response Plan

The first and fifth items directly answer the reviewers' requested
clarifications. The additional experiments and label audit strengthen the
paper beyond the minimum requested revision.

1. Clearly distinguish the accepted practical-configuration comparison from an architecture-controlled comparison.
2. Add a matched 640-pixel comparison using the same split and evaluation protocol.
3. Attempt a higher-resolution RT-DETR-L run and document any resource constraint.
4. Replace semantic per-class claims with results from the corrected canonical class mapping.
5. Keep Jetson, TensorRT, and edge-device benchmarking as future work unless measurements are actually performed.
6. State explicitly that matched resolution does not equalize epochs, training compute, memory, FLOPs, or model capacity.

## Provenance

This record summarizes the complete acceptance email received in June 2026. The original email text is stored outside the repository in the Codex attachment archive.
