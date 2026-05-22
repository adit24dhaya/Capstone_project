#!/usr/bin/env python3
"""Build lightweight paper-ready summaries from local experiment artifacts."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports" / "publication"
KAGGLE_V16 = ROOT / "kaggle_cli_output" / "version16_artifacts"
YOLO11M_ARTIFACT = ROOT / "local_artifacts" / "yolo11m_960_publication_outputs_20260522_061024"
YOLO11M_RUN = YOLO11M_ARTIFACT / "outputs" / "nautilus" / "runs" / "detector_train" / "yolo11m_960_publication"


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def f(value: Any, digits: int = 4) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def find_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str] | None:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    return None


def parse_yolo11m_test_speed() -> str:
    log_path = YOLO11M_ARTIFACT / "logs" / "yolo11m_960_publication_overnight.log"
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    speeds = re.findall(r"Speed: .*? ([0-9.]+)ms inference", text)
    return speeds[-1] if speeds else ""


def load_yolo11m_rows() -> list[dict[str, Any]]:
    metrics = read_csv(YOLO11M_RUN / "detector_train_metrics.csv")
    speed_ms = parse_yolo11m_test_speed()
    rows = []
    for row in metrics:
        rows.append(
            {
                "model": "YOLO11m 960",
                "family": "CNN / YOLO stronger baseline",
                "source": "Nautilus RTX 2080 Ti run",
                "split": row["split"],
                "precision": f(row["precision"]),
                "recall": f(row["recall"]),
                "mAP50": f(row["mAP50"]),
                "mAP50_95": f(row["mAP50_95"]),
                "inference_ms": f(speed_ms, 1) if row["split"] == "test" else "",
                "fp_per_image": "",
                "notes": "imgsz=960, batch=2, early stopped at 51 epochs, best.pt used",
            }
        )
    return rows


def load_comparison_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    architecture = read_csv(KAGGLE_V16 / "architecture_comparison.csv")
    balanced = read_csv(KAGGLE_V16 / "hybrid_final_balanced_test_metrics.csv")
    adaptive = read_csv(KAGGLE_V16 / "adaptive_defect_aware_hybrid_test_metrics.csv")

    for model in ["YOLO11s", "RT-DETR-L", "Hybrid YOLO11s + RT-DETR-L"]:
        row = find_row(architecture, model=model, split="test")
        if not row:
            continue
        rows.append(
            {
                "model": model,
                "family": row.get("family", ""),
                "source": "Kaggle v16 final analysis",
                "split": "test",
                "precision": f(row.get("precision")),
                "recall": f(row.get("recall")),
                "mAP50": f(row.get("mAP50")),
                "mAP50_95": f(row.get("mAP50_95")),
                "inference_ms": f(row.get("speed_inference_ms"), 1),
                "fp_per_image": f(row.get("false_positives_per_image")),
                "notes": "Final Kaggle detector comparison row",
            }
        )

    for source_row, source_name in [
        (balanced[0] if balanced else None, "Kaggle v16 final balanced WBF"),
        (adaptive[0] if adaptive else None, "Kaggle v16 adaptive fusion"),
    ]:
        if not source_row:
            continue
        rows.append(
            {
                "model": source_row.get("model", ""),
                "family": source_row.get("family", ""),
                "source": source_name,
                "split": source_row.get("split", "test"),
                "precision": f(source_row.get("precision")),
                "recall": f(source_row.get("recall")),
                "mAP50": f(source_row.get("mAP50")),
                "mAP50_95": f(source_row.get("mAP50_95")),
                "inference_ms": "",
                "fp_per_image": f(source_row.get("fp_per_image")),
                "notes": "Fusion result used for novelty discussion, not the highest-accuracy row",
            }
        )

    rows.extend(load_yolo11m_rows())
    rows.sort(key=lambda item: (item["split"] != "test", item["model"]))
    return rows


def best_yolo11m_epoch() -> dict[str, str] | None:
    rows = read_csv(YOLO11M_RUN / "results.csv")
    if not rows:
        return None
    metric_key = "metrics/mAP50-95(B)"
    return max(rows, key=lambda row: float(row.get(metric_key, "0") or 0))


def write_results_markdown(comparison_rows: list[dict[str, Any]]) -> None:
    best_row = max(
        [row for row in comparison_rows if row["split"] == "test" and row["mAP50_95"]],
        key=lambda row: float(row["mAP50_95"]),
    )
    yolo11s = next(row for row in comparison_rows if row["model"] == "YOLO11s" and row["split"] == "test")
    yolo11m = next(row for row in comparison_rows if row["model"] == "YOLO11m 960" and row["split"] == "test")
    delta = float(yolo11m["mAP50_95"]) - float(yolo11s["mAP50_95"])
    best_epoch = best_yolo11m_epoch()

    lines = [
        "# ESCS'26 Publication Results Summary",
        "",
        "## Current Main Result",
        "",
        (
            f"The strongest completed model is **{best_row['model']}** on the test split "
            f"with precision `{best_row['precision']}`, recall `{best_row['recall']}`, "
            f"mAP50 `{best_row['mAP50']}`, and mAP50-95 `{best_row['mAP50_95']}`."
        ),
        "",
        (
            "Compared with the Kaggle YOLO11s baseline, YOLO11m 960 improves "
            f"test mAP50-95 from `{yolo11s['mAP50_95']}` to `{yolo11m['mAP50_95']}` "
            f"for an absolute gain of `{delta:.4f}`."
        ),
        "",
        "## Paper Claim To Use",
        "",
        (
            "A medium YOLO detector trained at 960-pixel resolution improves PCB defect "
            "detection accuracy while remaining real-time on a commodity RTX 2080 Ti-class GPU. "
            "The broader system adds transformer-style and adaptive fusion experiments for "
            "embedded inspection tradeoff analysis."
        ),
        "",
        "## YOLO11m 960 Training Note",
        "",
        "The Nautilus run used `imgsz=960`, `batch=2`, `workers=0`, and early stopped after 51 epochs.",
    ]
    if best_epoch:
        lines.extend(
            [
                (
                    f"The best epoch in `results.csv` by validation mAP50-95 was epoch "
                    f"`{best_epoch.get('epoch')}` with mAP50-95 `{f(best_epoch.get('metrics/mAP50-95(B)'))}`."
                )
            ]
        )
    lines.extend(
        [
            "",
            "## Important Caveat",
            "",
            (
                "Kaggle v16 and Nautilus YOLO11m runs were executed on different GPU environments. "
                "Use accuracy metrics for model comparison, and discuss latency as hardware-specific."
            ),
            "",
            "## Evidence Files",
            "",
            "- `reports/publication/model_comparison.csv`",
            "- `reports/publication/yolo11m_960_training_summary.csv`",
            "- Local artifact backup: `local_artifacts/yolo11m_960_publication_outputs_20260522_061024/`",
        ]
    )
    (REPORT_DIR / "results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outline() -> None:
    outline = """# ESCS'26 Paper Outline

Working title:
Cost-Aware Adaptive Detector Fusion for Real-Time Embedded PCB Defect Inspection

## Abstract

Write 100-120 words. Mention PCB defect detection, embedded inspection, YOLO11m 960, RT-DETR, Faster R-CNN baseline, adaptive fusion, real-time latency, and inspection-cost metrics.

## 1. Introduction

- PCB manufacturing needs fast, reliable surface-defect inspection.
- False negatives are costly because missed defects can propagate downstream.
- Embedded deployment requires balancing accuracy, latency, and inspection burden.
- Contribution summary:
  - Strong YOLO11m 960 real-time detector baseline.
  - Comparison against YOLO11s, RT-DETR-L, and fusion variants.
  - Cost-aware and adaptive fusion analysis for industrial inspection.
  - Deployment-oriented discussion with ONNX/TensorRT readiness.

## 2. Related Work

- Classical PCB visual inspection.
- YOLO-family PCB defect detectors.
- Transformer-based object detectors for industrial inspection.
- Ensemble/fusion and calibration for dependable inspection.

## 3. Method

- Dataset and six-class taxonomy.
- YOLO conversion and train/val/test split.
- Detector baselines: YOLO11s, YOLO11m 960, RT-DETR-L, Faster R-CNN.
- Adaptive fusion policy and inspection-cost metric.
- Evaluation metrics: precision, recall, mAP50, mAP50-95, FP/image, latency.

## 4. Experiments

- Hardware and environment table.
- Training settings table.
- Model comparison table from `model_comparison.csv`.
- Per-class table for YOLO11m 960 and fusion models.
- Latency and deployment analysis.

## 5. Results and Discussion

- YOLO11m 960 is the strongest current accuracy result.
- Fusion improves the precision/recall tradeoff discussion but is not always the highest mAP model.
- Missing_hole remains easiest; Short/Spur localization is harder under stricter mAP50-95.
- Discuss 2080 Ti feasibility and why A100 is requested for final high-resolution ablations.

## 6. Limitations

- Single primary PCB dataset unless cross-dataset conversion is completed.
- Pseudo segmentation masks are box-derived, not true pixel masks.
- Latency differs across Kaggle and Nautilus hardware.

## 7. Conclusion

- Summarize real-time embedded inspection result and practical cost-aware evaluation.
"""
    (REPORT_DIR / "paper_outline.md").write_text(outline, encoding="utf-8")


def write_checklist() -> None:
    checklist = """# Final Experiment Checklist

## Completed Evidence

- YOLO11s baseline from Kaggle v16.
- RT-DETR-L transformer-style baseline from Kaggle v16.
- Hybrid YOLO11s + RT-DETR-L fusion experiments from Kaggle v16.
- Adaptive defect-aware fusion, defect-size, calibration, robustness, and inspection-cost artifacts from Kaggle v16.
- YOLO11m 960 Nautilus/RTX 2080 Ti result with saved `best.pt`, logs, curves, and metrics.

## Still Worth Running

1. Faster R-CNN baseline on the current split.
2. Optional YOLO11m 960 ONNX export for deployment evidence.
3. Optional final visualization set using YOLO11m 960 predictions.
4. Optional cross-dataset test if DeepPCB/DsPCBSD+/Mendeley YOLO data is ready.

## Faster R-CNN Command

```bash
cd ~/Capstone_project
mkdir -p ~/logs ~/backups

nohup bash -lc '
python tools/run_nautilus_experiments.py \\
  --experiment faster_rcnn \\
  --data-root ~/data \\
  --output-dir ~/outputs/nautilus \\
  --frcnn-epochs 5 \\
  --frcnn-batch 2 \\
  --workers 0

EXIT_CODE=$?

cd ~
zip -r backups/faster_rcnn_outputs_$(date +%Y%m%d_%H%M%S).zip \\
  outputs/nautilus/runs/faster_rcnn \\
  logs/faster_rcnn.log

echo "EXIT_CODE=$EXIT_CODE"
echo "Backup finished at $(date)"
exit $EXIT_CODE
' > ~/logs/faster_rcnn.log 2>&1 &
```

## Paper Priority

Start writing now using YOLO11m 960 as the main model and fusion/cost analysis as the novelty layer. Treat Faster R-CNN and cross-dataset results as additions if they finish before submission.
"""
    (REPORT_DIR / "final_experiment_checklist.md").write_text(checklist, encoding="utf-8")


def write_training_summary() -> None:
    rows = read_csv(YOLO11M_RUN / "results.csv")
    if not rows:
        return
    fieldnames = [
        "epoch",
        "train_box_loss",
        "train_cls_loss",
        "train_dfl_loss",
        "precision",
        "recall",
        "mAP50",
        "mAP50_95",
    ]
    compact = []
    for row in rows:
        compact.append(
            {
                "epoch": row.get("epoch", ""),
                "train_box_loss": f(row.get("train/box_loss")),
                "train_cls_loss": f(row.get("train/cls_loss")),
                "train_dfl_loss": f(row.get("train/dfl_loss")),
                "precision": f(row.get("metrics/precision(B)")),
                "recall": f(row.get("metrics/recall(B)")),
                "mAP50": f(row.get("metrics/mAP50(B)")),
                "mAP50_95": f(row.get("metrics/mAP50-95(B)")),
            }
        )
    write_csv(REPORT_DIR / "yolo11m_960_training_summary.csv", compact, fieldnames)


def write_metadata() -> None:
    metadata = {
        "generated_from": {
            "kaggle_v16": str(KAGGLE_V16.relative_to(ROOT)),
            "yolo11m_960_artifact": str(YOLO11M_ARTIFACT.relative_to(ROOT)),
        },
        "primary_completed_result": "YOLO11m 960 test mAP50-95 0.5347",
        "target_venue": "ESCS'26",
        "deadline": "2026-05-27",
    }
    (REPORT_DIR / "publication_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_rows = load_comparison_rows()
    fieldnames = [
        "model",
        "family",
        "source",
        "split",
        "precision",
        "recall",
        "mAP50",
        "mAP50_95",
        "inference_ms",
        "fp_per_image",
        "notes",
    ]
    write_csv(REPORT_DIR / "model_comparison.csv", comparison_rows, fieldnames)
    write_training_summary()
    write_results_markdown(comparison_rows)
    write_outline()
    write_checklist()
    write_metadata()
    print(f"Wrote publication package to {REPORT_DIR}")


if __name__ == "__main__":
    main()
