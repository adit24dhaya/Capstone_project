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
YOLO11L_ARTIFACT = ROOT / "local_artifacts" / "yolo11l_1280_publication_outputs_20260522_092005"
YOLO11L_RUN = YOLO11L_ARTIFACT / "outputs" / "nautilus" / "runs" / "detector_train" / "yolo11l_1280_publication"
YOLO11L_ONNX_ARTIFACT = ROOT / "local_artifacts" / "yolo11l_1280_onnx_export_20260522_102315"
YOLO11L_ONNX = (
    YOLO11L_ONNX_ARTIFACT
    / "outputs"
    / "nautilus"
    / "runs"
    / "detector_train"
    / "yolo11l_1280_publication"
    / "weights"
    / "best.onnx"
)
YOLO11L_SELECTED_FIGURES = ROOT / "local_artifacts" / "outputs" / "nautilus" / "runs" / "paper_figures" / "yolo11l_1280_selected_examples"
CHAMPION_EVAL_SWEEP = REPORT_DIR / "champion_eval_sweep.csv"


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


def parse_inference_ms(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    speeds = re.findall(r"Speed: .*? ([0-9.]+)ms inference", text)
    return speeds[-1] if speeds else ""


def load_nautilus_detector_rows(
    *,
    model_label: str,
    family: str,
    run_dir: Path,
    log_path: Path,
    notes: str,
) -> list[dict[str, Any]]:
    metrics = read_csv(run_dir / "detector_train_metrics.csv")
    speed_ms = parse_inference_ms(log_path)
    rows = []
    for row in metrics:
        rows.append(
            {
                "model": model_label,
                "family": family,
                "source": f"Nautilus RTX 2080 Ti {run_dir.name}",
                "split": row["split"],
                "precision": f(row["precision"]),
                "recall": f(row["recall"]),
                "mAP50": f(row["mAP50"]),
                "mAP50_95": f(row["mAP50_95"]),
                "inference_ms": f(speed_ms, 1) if row["split"] == "test" else "",
                "fp_per_image": "",
                "notes": notes,
            }
        )
    return rows


def load_yolo11l_rows() -> list[dict[str, Any]]:
    if not YOLO11L_RUN.exists():
        return []
    return load_nautilus_detector_rows(
        model_label="YOLO11l 1280",
        family="CNN / YOLO primary detector",
        run_dir=YOLO11L_RUN,
        log_path=YOLO11L_ARTIFACT / "logs" / "yolo11l_1280_publication_overnight.log",
        notes="imgsz=1280, batch=1, workers=0, best.pt official val+test",
    )


def load_yolo11m_rows() -> list[dict[str, Any]]:
    if not YOLO11M_RUN.exists():
        return []
    return load_nautilus_detector_rows(
        model_label="YOLO11m 960",
        family="CNN / YOLO medium-resolution ablation",
        run_dir=YOLO11M_RUN,
        log_path=YOLO11M_ARTIFACT / "logs" / "yolo11m_960_publication_overnight.log",
        notes="imgsz=960, batch=2, early stopped, best.pt used",
    )


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

    rows.extend(load_yolo11l_rows())
    rows.extend(load_yolo11m_rows())
    rows.sort(key=lambda item: (item["split"] != "test", -float(item["mAP50_95"] or 0), item["model"]))
    return rows


def best_training_epoch(run_dir: Path) -> dict[str, str] | None:
    rows = read_csv(run_dir / "results.csv")
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
    yolo11m = next(
        (row for row in comparison_rows if row["model"] == "YOLO11m 960" and row["split"] == "test"),
        None,
    )
    delta_vs_yolo11s = float(best_row["mAP50_95"]) - float(yolo11s["mAP50_95"])
    delta_vs_yolo11m = (
        float(best_row["mAP50_95"]) - float(yolo11m["mAP50_95"]) if yolo11m else None
    )
    best_epoch = best_training_epoch(YOLO11L_RUN if "YOLO11l" in best_row["model"] else YOLO11M_RUN)

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
            f"Compared with the Kaggle YOLO11s baseline (test mAP50-95 `{yolo11s['mAP50_95']}`), "
            f"the gain is `{delta_vs_yolo11s:.4f}` absolute."
        ),
    ]
    if yolo11m and delta_vs_yolo11m is not None:
        lines.append(
            (
                f"Compared with YOLO11m-960 (test mAP50-95 `{yolo11m['mAP50_95']}`), "
                f"the gain is `{delta_vs_yolo11m:.4f}` absolute."
            )
        )
    lines.extend(
        [
        "",
        "## Paper Claim To Use",
        "",
        (
            "A large YOLO detector at 1280px achieves strong six-class PCB defect detection "
            "on a commodity RTX 2080 Ti (test mAP50 near 0.99, mAP50-95 near 0.58). "
            "YOLO11m-960 and Kaggle fusion experiments support ablation and deployment tradeoffs."
        ),
        "",
        f"## {best_row['model']} Training Note",
        "",
        "Nautilus RTX 2080 Ti; see `results.csv` and overnight log in local artifact backup.",
        ]
    )
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
            "## Champion Evaluation Sweep",
            "",
            (
                "On May 24, 2026, the restored YOLO11l 1280 champion checkpoint was re-evaluated "
                "on Nautilus V100 at 1280 and 1536 image sizes, with and without Ultralytics "
                "test-time augmentation (`augment=True`). The best test mAP50-95 remained the "
                "original 1280/no-TTA setting: precision `0.9832`, recall `0.9881`, "
                "mAP50 `0.9898`, mAP50-95 `0.5769`."
            ),
            "",
            (
                "The sweep is useful for the paper because it shows that simply increasing "
                "inference size to 1536 or enabling TTA did not improve strict localization. "
                "Use 1280/no-TTA as the headline configuration, and mention the sweep as a "
                "negative ablation supporting the chosen deployment setting."
            ),
            "",
            "## Important Caveat",
            "",
            (
                "Kaggle v16 and Nautilus YOLO11 runs were executed on different GPU environments. "
                "Use accuracy metrics for model comparison, and discuss latency as hardware-specific."
            ),
            "",
            "## Evidence Files",
            "",
            "- `reports/publication/model_comparison.csv`",
            "- `reports/publication/champion_eval_sweep.csv`",
            "- `reports/publication/yolo11l_1280_training_summary.csv`",
            "- `reports/publication/yolo11m_960_training_summary.csv`",
            f"- Local artifact backup: `{YOLO11L_ARTIFACT.relative_to(ROOT)}/`",
            f"- ONNX deployment artifact: `{YOLO11L_ONNX.relative_to(ROOT)}`" if YOLO11L_ONNX.exists() else "- ONNX deployment artifact: pending",
            f"- Selected prediction figures: `{YOLO11L_SELECTED_FIGURES.relative_to(ROOT)}/`" if YOLO11L_SELECTED_FIGURES.exists() else "- Selected prediction figures: pending",
            f"- Prior ablation: `{YOLO11M_ARTIFACT.relative_to(ROOT)}/`",
        ]
    )
    (REPORT_DIR / "results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outline() -> None:
    outline = """# ESCS'26 Paper Outline

Working title:
Cost-Aware Adaptive Detector Fusion for Real-Time Embedded PCB Defect Inspection

## Abstract

Write 100-120 words. Lead with YOLO11l @ 1280 (test mAP50 ~0.99, mAP50-95 ~0.58). Mention RTX 2080 Ti, YOLO11m-960 ablation, optional RT-DETR/fusion/cost analysis.

## 1. Introduction

- PCB manufacturing needs fast, reliable surface-defect inspection.
- False negatives are costly because missed defects can propagate downstream.
- Embedded deployment requires balancing accuracy, latency, and inspection burden.
- Contribution summary:
  - Strong YOLO11l 1280 detector as primary result; YOLO11m-960 as efficiency ablation.
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
- Detector baselines: YOLO11l 1280 (primary), YOLO11s, YOLO11m 960, RT-DETR-L, Faster R-CNN (optional).
- Adaptive fusion policy and inspection-cost metric.
- Evaluation metrics: precision, recall, mAP50, mAP50-95, FP/image, latency.

## 4. Experiments

- Hardware and environment table.
- Training settings table.
- Model comparison table from `model_comparison.csv`.
- Champion inference sweep table from `champion_eval_sweep.csv`.
- Per-class table for YOLO11l 1280 and fusion models.
- Latency and deployment analysis.

## 5. Results and Discussion

- YOLO11l 1280 is the strongest current accuracy result.
- The 1280/no-TTA champion setting beat 1536 and TTA variants in strict test mAP50-95, so the final detector setting is empirically justified rather than arbitrary.
- Fusion improves the precision/recall tradeoff discussion but is not always the highest mAP model.
- Missing_hole remains easiest; Short/Spur localization is harder under stricter mAP50-95.
- Discuss 2080 Ti feasibility and why A100 is requested for final high-resolution ablations.

## 6. Limitations

- Single primary PCB dataset unless cross-dataset conversion is completed.
- Pseudo segmentation masks are box-derived, not true pixel masks.
- Latency differs across Kaggle and Nautilus hardware.

## 7. Conclusion

- Summarize real-time embedded inspection result and practical cost-aware evaluation.

## Acknowledgments

- Thank Dr. Paul Salvador Inventado for guidance and feedback throughout the project.
- Thank Professor Ryu for support with Nautilus/NRP access and research computing resources.
- Include the required NRP/Nautilus acknowledgment exactly:

```text
This work used resources available through the National Research Platform (NRP) at the University of California, San Diego. NRP has been developed, and is supported in part, by funding from National Science Foundation, from awards 1730158, 1540112, 1541349, 1826967, 2112167, 2100237, and 2120019, as well as additional funding from community partners. The CSUF Titan Supercomputing Center is one of the collaborative partners to contribute to NRP resources.
```

- Cite: The National Research Platform: Stretched, Multi-Tenant, Scientific Kubernetes Cluster.
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
- YOLO11l 1280 Nautilus/RTX 2080 Ti result in `local_artifacts/yolo11l_1280_publication_outputs_20260522_092005/`.
- YOLO11l 1280 ONNX export in `local_artifacts/yolo11l_1280_onnx_export_20260522_102315/`.
- YOLO11l 1280 champion eval sweep on Nautilus V100, saved in `reports/publication/champion_eval_sweep.csv`; 1280/no-TTA remains best.
- Six selected prediction examples in `local_artifacts/outputs/nautilus/runs/paper_figures/yolo11l_1280_selected_examples/`.

## Still Worth Running

1. Faster R-CNN baseline on the current split.
2. Optional cross-dataset test if DeepPCB/DsPCBSD+/Mendeley YOLO data is ready.
3. Unified same-workflow evaluation table for all saved YOLO checkpoints, then stop tuning unless it reveals a reproducibility gap.

## Repeatable Export Command

```bash
python tools/run_nautilus_experiments.py \\
  --experiment detector_export \\
  --data-root ~/data \\
  --output-dir ~/outputs/nautilus \\
  --yolo-weights ~/outputs/nautilus/runs/detector_train/yolo11l_1280_publication/weights/best.pt \\
  --imgsz 1280 \\
  --export-format onnx
```

## Repeatable Visual Examples Command

```bash
python tools/run_nautilus_experiments.py \\
  --experiment visual_examples \\
  --data-root ~/data \\
  --output-dir ~/outputs/nautilus \\
  --yolo-weights ~/outputs/nautilus/runs/detector_train/yolo11l_1280_publication/weights/best.pt \\
  --run-name yolo11l_1280_selected_examples \\
  --imgsz 1280 \\
  --split test \\
  --prediction-save-limit 6
```

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

## Unified Paper Evaluation Command

Run this after restoring the saved checkpoint folders under `~/outputs/nautilus/runs/detector_train/`. It re-evaluates every available YOLO checkpoint with the same dataset YAML, split, evaluator, and workers setting, then writes CSV/JSON summaries plus paper figures.

```bash
cd ~/Capstone_project
git pull
mkdir -p ~/logs ~/backups

python tools/run_nautilus_experiments.py \\
  --experiment paper_unified_eval \\
  --data-root ~/data \\
  --output-dir ~/outputs/nautilus \\
  --workers 0 \\
  --device 0 \\
  --prediction-save-limit 6 \\
  2>&1 | tee ~/logs/paper_unified_eval.log

cd ~
zip -r backups/paper_unified_eval_$(date +%Y%m%d_%H%M%S).zip \\
  outputs/nautilus/runs/paper_unified_eval \\
  logs/paper_unified_eval.log
```

Expected outputs:

- `~/outputs/nautilus/runs/paper_unified_eval/paper_unified_eval_metrics.csv`
- `~/outputs/nautilus/runs/paper_unified_eval/paper_unified_eval_per_class.csv`
- `~/outputs/nautilus/runs/paper_unified_eval/paper_unified_eval_summary.json`
- `~/outputs/nautilus/runs/paper_unified_eval/paper_unified_eval_summary.md`
- `~/outputs/nautilus/runs/paper_unified_eval/figures/*.png`
- `~/outputs/nautilus/runs/paper_unified_eval/qualitative_examples_*/`

## Paper Priority

Start writing now using YOLO11l @ 1280 as the main detector. Use YOLO11m-960 and Kaggle fusion as ablations. Treat Faster R-CNN and cross-dataset as optional before ESCS (May 27).

## Required Acknowledgments

- Thank Dr. Paul Salvador Inventado for guidance and feedback throughout the project.
- Thank Professor Ryu for support with Nautilus/NRP access and research computing resources.
- Include the required NRP/Nautilus acknowledgment exactly in the paper acknowledgments section:

```text
This work used resources available through the National Research Platform (NRP) at the University of California, San Diego. NRP has been developed, and is supported in part, by funding from National Science Foundation, from awards 1730158, 1540112, 1541349, 1826967, 2112167, 2100237, and 2120019, as well as additional funding from community partners. The CSUF Titan Supercomputing Center is one of the collaborative partners to contribute to NRP resources.
```

- Add the NRP citation: The National Research Platform: Stretched, Multi-Tenant, Scientific Kubernetes Cluster.
"""
    (REPORT_DIR / "final_experiment_checklist.md").write_text(checklist, encoding="utf-8")


def write_training_summary(run_dir: Path, output_name: str) -> None:
    rows = read_csv(run_dir / "results.csv")
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
    write_csv(REPORT_DIR / output_name, compact, fieldnames)


def write_metadata() -> None:
    metadata = {
        "generated_from": {
            "kaggle_v16": str(KAGGLE_V16.relative_to(ROOT)),
            "yolo11m_960_artifact": str(YOLO11M_ARTIFACT.relative_to(ROOT)),
            "yolo11l_1280_artifact": str(YOLO11L_ARTIFACT.relative_to(ROOT)),
            "yolo11l_1280_onnx_artifact": str(YOLO11L_ONNX_ARTIFACT.relative_to(ROOT)),
            "yolo11l_1280_selected_figures": str(YOLO11L_SELECTED_FIGURES.relative_to(ROOT)),
            "champion_eval_sweep": str(CHAMPION_EVAL_SWEEP.relative_to(ROOT)),
        },
        "primary_completed_result": "YOLO11l 1280 test mAP50-95 0.5769",
        "primary_eval_setting": "imgsz=1280, augment=False; 1536 and TTA did not improve test mAP50-95",
        "previous_headline": "YOLO11m 960 test mAP50-95 0.5347",
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
    if YOLO11L_RUN.exists():
        write_training_summary(YOLO11L_RUN, "yolo11l_1280_training_summary.csv")
    if YOLO11M_RUN.exists():
        write_training_summary(YOLO11M_RUN, "yolo11m_960_training_summary.csv")
    write_results_markdown(comparison_rows)
    write_outline()
    write_checklist()
    write_metadata()
    print(f"Wrote publication package to {REPORT_DIR}")


if __name__ == "__main__":
    main()
