#!/usr/bin/env python3
"""Render camera-ready Markdown and LaTeX tables directly from experiment CSVs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str, digits: int = 3) -> str:
    return f"{float(row[key]):.{digits}f}"


def model_label(model: str, imgsz: str) -> str:
    labels = {
        "YOLO11s_1280": "YOLO11s",
        "RTDETR_L_640": "RT-DETR-L",
        "YOLO11s_existing_checkpoint_at_640": "YOLO11s checkpoint diagnostic",
        "YOLO11s_640_matched": "YOLO11s",
        "RTDETR_L_640_matched": "RT-DETR-L",
        "RTDETR_L_high_resolution": "RT-DETR-L",
    }
    return f"{labels.get(model, model)} {imgsz}"


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("&", r"\&")
        .replace("%", r"\%")
    )


def render_markdown_metrics(rows: list[dict[str, str]], split: str) -> str:
    selected = [row for row in rows if row["split"] == split]
    lines = [
        "| Model | Precision | Recall | F1 | mAP50 | mAP50-95 | Inference (ms) | Total (ms) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            "| "
            + " | ".join(
                [
                    model_label(row["model"], row["imgsz"]),
                    number(row, "precision"),
                    number(row, "recall"),
                    number(row, "f1"),
                    number(row, "mAP50"),
                    number(row, "mAP50_95"),
                    number(row, "inference_ms", 1),
                    number(row, "total_ms", 1),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_latex_metrics(rows: list[dict[str, str]], split: str) -> str:
    selected = [row for row in rows if row["split"] == split]
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\hline",
        r"Model & P & R & F1 & mAP50 & mAP50--95 & Infer. ms & Total ms \\",
        r"\hline",
    ]
    for row in selected:
        lines.append(
            " & ".join(
                [
                    latex_escape(model_label(row["model"], row["imgsz"])),
                    number(row, "precision"),
                    number(row, "recall"),
                    number(row, "f1"),
                    number(row, "mAP50"),
                    number(row, "mAP50_95"),
                    number(row, "inference_ms", 1),
                    number(row, "total_ms", 1),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def render_per_class_markdown(rows: list[dict[str, str]], split: str) -> str:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["split"] == split:
            grouped[row["class_name"]][row["model"]] = row
    model_order = ["YOLO11s_640_matched", "RTDETR_L_640_matched"]
    lines = [
        "| Class | Model | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for class_name in sorted(grouped):
        for model in model_order:
            row = grouped[class_name].get(model)
            if row is None:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        class_name,
                        model_label(model, row["imgsz"]),
                        number(row, "precision"),
                        number(row, "recall"),
                        number(row, "mAP50"),
                        number(row, "mAP50_95"),
                    ]
                )
                + " |"
            )
    return "\n".join(lines)


def render_per_class_latex(rows: list[dict[str, str]], split: str) -> str:
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        if row["split"] == split:
            grouped[row["class_name"]][row["model"]] = row
    model_order = ["YOLO11s_640_matched", "RTDETR_L_640_matched"]
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\hline",
        r"Class & Model & P & R & mAP50 & mAP50--95 \\",
        r"\hline",
    ]
    for class_name in sorted(grouped):
        for model in model_order:
            row = grouped[class_name].get(model)
            if row is None:
                continue
            lines.append(
                " & ".join(
                    [
                        latex_escape(class_name),
                        latex_escape(model_label(model, row["imgsz"])),
                        number(row, "precision"),
                        number(row, "recall"),
                        number(row, "mAP50"),
                        number(row, "mAP50_95"),
                    ]
                )
                + r" \\"
            )
    lines.extend([r"\hline", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def find_row(rows: list[dict[str, str]], model: str, split: str) -> dict[str, str]:
    for row in rows:
        if row["model"] == model and row["split"] == split:
            return row
    raise KeyError(f"Missing result row: model={model}, split={split}")


def render_result_macros(
    legacy: list[dict[str, str]],
    diagnostic: list[dict[str, str]],
    controlled: list[dict[str, str]],
) -> str:
    rows = {
        "OriginalYolo": find_row(legacy, "YOLO11s_1280", "test"),
        "OriginalRT": find_row(legacy, "RTDETR_L_640", "test"),
        "DiagnosticYolo": find_row(
            diagnostic, "YOLO11s_existing_checkpoint_at_640", "test"
        ),
        "MatchedYolo": find_row(controlled, "YOLO11s_640_matched", "test"),
        "MatchedRT": find_row(controlled, "RTDETR_L_640_matched", "test"),
        "HighRT": find_row(controlled, "RTDETR_L_high_resolution", "test"),
    }
    metrics = {
        "Precision": ("precision", 3),
        "Recall": ("recall", 3),
        "FOne": ("f1", 3),
        "MAPFifty": ("mAP50", 3),
        "MAPStrict": ("mAP50_95", 3),
        "InferenceMs": ("inference_ms", 1),
        "TotalMs": ("total_ms", 1),
    }
    lines = ["% Generated from saved evaluation CSV files. Do not edit manually."]
    for prefix, row in rows.items():
        lines.append(
            rf"\newcommand{{\{prefix}ImageSize}}{{{int(float(row['imgsz']))}}}"
        )
        for suffix, (field, digits) in metrics.items():
            lines.append(
                rf"\newcommand{{\{prefix}{suffix}}}"
                rf"{{{float(row[field]):.{digits}f}}}"
            )
    matched_yolo = rows["MatchedYolo"]
    matched_rt = rows["MatchedRT"]
    lines.append(
        rf"\newcommand{{\MatchedRecallDelta}}"
        rf"{{{float(matched_yolo['recall']) - float(matched_rt['recall']):.3f}}}"
    )
    lines.append(
        rf"\newcommand{{\MatchedMAPStrictDelta}}"
        rf"{{{float(matched_yolo['mAP50_95']) - float(matched_rt['mAP50_95']):.3f}}}"
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-metrics", type=Path, required=True)
    parser.add_argument("--diagnostic-metrics", type=Path, required=True)
    parser.add_argument("--controlled-metrics", type=Path, required=True)
    parser.add_argument("--controlled-per-class", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    legacy = read_csv(args.legacy_metrics)
    diagnostic = read_csv(args.diagnostic_metrics)
    controlled = read_csv(args.controlled_metrics)
    per_class = read_csv(args.controlled_per_class)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    markdown = "\n\n".join(
        [
            "# Generated Resolution-Control Results",
            "## Accepted Practical Configurations - Test",
            render_markdown_metrics(legacy, "test"),
            "## Accepted Practical Configurations - Validation",
            render_markdown_metrics(legacy, "val"),
            "## Evaluation-Only 640 Diagnostic - Test",
            render_markdown_metrics(diagnostic, "test"),
            "## Controlled and High-Resolution Runs - Test",
            render_markdown_metrics(controlled, "test"),
            "## Controlled and High-Resolution Runs - Validation",
            render_markdown_metrics(controlled, "val"),
            "## Matched-Resolution Per-Class Results - Test",
            render_per_class_markdown(per_class, "test"),
        ]
    )
    (args.output_dir / "resolution_control_results.md").write_text(
        markdown + "\n"
    )
    (args.output_dir / "practical_test_table.tex").write_text(
        render_latex_metrics(legacy, "test")
    )
    (args.output_dir / "practical_validation_table.tex").write_text(
        render_latex_metrics(legacy, "val")
    )
    (args.output_dir / "controlled_test_table.tex").write_text(
        render_latex_metrics(controlled, "test")
    )
    (args.output_dir / "controlled_validation_table.tex").write_text(
        render_latex_metrics(controlled, "val")
    )
    (args.output_dir / "controlled_per_class_test_table.tex").write_text(
        render_per_class_latex(per_class, "test")
    )
    (args.output_dir / "result_macros.tex").write_text(
        render_result_macros(legacy, diagnostic, controlled)
    )
    print(f"Generated manuscript tables in {args.output_dir}")


if __name__ == "__main__":
    main()
