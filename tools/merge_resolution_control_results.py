#!/usr/bin/env python3
"""Merge independently evaluated resolution-control runs with protocol checks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED_MODELS = {
    "YOLO11s_640_matched": 640,
    "RTDETR_L_640_matched": 640,
    "RTDETR_L_high_resolution": None,
}
EXPECTED_SPLITS = {"val", "test"}
MODEL_ALIASES = {
    "rtdetr_l_640_matched": "RTDETR_L_640_matched",
    "rtdetr_l_high_resolution": "RTDETR_L_high_resolution",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return reader.fieldnames, list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def normalize_models(rows: list[dict[str, str]]) -> None:
    for row in rows:
        row["model"] = MODEL_ALIASES.get(row["model"], row["model"])


def validate_aggregate(rows: list[dict[str, str]]) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        model = row["model"]
        split = row["split"]
        if model not in EXPECTED_MODELS:
            raise ValueError(f"Unexpected model in aggregate CSV: {model}")
        if split not in EXPECTED_SPLITS:
            raise ValueError(f"Unexpected split for {model}: {split}")
        key = (model, split)
        if key in seen:
            raise ValueError(f"Duplicate aggregate row: {model}/{split}")
        seen.add(key)
        if int(row["batch"]) != 1:
            raise ValueError(f"{model}/{split} was not evaluated with batch=1")
        if row["augment"].strip().lower() not in {"false", "0"}:
            raise ValueError(f"{model}/{split} used test-time augmentation")
        expected_imgsz = EXPECTED_MODELS[model]
        if expected_imgsz is not None and int(float(row["imgsz"])) != expected_imgsz:
            raise ValueError(
                f"{model}/{split} used imgsz={row['imgsz']}, expected {expected_imgsz}"
            )
        for metric in (
            "precision",
            "recall",
            "f1",
            "mAP50",
            "mAP50_95",
            "preprocess_ms",
            "inference_ms",
            "postprocess_ms",
            "total_ms",
        ):
            float(row[metric])

    expected = {
        (model, split)
        for model in EXPECTED_MODELS
        for split in EXPECTED_SPLITS
    }
    missing = expected - seen
    if missing:
        raise ValueError(f"Missing aggregate rows: {sorted(missing)}")

    high_res = {
        int(float(row["imgsz"]))
        for row in rows
        if row["model"] == "RTDETR_L_high_resolution"
    }
    if len(high_res) != 1 or next(iter(high_res)) not in {960, 1280}:
        raise ValueError(f"Invalid high-resolution RT-DETR image size: {high_res}")


def validate_per_class(rows: list[dict[str, str]]) -> None:
    seen: set[tuple[str, str, str]] = set()
    class_sets: dict[tuple[str, str], set[str]] = {}
    for row in rows:
        model = row["model"]
        split = row["split"]
        if model not in EXPECTED_MODELS or split not in EXPECTED_SPLITS:
            raise ValueError(f"Unexpected per-class row: {model}/{split}")
        key = (model, split, row["class_name"])
        if key in seen:
            raise ValueError(f"Duplicate per-class row: {key}")
        seen.add(key)
        class_sets.setdefault((model, split), set()).add(row["class_name"])
        for metric in ("precision", "recall", "mAP50", "mAP50_95"):
            float(row[metric])

    reference: set[str] | None = None
    for model in EXPECTED_MODELS:
        for split in EXPECTED_SPLITS:
            classes = class_sets.get((model, split), set())
            if len(classes) != 6:
                raise ValueError(
                    f"{model}/{split} has {len(classes)} classes, expected 6"
                )
            if reference is None:
                reference = classes
            elif classes != reference:
                raise ValueError(
                    f"Per-class taxonomy mismatch for {model}/{split}: {classes}"
                )


def merge(paths: list[Path]) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames: list[str] | None = None
    rows: list[dict[str, str]] = []
    for path in paths:
        current_fields, current_rows = read_csv(path)
        if fieldnames is None:
            fieldnames = current_fields
        elif current_fields != fieldnames:
            raise ValueError(f"CSV schema mismatch in {path}")
        rows.extend(current_rows)
    if fieldnames is None:
        raise ValueError("No CSV inputs supplied")
    return fieldnames, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, action="append", required=True)
    parser.add_argument("--per-class", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    aggregate_fields, aggregate_rows = merge(args.aggregate)
    per_class_fields, per_class_rows = merge(args.per_class)
    normalize_models(aggregate_rows)
    normalize_models(per_class_rows)
    validate_aggregate(aggregate_rows)
    validate_per_class(per_class_rows)

    aggregate_rows.sort(key=lambda row: (row["model"], row["split"]))
    per_class_rows.sort(
        key=lambda row: (row["model"], row["split"], int(row["class_id"]))
    )
    write_csv(
        args.output_dir / "paper_unified_eval_metrics.csv",
        aggregate_fields,
        aggregate_rows,
    )
    write_csv(
        args.output_dir / "paper_unified_eval_per_class.csv",
        per_class_fields,
        per_class_rows,
    )
    print(
        f"Merged {len(aggregate_rows)} aggregate and "
        f"{len(per_class_rows)} per-class rows into {args.output_dir}"
    )


if __name__ == "__main__":
    main()
