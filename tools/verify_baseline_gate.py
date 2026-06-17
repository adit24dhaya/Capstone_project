#!/usr/bin/env python3
"""Stop the resolution-control pipeline if rebuilt baseline metrics drift."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

METRICS = ("precision", "recall", "mAP50", "mAP50_95")


def read_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="") as handle:
        return {
            (row["model"], row["split"]): row
            for row in csv.DictReader(handle)
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=0.001)
    parser.add_argument(
        "--model-map",
        type=json.loads,
        default={},
        help='JSON mapping from actual model name to expected model name',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    expected = read_rows(args.expected)
    actual = read_rows(args.actual)
    comparisons = []
    failures = []

    for (actual_model, split), actual_row in sorted(actual.items()):
        expected_model = args.model_map.get(actual_model, actual_model)
        expected_row = expected.get((expected_model, split))
        if expected_row is None:
            failures.append(
                f"Missing expected row for model={expected_model}, split={split}"
            )
            continue
        for metric in METRICS:
            expected_value = float(expected_row[metric])
            actual_value = float(actual_row[metric])
            difference = abs(actual_value - expected_value)
            passed = difference <= args.tolerance
            comparisons.append(
                {
                    "actual_model": actual_model,
                    "expected_model": expected_model,
                    "split": split,
                    "metric": metric,
                    "expected": expected_value,
                    "actual": actual_value,
                    "absolute_difference": difference,
                    "tolerance": args.tolerance,
                    "passed": passed,
                }
            )
            if not passed:
                failures.append(
                    f"{actual_model}/{split}/{metric}: "
                    f"expected {expected_value:.8f}, got {actual_value:.8f}, "
                    f"difference {difference:.8f}"
                )

    report = {
        "passed": not failures,
        "expected_csv": str(args.expected),
        "actual_csv": str(args.actual),
        "tolerance": args.tolerance,
        "comparisons": comparisons,
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit("Baseline reproduction gate failed")


if __name__ == "__main__":
    main()
