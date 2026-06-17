#!/usr/bin/env python3
"""Create an auditable manifest for the ESCS'26 resolution-control runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--high-resolution", type=int, required=True)
    parser.add_argument(
        "--fallback-reason",
        default="",
        help="Empty unless 1280 failed with a confirmed CUDA OOM",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    workspace = args.workspace.resolve()
    experiment_root = workspace / "resolution-control"
    repo = workspace / "repo"
    checkpoints = workspace / "checkpoints"
    outputs = experiment_root / "outputs"

    training_runs = {
        "yolo11s_640": outputs
        / "runs/detector_train/yolo11s_640_matched",
        "rtdetr_l_640": outputs
        / "runs/rtdetr/rtdetr_l_640_matched",
        "rtdetr_l_high_resolution": outputs
        / "runs/rtdetr/rtdetr_l_high_resolution",
    }
    manifest = {
        "schema_version": 1,
        "purpose": "ESCS'26 matched-resolution and high-resolution camera-ready analysis",
        "protocol": {
            "seed": 42,
            "evaluation_batch": 1,
            "evaluation_workers": 0,
            "test_time_augmentation": False,
            "required_gpu": "Tesla-V100-SXM2-32GB",
            "yolo11s_640": {"epochs": 50, "training_batch": 12, "imgsz": 640},
            "rtdetr_l_640": {"epochs": 10, "training_batch": 4, "imgsz": 640},
            "rtdetr_l_high_resolution": {
                "epochs": 10,
                "training_batch": 1,
                "imgsz": args.high_resolution,
                "preferred_imgsz": 1280,
                "fallback_imgsz": 960,
                "fallback_reason": args.fallback_reason or None,
            },
        },
        "datasets": {
            "legacy": load_json(
                experiment_root / "data/YOLO_PCB_legacy/dataset_manifest.json"
            ),
            "corrected": load_json(
                experiment_root / "data/YOLO_PCB_corrected/dataset_manifest.json"
            ),
        },
        "baseline_gate": load_json(
            experiment_root / "legacy-eval/baseline_gate.json"
        ),
        "checkpoints": {
            "accepted_yolo11s_1280": {
                "path": str(
                    checkpoints / "yolo11s_1280_kaggle_fair_best.pt"
                ),
                "sha256": file_sha256(
                    checkpoints / "yolo11s_1280_kaggle_fair_best.pt"
                ),
            },
            "accepted_rtdetr_l_640": {
                "path": str(checkpoints / "rtdetr_l_kaggle_fair_best.pt"),
                "sha256": file_sha256(
                    checkpoints / "rtdetr_l_kaggle_fair_best.pt"
                ),
            },
        },
        "training_runs": {},
        "evaluation_artifacts": {
            "legacy_metrics": str(
                experiment_root
                / "legacy-eval/runs/paper_unified_eval/paper_unified_eval_metrics.csv"
            ),
            "diagnostic_metrics": str(
                experiment_root
                / "diagnostic-eval/runs/paper_unified_eval/paper_unified_eval_metrics.csv"
            ),
            "controlled_metrics": str(
                experiment_root
                / "controlled-eval/runs/paper_unified_eval/paper_unified_eval_metrics.csv"
            ),
            "controlled_per_class": str(
                experiment_root
                / "controlled-eval/runs/paper_unified_eval/paper_unified_eval_per_class.csv"
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "git_commit": command_output(
                ["git", "-C", str(repo), "rev-parse", "HEAD"]
            ),
            "gpu": command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ]
            ),
            "torch": package_version("torch"),
            "ultralytics": package_version("ultralytics"),
            "numpy": package_version("numpy"),
            "opencv_python_headless": package_version(
                "opencv-python-headless"
            ),
            "albumentations": package_version("albumentations"),
        },
        "claims_guardrail": {
            "jetson_benchmarked": False,
            "tensorrt_benchmarked": False,
            "online_deployment_artifact": "https://adiivd-pcb-defect-detection.hf.space",
        },
    }
    for name, run_dir in training_runs.items():
        weights = run_dir / "weights/best.pt"
        manifest["training_runs"][name] = {
            "run_dir": str(run_dir),
            "best_weights": str(weights),
            "best_weights_sha256": file_sha256(weights),
            "args": (
                (run_dir / "args.yaml").read_text()
                if (run_dir / "args.yaml").exists()
                else None
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
