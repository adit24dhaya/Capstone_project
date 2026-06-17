#!/usr/bin/env python3
"""Combine independently produced experiment manifests into one audit record."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validate_worker(worker: dict, expected_mode: str) -> None:
    if worker["mode"] != expected_mode:
        raise ValueError(
            f"Expected worker mode {expected_mode}, got {worker['mode']}"
        )
    if worker["seed"] != 42:
        raise ValueError(f"{expected_mode} worker did not use seed 42")
    if worker["evaluation_batch"] != 1:
        raise ValueError(f"{expected_mode} worker did not evaluate at batch 1")
    if worker["workers"] != 0:
        raise ValueError(f"{expected_mode} worker used nonzero dataloader workers")
    if worker["tta"] is not False:
        raise ValueError(f"{expected_mode} worker used test-time augmentation")
    if "Tesla V100-SXM2-32GB" not in worker["gpu"]:
        raise ValueError(f"{expected_mode} worker used an unexpected GPU")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-manifest", type=Path, required=True)
    parser.add_argument("--rt640-manifest", type=Path, required=True)
    parser.add_argument("--rthigh-manifest", type=Path, required=True)
    parser.add_argument("--aggregate-csv", type=Path, required=True)
    parser.add_argument("--per-class-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    yolo = load_json(args.yolo_manifest)
    rt640 = load_json(args.rt640_manifest)
    rthigh = load_json(args.rthigh_manifest)
    validate_worker(rt640, "640")
    validate_worker(rthigh, "highres")

    datasets = [
        yolo["datasets"]["corrected"],
        rt640["dataset_manifest"],
        rthigh["dataset_manifest"],
    ]
    dataset_hashes = [canonical_hash(dataset) for dataset in datasets]
    if len(set(dataset_hashes)) != 1:
        raise ValueError(
            f"Corrected dataset manifests differ across workers: {dataset_hashes}"
        )

    yolo_protocol = yolo["protocol"]
    if (
        yolo_protocol["seed"] != 42
        or yolo_protocol["evaluation_batch"] != 1
        or yolo_protocol["evaluation_workers"] != 0
        or yolo_protocol["test_time_augmentation"] is not False
    ):
        raise ValueError("YOLO manifest does not match the controlled protocol")

    if rt640["imgsz"] != 640:
        raise ValueError(f"RT-DETR matched worker used imgsz={rt640['imgsz']}")
    if rthigh["imgsz"] not in {960, 1280}:
        raise ValueError(
            f"RT-DETR high-resolution worker used imgsz={rthigh['imgsz']}"
        )
    if rthigh["imgsz"] == 960 and not rthigh["fallback_reason"]:
        raise ValueError("960 fallback is missing a confirmed OOM reason")
    if rthigh["imgsz"] == 1280 and rthigh["fallback_reason"]:
        raise ValueError("1280 run unexpectedly records a fallback reason")

    combined = {
        "schema_version": 1,
        "purpose": "ESCS'26 resolution-control camera-ready audit manifest",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "seed": 42,
            "evaluation_batch": 1,
            "evaluation_workers": 0,
            "test_time_augmentation": False,
            "hardware_class": "Tesla-V100-SXM2-32GB",
            "same_validation_and_test_split": True,
            "matched_resolution": 640,
            "high_resolution": rthigh["imgsz"],
            "high_resolution_fallback_reason": rthigh["fallback_reason"],
        },
        "dataset": {
            "manifest_sha256": dataset_hashes[0],
            "manifest": datasets[0],
        },
        "baseline_gate": yolo["baseline_gate"],
        "training_runs": {
            "yolo11s_640": yolo["training_runs"]["yolo11s_640"],
            "rtdetr_l_640": {
                "weights": rt640["weights"],
                "weights_sha256": rt640["weights_sha256"],
                "epochs": rt640["epochs"],
                "training_batch": rt640["training_batch"],
                "imgsz": rt640["imgsz"],
            },
            "rtdetr_l_high_resolution": {
                "weights": rthigh["weights"],
                "weights_sha256": rthigh["weights_sha256"],
                "epochs": rthigh["epochs"],
                "training_batch": rthigh["training_batch"],
                "imgsz": rthigh["imgsz"],
                "fallback_reason": rthigh["fallback_reason"],
            },
        },
        "environments": {
            "yolo": yolo["environment"],
            "rtdetr_l_640": {
                key: rt640[key]
                for key in ("gpu", "hostname", "python", "torch", "ultralytics")
            },
            "rtdetr_l_high_resolution": {
                key: rthigh[key]
                for key in ("gpu", "hostname", "python", "torch", "ultralytics")
            },
        },
        "evaluation_artifacts": {
            "aggregate_csv": str(args.aggregate_csv),
            "aggregate_csv_sha256": sha256_file(args.aggregate_csv),
            "per_class_csv": str(args.per_class_csv),
            "per_class_csv_sha256": sha256_file(args.per_class_csv),
        },
        "claims_guardrail": {
            "jetson_benchmarked": False,
            "tensorrt_benchmarked": False,
            "online_deployment_artifact": (
                "https://adiivd-pcb-defect-detection.hf.space"
            ),
        },
        "source_manifests": {
            "yolo": str(args.yolo_manifest),
            "rtdetr_l_640": str(args.rt640_manifest),
            "rtdetr_l_high_resolution": str(args.rthigh_manifest),
        },
        "software_source": {
            "git_commit": command_output(["git", "rev-parse", "HEAD"]),
            "files": {
                str(path): sha256_file(path)
                for path in (
                    Path("tools/prepare_resolution_control_dataset.py"),
                    Path("tools/run_nautilus_experiments.py"),
                    Path("tools/generate_resolution_control_examples.py"),
                    Path("tools/merge_resolution_control_results.py"),
                    Path("tools/render_resolution_control_tables.py"),
                    Path("reports/publication/nautilus_rtdetr_worker.sh"),
                    Path("reports/publication/nautilus_yolo_finalizer.sh"),
                    Path("reports/publication/escs26_camera_ready.tex"),
                )
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(combined, indent=2) + "\n")
    print(f"Saved combined experiment manifest to {args.output}")


if __name__ == "__main__":
    main()
