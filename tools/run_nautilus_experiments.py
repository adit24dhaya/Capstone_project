#!/usr/bin/env python3
"""Portable Nautilus experiment runner for PCB defect experiments.

This script avoids Kaggle-only paths and writes all generated artifacts under
an output directory, defaulting to ``~/outputs/nautilus``.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


CLASS_NAMES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper",
]

CLASS_ALIASES = {
    "missing_hole": "Missing_hole",
    "missing hole": "Missing_hole",
    "mouse_bite": "Mouse_bite",
    "mousebite": "Mouse_bite",
    "mouse bite": "Mouse_bite",
    "open_circuit": "Open_circuit",
    "open circuit": "Open_circuit",
    "open": "Open_circuit",
    "short": "Short",
    "spur": "Spur",
    "spurious_copper": "Spurious_copper",
    "spurious copper": "Spurious_copper",
    "copper": "Spurious_copper",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class DatasetRecord:
    xml_path: str
    image_path: str
    class_name: str
    width: int
    height: int
    boxes: int


def expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def normalize_class(name: str) -> str:
    key = name.strip().replace("-", "_").replace(" ", "_").lower()
    key_space = key.replace("_", " ")
    if key in CLASS_ALIASES:
        return CLASS_ALIASES[key]
    if key_space in CLASS_ALIASES:
        return CLASS_ALIASES[key_space]
    raise ValueError(f"Unknown class label: {name!r}")


def find_current_pcb_root(data_root: Path) -> Path:
    candidates = [
        data_root / "current_pcb" / "PCB-DATASET-master",
        data_root / "PCB-DATASET-master",
        data_root,
    ]
    for candidate in candidates:
        if (candidate / "Annotations").exists() and (candidate / "images").exists():
            return candidate
    matches = list(data_root.rglob("PCB-DATASET-master"))
    for match in matches:
        if (match / "Annotations").exists() and (match / "images").exists():
            return match
    raise FileNotFoundError(
        "Could not locate PCB-DATASET-master with Annotations/ and images/. "
        f"Searched under {data_root}"
    )


def locate_image(dataset_root: Path, class_name: str, filename: str) -> Path:
    search_dirs = [
        dataset_root / "images" / class_name,
        dataset_root / "rotation" / f"{class_name}_rotation",
    ]
    for directory in search_dirs:
        candidate = directory / filename
        if candidate.exists():
            return candidate

    stem = Path(filename).stem
    for directory in search_dirs:
        for suffix in IMAGE_SUFFIXES:
            candidate = directory / f"{stem}{suffix}"
            if candidate.exists():
                return candidate

    matches = list((dataset_root / "images").rglob(filename))
    if matches:
        return matches[0]
    matches = list((dataset_root / "rotation").rglob(filename))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find image for {filename} ({class_name})")


def parse_xml_record(xml_path: Path, dataset_root: Path) -> tuple[DatasetRecord, list[tuple[int, float, float, float, float]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    if not filename:
        raise ValueError(f"Missing filename in {xml_path}")

    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing size in {xml_path}")
    width = int(float(size.findtext("width", "0")))
    height = int(float(size.findtext("height", "0")))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in {xml_path}: {width}x{height}")

    folder = root.findtext("folder") or xml_path.parent.name
    folder_class = normalize_class(folder)
    image_path = locate_image(dataset_root, folder_class, filename)

    yolo_boxes: list[tuple[int, float, float, float, float]] = []
    for obj in root.findall("object"):
        raw_name = obj.findtext("name") or folder_class
        class_name = normalize_class(raw_name)
        class_id = CLASS_NAMES.index(class_name)
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = max(0.0, float(box.findtext("xmin", "0")))
        ymin = max(0.0, float(box.findtext("ymin", "0")))
        xmax = min(float(width), float(box.findtext("xmax", "0")))
        ymax = min(float(height), float(box.findtext("ymax", "0")))
        if xmax <= xmin or ymax <= ymin:
            continue
        x_center = ((xmin + xmax) / 2.0) / width
        y_center = ((ymin + ymax) / 2.0) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        yolo_boxes.append((class_id, x_center, y_center, box_width, box_height))

    record = DatasetRecord(
        xml_path=str(xml_path),
        image_path=str(image_path),
        class_name=folder_class,
        width=width,
        height=height,
        boxes=len(yolo_boxes),
    )
    return record, yolo_boxes


def iter_xml_files(dataset_root: Path) -> list[Path]:
    return sorted((dataset_root / "Annotations").rglob("*.xml"))


def split_records(records: list[tuple[DatasetRecord, list[tuple[int, float, float, float, float]]]], seed: int) -> dict[str, list[tuple[DatasetRecord, list[tuple[int, float, float, float, float]]]]]:
    rng = random.Random(seed)
    by_class: dict[str, list[tuple[DatasetRecord, list[tuple[int, float, float, float, float]]]]] = {
        name: [] for name in CLASS_NAMES
    }
    for item in records:
        by_class[item[0].class_name].append(item)

    splits = {"train": [], "val": [], "test": []}
    for class_name, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        splits["train"].extend(items[:n_train])
        splits["val"].extend(items[n_train : n_train + n_val])
        splits["test"].extend(items[n_train + n_val :])

    for split_items in splits.values():
        rng.shuffle(split_items)
    return splits


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def copy_or_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src)
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unsupported file mode: {mode}")


def convert_current_pcb_to_yolo(data_root: Path, output_dir: Path, seed: int, file_mode: str) -> dict:
    dataset_root = find_current_pcb_root(data_root)
    yolo_root = output_dir / "datasets" / "current_pcb_yolo"
    manifest_path = output_dir / "current_pcb_conversion_manifest.json"
    summary_path = output_dir / "current_pcb_conversion_summary.json"

    parsed: list[tuple[DatasetRecord, list[tuple[int, float, float, float, float]]]] = []
    skipped: list[dict[str, str]] = []
    for xml_path in iter_xml_files(dataset_root):
        try:
            parsed.append(parse_xml_record(xml_path, dataset_root))
        except Exception as exc:  # noqa: BLE001 - summary should capture all conversion issues.
            skipped.append({"xml_path": str(xml_path), "error": str(exc)})

    splits = split_records(parsed, seed)
    manifest: list[dict] = []
    for split_name, items in splits.items():
        for record, boxes in items:
            src = Path(record.image_path)
            dst_image = yolo_root / "images" / split_name / src.name
            dst_label = yolo_root / "labels" / split_name / f"{src.stem}.txt"
            copy_or_link(src, dst_image, file_mode)
            label_lines = [
                f"{class_id} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}"
                for class_id, xc, yc, bw, bh in boxes
            ]
            write_text(dst_label, "\n".join(label_lines) + ("\n" if label_lines else ""))
            manifest.append({**asdict(record), "split": split_name, "label_path": str(dst_label), "yolo_image_path": str(dst_image)})

    data_yaml = yolo_root / "data.yaml"
    yaml_text = "\n".join(
        [
            f"path: {yolo_root}",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "names:",
            *[f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)],
            "",
        ]
    )
    write_text(data_yaml, yaml_text)

    class_counts: dict[str, dict[str, int]] = {split: {name: 0 for name in CLASS_NAMES} for split in splits}
    image_counts = {split: len(items) for split, items in splits.items()}
    box_counts = {split: 0 for split in splits}
    for row in manifest:
        class_counts[row["split"]][row["class_name"]] += 1
        box_counts[row["split"]] += int(row["boxes"])

    summary = {
        "dataset_root": str(dataset_root),
        "yolo_root": str(yolo_root),
        "data_yaml": str(data_yaml),
        "classes": CLASS_NAMES,
        "image_counts": image_counts,
        "box_counts": box_counts,
        "class_image_counts": class_counts,
        "records": len(parsed),
        "skipped": skipped,
        "file_mode": file_mode,
    }
    write_text(manifest_path, json.dumps(manifest, indent=2))
    write_text(summary_path, json.dumps(summary, indent=2))
    return summary


def environment_summary() -> dict:
    result: dict = {"python": sys.version}
    try:
        import torch

        result["torch"] = torch.__version__
        result["cuda_available"] = torch.cuda.is_available()
        result["cuda_build"] = torch.version.cuda
        result["gpu_count"] = torch.cuda.device_count()
        result["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception as exc:  # noqa: BLE001
        result["torch_error"] = str(exc)

    for package in ["ultralytics", "albumentations", "cv2", "pandas", "numpy"]:
        try:
            module = __import__(package)
            result[package] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # noqa: BLE001
            result[f"{package}_error"] = str(exc)
    return result


def run_smoke(args: argparse.Namespace) -> None:
    output_dir = expand_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = environment_summary()
    conversion = convert_current_pcb_to_yolo(
        data_root=expand_path(args.data_root),
        output_dir=output_dir,
        seed=args.seed,
        file_mode=args.file_mode,
    )
    smoke = {"environment": env, "conversion": conversion, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    write_text(output_dir / "smoke_check.json", json.dumps(smoke, indent=2))
    print(json.dumps(smoke, indent=2))


def run_yolo_smoke(args: argparse.Namespace) -> None:
    output_dir = expand_path(args.output_dir)
    summary_path = output_dir / "current_pcb_conversion_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
    else:
        summary = convert_current_pcb_to_yolo(
            data_root=expand_path(args.data_root),
            output_dir=output_dir,
            seed=args.seed,
            file_mode=args.file_mode,
        )

    from ultralytics import YOLO

    model = YOLO(args.yolo_model)
    train_dir = output_dir / "runs" / "yolo_smoke"
    results = model.train(
        data=summary["data_yaml"],
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(train_dir),
        name="current_pcb",
        exist_ok=True,
        patience=3,
    )
    result_summary = {
        "data_yaml": summary["data_yaml"],
        "run_dir": str(train_dir / "current_pcb"),
        "model": args.yolo_model,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "results_type": str(type(results)),
    }
    write_text(output_dir / "yolo_smoke_summary.json", json.dumps(result_summary, indent=2))
    print(json.dumps(result_summary, indent=2))


def placeholder(name: str) -> None:
    raise SystemExit(
        f"Experiment {name!r} is not implemented yet. "
        "Run --experiment smoke or --experiment yolo_smoke first."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=["smoke", "yolo_smoke", "faster_rcnn", "cross_dataset", "segmentation_pilot"],
        required=True,
    )
    parser.add_argument("--data-root", default="~/data", help="Root containing current_pcb/, deeppcb/, etc.")
    parser.add_argument("--output-dir", default="~/outputs/nautilus", help="Directory for all generated outputs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--file-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--yolo-model", default="yolo11n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.experiment == "smoke":
        run_smoke(args)
    elif args.experiment == "yolo_smoke":
        run_yolo_smoke(args)
    else:
        placeholder(args.experiment)


if __name__ == "__main__":
    main()
