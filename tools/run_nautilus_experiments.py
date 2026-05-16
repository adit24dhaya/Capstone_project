#!/usr/bin/env python3
"""Portable Nautilus experiment runner for PCB defect experiments.

This script avoids Kaggle-only paths and writes all generated artifacts under
an output directory, defaulting to ``~/outputs/nautilus``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
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

DSPCBSD_CLASS_ALIASES = {
    "SH": "Short",
    "short": "Short",
    "short_circuit": "Short",
    "SP": "Spur",
    "spur": "Spur",
    "SC": "Spurious_copper",
    "spurious_copper": "Spurious_copper",
    "spurious copper": "Spurious_copper",
    "OP": "Open_circuit",
    "open_circuit": "Open_circuit",
    "open circuit": "Open_circuit",
    "MB": "Mouse_bite",
    "mouse_bite": "Mouse_bite",
    "mouse bite": "Mouse_bite",
}

SIZE_BINS = [
    ("tiny", 0.001),
    ("small", 0.005),
    ("medium", 0.02),
    ("large", math.inf),
]


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


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_yaml(path: Path) -> dict:
    import yaml

    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in YAML file: {path}")
    return data


def resolve_dataset_path(base: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def split_source_to_images(source: Path) -> list[Path]:
    if source.is_file() and source.suffix.lower() == ".txt":
        paths = []
        base = source.parent
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            paths.append(resolve_dataset_path(base, line.strip()))
        return paths
    if source.is_dir():
        return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        return [source]
    raise FileNotFoundError(f"Could not resolve image source: {source}")


def infer_label_path(image_path: Path, data_root: Path, split: str) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = parts.index("images")
        label_parts = parts[:idx] + ["labels"] + parts[idx + 1 :]
        return Path(*label_parts).with_suffix(".txt")
    return data_root / "labels" / split / f"{image_path.stem}.txt"


def load_yolo_items_from_data_yaml(data_yaml: Path, split: str) -> list[dict]:
    from PIL import Image

    cfg = load_yaml(data_yaml)
    root = resolve_dataset_path(data_yaml.parent, cfg.get("path", "."))
    source_value = cfg.get(split) or cfg.get("val")
    if not source_value:
        raise ValueError(f"No {split!r} or 'val' split found in {data_yaml}")
    source = resolve_dataset_path(root, source_value)
    image_paths = split_source_to_images(source)
    items: list[dict] = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            width, height = img.size
        label_path = infer_label_path(image_path, root, split)
        boxes: list[list[float]] = []
        labels: list[int] = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                class_id, box = yolo_label_to_xyxy(line, width, height)
                if box[2] > box[0] and box[3] > box[1]:
                    boxes.append(box)
                    labels.append(class_id)
        items.append(
            {
                "image_path": image_path,
                "label_path": label_path,
                "width": width,
                "height": height,
                "boxes": boxes,
                "labels": labels,
            }
        )
    return items


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


def normalize_external_class(name: str, class_map: dict[str, str]) -> str | None:
    raw = str(name).strip()
    candidates = [raw, raw.replace("-", "_"), raw.replace("_", " "), raw.lower(), raw.upper()]
    for candidate in candidates:
        if candidate in class_map:
            return class_map[candidate]
        if candidate in DSPCBSD_CLASS_ALIASES:
            return DSPCBSD_CLASS_ALIASES[candidate]
    key = raw.replace("-", "_").replace(" ", "_").lower()
    if key in CLASS_ALIASES:
        return CLASS_ALIASES[key]
    if raw in CLASS_NAMES:
        return raw
    return None


def locate_coco_image(coco_root: Path, file_name: str) -> Path:
    direct = coco_root / file_name
    if direct.exists():
        return direct
    candidates = [
        coco_root / "images" / file_name,
        coco_root / "train2017" / file_name,
        coco_root / "val2017" / file_name,
        coco_root / "test2017" / file_name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(coco_root.rglob(Path(file_name).name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not locate COCO image: {file_name}")


def infer_coco_split(json_path: Path) -> str:
    name = json_path.stem.lower()
    if "test" in name:
        return "test"
    if "val" in name:
        return "val"
    return "train"


def find_coco_annotation_files(coco_root: Path) -> list[Path]:
    candidates = []
    for directory in [coco_root, coco_root / "annotations"]:
        if directory.exists():
            candidates.extend(directory.glob("*.json"))
    return sorted(path for path in candidates if path.is_file())


def convert_coco_to_yolo_dataset(
    coco_root: Path,
    output_dir: Path,
    dataset_name: str,
    file_mode: str,
    class_map: dict[str, str] | None = None,
) -> dict:
    """Convert COCO-style detection annotations into the six-class YOLO taxonomy.

    This is intended for DsPCBSD+ and other publication datasets. Unmapped
    categories are skipped by default so the benchmark remains comparable with
    the original six-class capstone taxonomy.
    """

    class_map = class_map or {}
    annotation_files = find_coco_annotation_files(coco_root)
    if not annotation_files:
        raise FileNotFoundError(f"No COCO JSON annotation files found under {coco_root}")

    yolo_root = output_dir / "datasets" / f"{dataset_name}_yolo"
    summary_rows: list[dict] = []
    skipped_categories: dict[str, int] = defaultdict(int)
    image_counts: dict[str, int] = defaultdict(int)
    box_counts: dict[str, int] = defaultdict(int)

    for ann_path in annotation_files:
        split = infer_coco_split(ann_path)
        data = json.loads(ann_path.read_text())
        categories = {cat["id"]: cat.get("name", str(cat["id"])) for cat in data.get("categories", [])}
        images = {image["id"]: image for image in data.get("images", [])}
        grouped_annotations: dict[int, list[dict]] = defaultdict(list)
        for ann in data.get("annotations", []):
            grouped_annotations[ann["image_id"]].append(ann)

        for image_id, image_info in images.items():
            file_name = image_info.get("file_name")
            if not file_name:
                continue
            src_image = locate_coco_image(coco_root, file_name)
            width = int(image_info.get("width") or 0)
            height = int(image_info.get("height") or 0)
            if width <= 0 or height <= 0:
                from PIL import Image

                with Image.open(src_image) as img:
                    width, height = img.size

            dst_image = yolo_root / "images" / split / src_image.name
            dst_label = yolo_root / "labels" / split / f"{src_image.stem}.txt"
            copy_or_link(src_image, dst_image, file_mode)

            label_lines: list[str] = []
            for ann in grouped_annotations.get(image_id, []):
                raw_cat = categories.get(ann.get("category_id"), str(ann.get("category_id")))
                mapped = normalize_external_class(raw_cat, class_map)
                if mapped is None:
                    skipped_categories[raw_cat] += 1
                    continue
                x, y, bw, bh = [float(v) for v in ann.get("bbox", [0, 0, 0, 0])]
                if bw <= 0 or bh <= 0:
                    continue
                xc = (x + bw / 2.0) / width
                yc = (y + bh / 2.0) / height
                nw = bw / width
                nh = bh / height
                class_id = CLASS_NAMES.index(mapped)
                label_lines.append(f"{class_id} {xc:.8f} {yc:.8f} {nw:.8f} {nh:.8f}")
            write_text(dst_label, "\n".join(label_lines) + ("\n" if label_lines else ""))
            image_counts[split] += 1
            box_counts[split] += len(label_lines)
            summary_rows.append(
                {
                    "split": split,
                    "image_path": str(dst_image),
                    "label_path": str(dst_label),
                    "width": width,
                    "height": height,
                    "boxes": len(label_lines),
                    "source_json": str(ann_path),
                }
            )

    data_yaml = yolo_root / "data.yaml"
    split_lines = []
    for split in ["train", "val", "test"]:
        if (yolo_root / "images" / split).exists():
            split_lines.append(f"{split}: images/{split}")
    yaml_text = "\n".join(
        [
            f"path: {yolo_root}",
            *split_lines,
            "names:",
            *[f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES)],
            "",
        ]
    )
    write_text(data_yaml, yaml_text)
    write_csv(yolo_root / "conversion_manifest.csv", summary_rows)

    summary = {
        "dataset_name": dataset_name,
        "coco_root": str(coco_root),
        "yolo_root": str(yolo_root),
        "data_yaml": str(data_yaml),
        "classes": CLASS_NAMES,
        "image_counts": dict(image_counts),
        "box_counts": dict(box_counts),
        "skipped_categories": dict(skipped_categories),
        "records": len(summary_rows),
        "file_mode": file_mode,
    }
    write_text(output_dir / f"{dataset_name}_coco_conversion_summary.json", json.dumps(summary, indent=2))
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


def load_conversion_summary(args: argparse.Namespace) -> dict:
    output_dir = expand_path(args.output_dir)
    summary_path = output_dir / "current_pcb_conversion_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    return convert_current_pcb_to_yolo(
        data_root=expand_path(args.data_root),
        output_dir=output_dir,
        seed=args.seed,
        file_mode=args.file_mode,
    )


def yolo_label_to_xyxy(line: str, width: int, height: int) -> tuple[int, list[float]]:
    class_id_s, xc_s, yc_s, bw_s, bh_s = line.strip().split()
    class_id = int(class_id_s)
    xc = float(xc_s) * width
    yc = float(yc_s) * height
    bw = float(bw_s) * width
    bh = float(bh_s) * height
    x1 = max(0.0, xc - bw / 2.0)
    y1 = max(0.0, yc - bh / 2.0)
    x2 = min(float(width), xc + bw / 2.0)
    y2 = min(float(height), yc + bh / 2.0)
    return class_id, [x1, y1, x2, y2]


def load_yolo_dataset_items(yolo_root: Path, split: str) -> list[dict]:
    from PIL import Image

    image_dir = yolo_root / "images" / split
    label_dir = yolo_root / "labels" / split
    items: list[dict] = []
    for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        with Image.open(image_path) as img:
            width, height = img.size
        label_path = label_dir / f"{image_path.stem}.txt"
        boxes: list[list[float]] = []
        labels: list[int] = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                class_id, box = yolo_label_to_xyxy(line, width, height)
                if box[2] > box[0] and box[3] > box[1]:
                    boxes.append(box)
                    labels.append(class_id)
        items.append(
            {
                "image_path": image_path,
                "label_path": label_path,
                "width": width,
                "height": height,
                "boxes": boxes,
                "labels": labels,
            }
        )
    return items


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def detection_metrics_iou50(
    ground_truth: list[dict],
    predictions: list[dict],
    score_threshold: float,
    iou_threshold: float = 0.5,
) -> dict:
    rows = []
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for class_id, class_name in enumerate(CLASS_NAMES):
        tp = fp = fn = 0
        for gt_item, pred_item in zip(ground_truth, predictions):
            gt_boxes = [box for box, label in zip(gt_item["boxes"], gt_item["labels"]) if label == class_id]
            pred_pairs = [
                (box, score)
                for box, label, score in zip(pred_item["boxes"], pred_item["labels"], pred_item["scores"])
                if label == class_id and score >= score_threshold
            ]
            pred_pairs.sort(key=lambda pair: pair[1], reverse=True)
            matched_gt: set[int] = set()
            for pred_box, _score in pred_pairs:
                best_iou = 0.0
                best_idx = -1
                for idx, gt_box in enumerate(gt_boxes):
                    if idx in matched_gt:
                        continue
                    iou = box_iou(pred_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = idx
                if best_iou >= iou_threshold and best_idx >= 0:
                    tp += 1
                    matched_gt.add(best_idx)
                else:
                    fp += 1
            fn += max(0, len(gt_boxes) - len(matched_gt))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
            }
        )
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

    precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0
    recall = totals["tp"] / (totals["tp"] + totals["fn"]) if totals["tp"] + totals["fn"] else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "fp_per_image": totals["fp"] / len(ground_truth) if ground_truth else 0.0,
        "per_class": rows,
    }


def size_bin_for_box(box: list[float], width: int, height: int) -> str:
    area_ratio = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]) / max(1.0, float(width * height))
    for name, upper in SIZE_BINS:
        if area_ratio < upper:
            return name
    return "large"


def match_detection_events(
    gt_item: dict,
    pred_item: dict,
    score_threshold: float,
    iou_threshold: float = 0.5,
) -> list[dict]:
    events: list[dict] = []
    gt_boxes = gt_item["boxes"]
    gt_labels = gt_item["labels"]
    pred_rows = [
        (box, int(label), float(score))
        for box, label, score in zip(pred_item["boxes"], pred_item["labels"], pred_item["scores"])
        if float(score) >= score_threshold
    ]
    pred_rows.sort(key=lambda row: row[2], reverse=True)
    matched_gt: set[int] = set()

    for pred_box, pred_label, score in pred_rows:
        best_iou = 0.0
        best_idx = -1
        for idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
            if idx in matched_gt or int(gt_label) != pred_label:
                continue
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_threshold and best_idx >= 0:
            matched_gt.add(best_idx)
            events.append(
                {
                    "event": "TP",
                    "class_id": pred_label,
                    "class_name": CLASS_NAMES[pred_label],
                    "score": score,
                    "iou": best_iou,
                    "size_bin": size_bin_for_box(gt_boxes[best_idx], gt_item["width"], gt_item["height"]),
                }
            )
        else:
            events.append(
                {
                    "event": "FP",
                    "class_id": pred_label,
                    "class_name": CLASS_NAMES[pred_label] if 0 <= pred_label < len(CLASS_NAMES) else str(pred_label),
                    "score": score,
                    "iou": best_iou,
                    "size_bin": size_bin_for_box(pred_box, gt_item["width"], gt_item["height"]),
                }
            )

    for idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
        if idx in matched_gt:
            continue
        events.append(
            {
                "event": "FN",
                "class_id": int(gt_label),
                "class_name": CLASS_NAMES[int(gt_label)],
                "score": None,
                "iou": 0.0,
                "size_bin": size_bin_for_box(gt_box, gt_item["width"], gt_item["height"]),
            }
        )
    return events


def dataset_events(ground_truth: list[dict], predictions: list[dict], score_threshold: float) -> list[dict]:
    rows: list[dict] = []
    for image_idx, (gt_item, pred_item) in enumerate(zip(ground_truth, predictions)):
        for event in match_detection_events(gt_item, pred_item, score_threshold):
            rows.append({"image_idx": image_idx, **event})
    return rows


def summarize_events_by_size(events: list[dict], images: int) -> list[dict]:
    rows = []
    for size_name, _upper in SIZE_BINS:
        subset = [row for row in events if row["size_bin"] == size_name]
        tp = sum(row["event"] == "TP" for row in subset)
        fp = sum(row["event"] == "FP" for row in subset)
        fn = sum(row["event"] == "FN" for row in subset)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append(
            {
                "size_bin": size_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "fp_per_image": fp / max(1, images),
            }
        )
    return rows


def summarize_cost(events: list[dict], images: int, missed_defect_penalty: float) -> dict:
    fp = sum(row["event"] == "FP" for row in events)
    fn = sum(row["event"] == "FN" for row in events)
    fp_per_image = fp / max(1, images)
    fn_per_image = fn / max(1, images)
    return {
        "false_positives": fp,
        "missed_defects": fn,
        "images": images,
        "fp_per_image": fp_per_image,
        "fn_per_image": fn_per_image,
        "missed_defect_penalty": missed_defect_penalty,
        "inspection_cost": fp_per_image + missed_defect_penalty * fn_per_image,
    }


def summarize_calibration(events: list[dict], bins: int) -> tuple[dict, list[dict]]:
    pred_events = [row for row in events if row["event"] in {"TP", "FP"} and row["score"] is not None]
    if not pred_events:
        return {"ece": 0.0, "samples": 0, "bins": bins}, []
    rows = []
    ece = 0.0
    for idx in range(bins):
        low = idx / bins
        high = (idx + 1) / bins
        if idx == bins - 1:
            subset = [row for row in pred_events if low <= float(row["score"]) <= high]
        else:
            subset = [row for row in pred_events if low <= float(row["score"]) < high]
        count = len(subset)
        accuracy = sum(row["event"] == "TP" for row in subset) / count if count else 0.0
        confidence = sum(float(row["score"]) for row in subset) / count if count else 0.0
        gap = abs(accuracy - confidence)
        ece += (count / len(pred_events)) * gap
        rows.append(
            {
                "bin": idx,
                "confidence_low": low,
                "confidence_high": high,
                "count": count,
                "accuracy": accuracy,
                "avg_confidence": confidence,
                "abs_gap": gap,
            }
        )
    return {"ece": ece, "samples": len(pred_events), "bins": bins}, rows


def predict_ultralytics_on_items(model_path: str, items: list[dict], args: argparse.Namespace) -> tuple[list[dict], float]:
    from tqdm import tqdm
    from ultralytics import YOLO

    model = YOLO(model_path)
    predictions: list[dict] = []
    started = time.perf_counter()
    for item in tqdm(items, desc=f"Predict {Path(model_path).name}"):
        result = model.predict(
            source=str(item["image_path"]),
            imgsz=args.imgsz,
            conf=args.pred_conf,
            iou=args.nms_iou,
            device=args.device,
            verbose=False,
        )[0]
        if result.boxes is None:
            predictions.append({"boxes": [], "labels": [], "scores": []})
            continue
        predictions.append(
            {
                "boxes": result.boxes.xyxy.detach().cpu().tolist(),
                "labels": [int(x) for x in result.boxes.cls.detach().cpu().tolist()],
                "scores": [float(x) for x in result.boxes.conf.detach().cpu().tolist()],
            }
        )
    elapsed = time.perf_counter() - started
    return predictions, (elapsed / max(1, len(items))) * 1000.0


def evaluate_prediction_bundle(
    items: list[dict],
    predictions: list[dict],
    output_prefix: Path,
    model_name: str,
    split: str,
    args: argparse.Namespace,
) -> dict:
    gt = [{"boxes": item["boxes"], "labels": item["labels"], "width": item["width"], "height": item["height"]} for item in items]
    metrics = detection_metrics_iou50(gt, predictions, score_threshold=args.eval_conf)
    metrics.update({"model": model_name, "split": split, "images": len(items), "eval_conf": args.eval_conf})
    events = dataset_events(gt, predictions, score_threshold=args.eval_conf)
    size_rows = summarize_events_by_size(events, len(items))
    cost_row = summarize_cost(events, len(items), args.missed_defect_penalty)
    calibration_metrics, calibration_rows = summarize_calibration(events, args.calibration_bins)
    calibration_metrics.update({"model": model_name, "split": split})

    write_csv(output_prefix.with_name(f"{output_prefix.name}_metrics.csv"), [metrics])
    write_csv(output_prefix.with_name(f"{output_prefix.name}_per_class.csv"), metrics["per_class"])
    write_csv(output_prefix.with_name(f"{output_prefix.name}_defect_size.csv"), size_rows)
    write_csv(output_prefix.with_name(f"{output_prefix.name}_inspection_cost.csv"), [cost_row])
    write_csv(output_prefix.with_name(f"{output_prefix.name}_calibration_bins.csv"), calibration_rows)
    write_text(output_prefix.with_name(f"{output_prefix.name}_calibration_metrics.json"), json.dumps(calibration_metrics, indent=2))
    write_text(output_prefix.with_name(f"{output_prefix.name}_predictions.json"), json.dumps(predictions[: args.prediction_save_limit], indent=2))
    return metrics


def run_convert_coco(args: argparse.Namespace) -> None:
    output_dir = expand_path(args.output_dir)
    if not args.coco_root:
        raise SystemExit("--coco-root is required for --experiment convert_coco")
    class_map = json.loads(args.coco_class_map) if args.coco_class_map else {}
    summary = convert_coco_to_yolo_dataset(
        coco_root=expand_path(args.coco_root),
        output_dir=output_dir,
        dataset_name=args.dataset_name,
        file_mode=args.file_mode,
        class_map=class_map,
    )
    print(json.dumps(summary, indent=2))


def run_detector_train(args: argparse.Namespace) -> None:
    from ultralytics import YOLO

    output_dir = expand_path(args.output_dir)
    summary = load_conversion_summary(args)
    data_yaml = args.external_data_yaml or summary["data_yaml"]
    model = YOLO(args.yolo_model)
    name = args.run_name or Path(args.yolo_model).stem
    train_dir = output_dir / "runs" / "detector_train"
    result = model.train(
        data=data_yaml,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=str(train_dir),
        name=name,
        exist_ok=True,
        patience=args.patience,
        cos_lr=args.cos_lr,
        cache=args.cache,
        close_mosaic=args.close_mosaic,
    )
    best_weights = train_dir / name / "weights" / "best.pt"
    rows = []
    eval_model = YOLO(str(best_weights if best_weights.exists() else args.yolo_model))
    for split in ["val", "test"]:
        metrics = eval_model.val(data=data_yaml, imgsz=args.imgsz, batch=args.batch, device=args.device, split=split)
        rows.append(
            {
                "model": name,
                "split": split,
                "weights": str(best_weights),
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "mAP50": float(metrics.box.map50),
                "mAP50_95": float(metrics.box.map),
            }
        )
    write_csv(train_dir / name / "detector_train_metrics.csv", rows)
    summary_out = {"run_dir": str(train_dir / name), "best_weights": str(best_weights), "results_type": str(type(result)), "metrics": rows}
    write_text(train_dir / name / "detector_train_summary.json", json.dumps(summary_out, indent=2))
    print(json.dumps(summary_out, indent=2))


def run_detector_eval(args: argparse.Namespace) -> None:
    output_dir = expand_path(args.output_dir)
    if args.external_data_yaml:
        data_yaml = expand_path(args.external_data_yaml)
        items = load_yolo_items_from_data_yaml(data_yaml, args.split)
    else:
        summary = load_conversion_summary(args)
        data_yaml = Path(summary["data_yaml"])
        items = load_yolo_dataset_items(Path(summary["yolo_root"]), args.split)

    run_dir = output_dir / "runs" / "detector_eval"
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions, mean_ms = predict_ultralytics_on_items(args.yolo_weights, items, args)
    prefix = run_dir / f"{Path(args.yolo_weights).stem}_{args.split}"
    metrics = evaluate_prediction_bundle(items, predictions, prefix, Path(args.yolo_weights).stem, args.split, args)
    metrics.update({"data_yaml": str(data_yaml), "mean_ms": mean_ms})
    write_csv(prefix.with_name(f"{prefix.name}_metrics.csv"), [metrics])
    print(json.dumps(metrics, indent=2))


def wbf_fuse_image(
    yolo_pred: dict,
    transformer_pred: dict,
    width: int,
    height: int,
    policy: dict,
    args: argparse.Namespace,
) -> dict:
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Install ensemble-boxes before running adaptive fusion") from exc

    fused_boxes: list[list[float]] = []
    fused_scores: list[float] = []
    fused_labels: list[int] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        class_policy = policy["classes"].get(class_name, {})
        yolo_weight = float(class_policy.get("yolo_weight", 0.5))
        transformer_weight = float(class_policy.get("transformer_weight", 0.5))
        threshold = float(class_policy.get("conf_threshold", args.eval_conf))
        boxes_list = []
        scores_list = []
        labels_list = []
        for pred in [yolo_pred, transformer_pred]:
            boxes = []
            scores = []
            labels = []
            for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
                if int(label) != class_id or float(score) < args.pred_conf:
                    continue
                x1, y1, x2, y2 = box
                boxes.append(
                    [
                        max(0.0, min(1.0, x1 / width)),
                        max(0.0, min(1.0, y1 / height)),
                        max(0.0, min(1.0, x2 / width)),
                        max(0.0, min(1.0, y2 / height)),
                    ]
                )
                scores.append(float(score))
                labels.append(class_id)
            boxes_list.append(boxes)
            scores_list.append(scores)
            labels_list.append(labels)
        if not any(boxes_list):
            continue
        boxes, scores, labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=[yolo_weight, transformer_weight],
            iou_thr=args.fusion_iou,
            skip_box_thr=args.pred_conf,
        )
        for box, score, label in zip(boxes, scores, labels):
            if float(score) < threshold:
                continue
            fused_boxes.append([box[0] * width, box[1] * height, box[2] * width, box[3] * height])
            fused_scores.append(float(score))
            fused_labels.append(int(label))
    return {"boxes": fused_boxes, "scores": fused_scores, "labels": fused_labels}


def learn_adaptive_policy(val_items: list[dict], yolo_preds: list[dict], transformer_preds: list[dict], args: argparse.Namespace) -> dict:
    gt = [{"boxes": item["boxes"], "labels": item["labels"], "width": item["width"], "height": item["height"]} for item in val_items]
    yolo_metrics = detection_metrics_iou50(gt, yolo_preds, score_threshold=args.eval_conf)
    transformer_metrics = detection_metrics_iou50(gt, transformer_preds, score_threshold=args.eval_conf)
    policy = {
        "method": "validation_calibrated_classwise_wbf",
        "fusion_iou": args.fusion_iou,
        "base_conf": args.eval_conf,
        "classes": {},
    }
    for class_id, class_name in enumerate(CLASS_NAMES):
        y = yolo_metrics["per_class"][class_id]
        t = transformer_metrics["per_class"][class_id]
        y_f1 = 2 * y["precision"] * y["recall"] / (y["precision"] + y["recall"]) if y["precision"] + y["recall"] else 0.0
        t_f1 = 2 * t["precision"] * t["recall"] / (t["precision"] + t["recall"]) if t["precision"] + t["recall"] else 0.0
        delta = max(-0.25, min(0.25, (y_f1 - t_f1) * 0.25))
        conf = args.eval_conf
        if y["precision"] < 0.65 and t["precision"] < 0.65:
            conf += 0.10
        if y["recall"] < 0.75 and t["recall"] < 0.75:
            conf = max(args.pred_conf, conf - 0.05)
        policy["classes"][class_name] = {
            "yolo_weight": round(0.5 + delta, 3),
            "transformer_weight": round(0.5 - delta, 3),
            "conf_threshold": round(conf, 3),
            "yolo_val_precision": y["precision"],
            "yolo_val_recall": y["recall"],
            "transformer_val_precision": t["precision"],
            "transformer_val_recall": t["recall"],
        }
    return policy


def run_adaptive_fusion(args: argparse.Namespace) -> None:
    output_dir = expand_path(args.output_dir)
    if not args.rtdetr_weights:
        raise SystemExit("--rtdetr-weights is required for adaptive fusion")
    summary = load_conversion_summary(args)
    yolo_root = Path(summary["yolo_root"])
    run_dir = output_dir / "runs" / "adaptive_fusion"
    run_dir.mkdir(parents=True, exist_ok=True)

    val_items = load_yolo_dataset_items(yolo_root, "val")
    test_items = load_yolo_dataset_items(yolo_root, "test")
    val_yolo, _ = predict_ultralytics_on_items(args.yolo_weights, val_items, args)
    val_transformer, _ = predict_ultralytics_on_items(args.rtdetr_weights, val_items, args)
    policy = learn_adaptive_policy(val_items, val_yolo, val_transformer, args)
    write_text(run_dir / "adaptive_defect_aware_policy.json", json.dumps(policy, indent=2))
    write_csv(
        run_dir / "adaptive_defect_aware_policy.csv",
        [{"class_name": name, **values} for name, values in policy["classes"].items()],
    )

    test_yolo, yolo_ms = predict_ultralytics_on_items(args.yolo_weights, test_items, args)
    test_transformer, transformer_ms = predict_ultralytics_on_items(args.rtdetr_weights, test_items, args)
    fused_preds = [
        wbf_fuse_image(y_pred, t_pred, item["width"], item["height"], policy, args)
        for item, y_pred, t_pred in zip(test_items, test_yolo, test_transformer)
    ]
    prefix = run_dir / "adaptive_defect_aware_hybrid_test"
    metrics = evaluate_prediction_bundle(test_items, fused_preds, prefix, "Adaptive Defect-Aware Hybrid", "test", args)
    metrics.update({"mean_component_ms": yolo_ms + transformer_ms})
    write_csv(prefix.with_name(f"{prefix.name}_metrics.csv"), [metrics])
    print(json.dumps({"policy": policy, "test": metrics}, indent=2))

class YoloDetectionDataset:
    def __init__(self, yolo_root: Path, split: str, max_items: int | None = None) -> None:
        self.items = load_yolo_dataset_items(yolo_root, split)
        if max_items is not None:
            self.items = self.items[:max_items]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        import torch
        from PIL import Image
        from torchvision.transforms import functional as F

        item = self.items[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        tensor = F.to_tensor(image)
        boxes = torch.tensor(item["boxes"], dtype=torch.float32)
        labels = torch.tensor([label + 1 for label in item["labels"]], dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) else torch.zeros((0,), dtype=torch.float32)
        target = {
            "boxes": boxes.reshape(-1, 4),
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
        }
        return tensor, target


def collate_detection_batch(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)


def run_faster_rcnn(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from tqdm import tqdm

    output_dir = expand_path(args.output_dir)
    summary = load_conversion_summary(args)
    yolo_root = Path(summary["yolo_root"])
    run_dir = output_dir / "runs" / "faster_rcnn"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = YoloDetectionDataset(yolo_root, "train", max_items=args.max_train_items)
    val_dataset = YoloDetectionDataset(yolo_root, "val", max_items=args.max_eval_items)
    test_dataset = YoloDetectionDataset(yolo_root, "test", max_items=args.max_eval_items)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.frcnn_batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_detection_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_detection_batch,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_detection_batch,
    )

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() and args.device != "cpu" else "cpu")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if args.pretrained else None
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(CLASS_NAMES) + 1)
    model.to(device)
    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.frcnn_lr,
        momentum=0.9,
        weight_decay=0.0005,
    )

    history: list[dict] = []
    for epoch in range(args.frcnn_epochs):
        model.train()
        total_loss = 0.0
        batches = 0
        for images, targets in tqdm(train_loader, desc=f"Faster R-CNN epoch {epoch + 1}/{args.frcnn_epochs}"):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]
            losses = model(images, targets)
            loss = sum(losses.values())
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            batches += 1
        history.append({"epoch": epoch + 1, "train_loss": total_loss / max(1, batches)})
        write_csv(run_dir / "training_history.csv", history, ["epoch", "train_loss"])

    def collect_predictions(loader: DataLoader, dataset: YoloDetectionDataset, split: str) -> tuple[list[dict], dict]:
        model.eval()
        preds: list[dict] = []
        started = time.perf_counter()
        with torch.no_grad():
            for images, _targets in tqdm(loader, desc=f"Faster R-CNN predict {split}"):
                outputs = model([img.to(device) for img in images])
                for output in outputs:
                    preds.append(
                        {
                            "boxes": output["boxes"].detach().cpu().tolist(),
                            "labels": [int(x) - 1 for x in output["labels"].detach().cpu().tolist()],
                            "scores": [float(x) for x in output["scores"].detach().cpu().tolist()],
                        }
                    )
        elapsed = time.perf_counter() - started
        gt = [{"boxes": item["boxes"], "labels": item["labels"]} for item in dataset.items]
        metrics = detection_metrics_iou50(gt, preds, score_threshold=args.eval_conf)
        metrics.update(
            {
                "model": "Faster R-CNN ResNet50-FPN",
                "split": split,
                "images": len(dataset),
                "eval_conf": args.eval_conf,
                "mean_ms": (elapsed / max(1, len(dataset))) * 1000.0,
            }
        )
        return preds, metrics

    val_preds, val_metrics = collect_predictions(val_loader, val_dataset, "val")
    test_preds, test_metrics = collect_predictions(test_loader, test_dataset, "test")
    torch.save(model.state_dict(), run_dir / "faster_rcnn_resnet50_fpn.pt")
    write_text(run_dir / "val_predictions.json", json.dumps(val_preds[: args.prediction_save_limit], indent=2))
    write_text(run_dir / "test_predictions.json", json.dumps(test_preds[: args.prediction_save_limit], indent=2))
    write_csv(run_dir / "faster_rcnn_metrics.csv", [val_metrics, test_metrics])
    write_csv(run_dir / "faster_rcnn_per_class_test.csv", test_metrics["per_class"])
    write_text(run_dir / "faster_rcnn_summary.json", json.dumps({"val": val_metrics, "test": test_metrics}, indent=2))
    print(json.dumps({"val": val_metrics, "test": test_metrics}, indent=2))


def run_cross_dataset(args: argparse.Namespace) -> None:
    """Run external YOLO-format evaluation when external data.yaml files exist."""
    from ultralytics import YOLO

    output_dir = expand_path(args.output_dir)
    data_root = expand_path(args.data_root)
    run_dir = output_dir / "runs" / "cross_dataset"
    run_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    if args.external_data_yaml:
        candidates.append(expand_path(args.external_data_yaml))
    candidates.extend(
        [
            data_root / "deeppcb_yolo" / "data.yaml",
            data_root / "deeppcb" / "data.yaml",
            data_root / "mendeley_pcb_yolo" / "data.yaml",
            data_root / "mendeley_pcb" / "data.yaml",
            data_root / "dspcbsd_yolo" / "data.yaml",
            data_root / "dspcbsd" / "data.yaml",
        ]
    )
    existing = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate.exists() and candidate not in seen:
            existing.append(candidate)
            seen.add(candidate)

    if not existing:
        status = {
            "status": "missing_external_yolo_dataset",
            "message": "No external YOLO data.yaml found. Add DeepPCB/Mendeley converted data.yaml, then rerun.",
            "searched": [str(path) for path in candidates],
        }
        write_text(run_dir / "cross_dataset_status.json", json.dumps(status, indent=2))
        print(json.dumps(status, indent=2))
        return

    model = YOLO(args.yolo_weights)
    rows = []
    for data_yaml in existing:
        result = model.val(data=str(data_yaml), imgsz=args.imgsz, batch=args.batch, device=args.device, split=args.split)
        row = {
            "dataset_yaml": str(data_yaml),
            "split": args.split,
            "precision": float(result.box.mp),
            "recall": float(result.box.mr),
            "mAP50": float(result.box.map50),
            "mAP50_95": float(result.box.map),
        }
        rows.append(row)
    write_csv(run_dir / "cross_dataset_metrics.csv", rows)
    print(json.dumps(rows, indent=2))


def run_segmentation_pilot(args: argparse.Namespace) -> None:
    """Create a pseudo-mask segmentation pilot artifact from detection boxes."""
    output_dir = expand_path(args.output_dir)
    summary = load_conversion_summary(args)
    yolo_root = Path(summary["yolo_root"])
    run_dir = output_dir / "runs" / "segmentation_pilot"
    run_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {
            "description": "Pseudo-mask pilot generated from PCB bounding boxes. Rectangular masks are not ground-truth segmentation.",
            "source": "current_pcb_yolo detection boxes",
        },
        "categories": [{"id": i + 1, "name": name} for i, name in enumerate(CLASS_NAMES)],
        "images": [],
        "annotations": [],
    }
    ann_id = 1
    for split in ["train", "val", "test"]:
        for item in load_yolo_dataset_items(yolo_root, split):
            image_id = len(coco["images"]) + 1
            coco["images"].append(
                {
                    "id": image_id,
                    "file_name": str(item["image_path"]),
                    "width": item["width"],
                    "height": item["height"],
                    "split": split,
                }
            )
            for box, label in zip(item["boxes"], item["labels"]):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                segmentation = [[x1, y1, x2, y1, x2, y2, x1, y2]]
                coco["annotations"].append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": label + 1,
                        "bbox": [x1, y1, width, height],
                        "area": width * height,
                        "iscrowd": 0,
                        "segmentation": segmentation,
                    }
                )
                ann_id += 1

    summary_out = {
        "status": "pseudo_masks_created",
        "warning": "These are rectangular pseudo-masks derived from bounding boxes, not true pixel-level annotations.",
        "images": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "classes": CLASS_NAMES,
        "coco_json": str(run_dir / "pseudo_mask_coco.json"),
    }
    write_text(run_dir / "pseudo_mask_coco.json", json.dumps(coco, indent=2))
    write_text(run_dir / "segmentation_pilot_summary.json", json.dumps(summary_out, indent=2))
    print(json.dumps(summary_out, indent=2))


def run_research_batch(args: argparse.Namespace) -> None:
    output_dir = expand_path(args.output_dir)
    batch_log = output_dir / "research_batch_status.json"
    steps = []
    for name, fn in [
        ("smoke", run_smoke),
        ("segmentation_pilot", run_segmentation_pilot),
        ("faster_rcnn", run_faster_rcnn),
        ("cross_dataset", run_cross_dataset),
    ]:
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            fn(args)
            steps.append({"step": name, "status": "completed", "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        except Exception as exc:  # noqa: BLE001
            steps.append(
                {
                    "step": name,
                    "status": "failed",
                    "started": started,
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "error": repr(exc),
                }
            )
            write_text(batch_log, json.dumps({"steps": steps}, indent=2))
            raise
        write_text(batch_log, json.dumps({"steps": steps}, indent=2))
    print(json.dumps({"steps": steps}, indent=2))


def run_publication_batch(args: argparse.Namespace) -> None:
    """Run publication-oriented analyses that do not require retraining by default."""
    output_dir = expand_path(args.output_dir)
    batch_log = output_dir / "publication_batch_status.json"
    steps: list[dict] = []
    selected_steps = [
        ("smoke", run_smoke),
        ("segmentation_pilot", run_segmentation_pilot),
        ("detector_eval", run_detector_eval),
    ]
    if args.rtdetr_weights:
        selected_steps.append(("adaptive_fusion", run_adaptive_fusion))
    selected_steps.append(("cross_dataset", run_cross_dataset))

    for name, fn in selected_steps:
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            fn(args)
            steps.append({"step": name, "status": "completed", "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S")})
        except Exception as exc:  # noqa: BLE001
            steps.append(
                {
                    "step": name,
                    "status": "failed",
                    "started": started,
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "error": repr(exc),
                }
            )
            write_text(batch_log, json.dumps({"steps": steps}, indent=2))
            raise
        write_text(batch_log, json.dumps({"steps": steps}, indent=2))
    print(json.dumps({"steps": steps}, indent=2))


def placeholder(name: str) -> None:
    raise SystemExit(
        f"Experiment {name!r} is not implemented yet. "
        "Run --experiment smoke or --experiment yolo_smoke first."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=[
            "smoke",
            "yolo_smoke",
            "detector_train",
            "detector_eval",
            "faster_rcnn",
            "cross_dataset",
            "segmentation_pilot",
            "convert_coco",
            "adaptive_fusion",
            "research_batch",
            "publication_batch",
        ],
        required=True,
    )
    parser.add_argument("--data-root", default="~/data", help="Root containing current_pcb/, deeppcb/, etc.")
    parser.add_argument("--output-dir", default="~/outputs/nautilus", help="Directory for all generated outputs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--file-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--yolo-model", default="yolo11n.pt")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--cos-lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--frcnn-epochs", type=int, default=5)
    parser.add_argument("--frcnn-batch", type=int, default=2)
    parser.add_argument("--frcnn-lr", type=float, default=0.005)
    parser.add_argument("--eval-conf", type=float, default=0.25)
    parser.add_argument("--pred-conf", type=float, default=0.01)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--fusion-iou", type=float, default=0.55)
    parser.add_argument("--missed-defect-penalty", type=float, default=5.0)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--prediction-save-limit", type=int, default=20)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--external-data-yaml", default=None)
    parser.add_argument("--yolo-weights", default="yolo11n.pt")
    parser.add_argument("--rtdetr-weights", default=None)
    parser.add_argument("--coco-root", default=None)
    parser.add_argument("--dataset-name", default="dspcbsd")
    parser.add_argument("--coco-class-map", default=None, help="JSON mapping from external category names to project class names.")
    parser.add_argument("--split", default="test")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.experiment == "smoke":
        run_smoke(args)
    elif args.experiment == "yolo_smoke":
        run_yolo_smoke(args)
    elif args.experiment == "detector_train":
        run_detector_train(args)
    elif args.experiment == "detector_eval":
        run_detector_eval(args)
    elif args.experiment == "faster_rcnn":
        run_faster_rcnn(args)
    elif args.experiment == "cross_dataset":
        run_cross_dataset(args)
    elif args.experiment == "segmentation_pilot":
        run_segmentation_pilot(args)
    elif args.experiment == "convert_coco":
        run_convert_coco(args)
    elif args.experiment == "adaptive_fusion":
        run_adaptive_fusion(args)
    elif args.experiment == "research_batch":
        run_research_batch(args)
    elif args.experiment == "publication_batch":
        run_publication_batch(args)
    else:
        placeholder(args.experiment)


if __name__ == "__main__":
    main()
