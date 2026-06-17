#!/usr/bin/env python3
"""Build the ESCS'26 PCB dataset with reproducible legacy or corrected labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15
DSPCBSD_PLUS_URL = "https://ndownloader.figshare.com/files/44069552"
DSPCBSD_PLUS_MD5 = "508334b65bdaea7336f4c1b5d5a80a81"
DSPCBSD_PLUS_DOI = "10.6084/m9.figshare.24970329.v1"
DSPCBSD_YOLO_CODES = ["SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO"]
DSPCBSD_TO_PROJECT_CLASS = {
    "SH": "Short",
    "SP": "Spur",
    "SC": "Spurious_copper",
    "OP": "Open_circuit",
    "MB": "Mouse_bite",
}
CANONICAL_CLASSES = [
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper",
]
LEGACY_CLASSES = [
    "Mouse_bite",
    "Spur",
    "Open_circuit",
    "Short",
    "Missing_hole",
    "Spurious_copper",
]
SYNTHETIC_TARGETS = {
    "Missing_hole": 320,
    "Mouse_bite": 240,
    "Spur": 240,
}
SYNTHETIC_PATCH_CONTEXT_RANGE = (1.4, 2.4)
EXPECTED_IMAGES = {"train": 5551, "val": 1016, "test": 1016}
EXPECTED_INSTANCES = {"val": 2106, "test": 2179}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

Record = tuple[Path, list[tuple[int, float, float, float, float]], str]


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_fingerprint(paths: Iterable[Path], root: Path, include_content: bool) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        if include_content:
            digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def normalize_class(raw_name: str) -> str:
    normalized = raw_name.strip().replace(" ", "_").lower()
    aliases = {name.lower(): name for name in CANONICAL_CLASSES}
    if normalized not in aliases:
        raise ValueError(f"Unknown PCB class: {raw_name!r}")
    return aliases[normalized]


def find_base_dataset_root(candidate: Path) -> Path:
    candidate = candidate.expanduser().resolve()
    probes = [candidate]
    probes.extend(path for path in candidate.rglob("Annotations") if path.is_dir())
    for probe in probes:
        root = probe.parent if probe.name == "Annotations" else probe
        if (root / "Annotations").is_dir() and (root / "images").is_dir():
            return root
    raise FileNotFoundError(f"Could not find PCB-DATASET root under {candidate}")


def find_dspcbsd_yolo_root(extract_root: Path) -> Path | None:
    candidates = [extract_root / "Data_YOLO"]
    candidates.extend(path for path in extract_root.rglob("Data_YOLO") if path.is_dir())
    for candidate in candidates:
        if (candidate / "images").is_dir() and (candidate / "labels").is_dir():
            return candidate
    return None


def download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; PCB-capstone-repro/1.0)"},
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=600) as response:
        with destination.open("wb") as handle:
            shutil.copyfileobj(response, handle)


def prepare_dspcbsd_plus(cache_dir: Path) -> Path:
    cache_root = cache_dir / "DsPCBSD_plus"
    cache_root.mkdir(parents=True, exist_ok=True)
    yolo_root = find_dspcbsd_yolo_root(cache_root)
    if yolo_root is not None:
        return yolo_root

    zip_path = cache_root / "DsPCBSD_plus.zip"
    if zip_path.exists() and md5_file(zip_path) != DSPCBSD_PLUS_MD5:
        zip_path.unlink()
    if not zip_path.exists():
        download_file(DSPCBSD_PLUS_URL, zip_path)
    actual_md5 = md5_file(zip_path)
    if actual_md5 != DSPCBSD_PLUS_MD5:
        raise ValueError(
            f"DsPCBSD+ checksum mismatch: expected {DSPCBSD_PLUS_MD5}, got {actual_md5}"
        )
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(cache_root)
    yolo_root = find_dspcbsd_yolo_root(cache_root)
    if yolo_root is None:
        raise FileNotFoundError(f"Could not find Data_YOLO under {cache_root}")
    return yolo_root


def voc_box_to_yolo(
    box: ET.Element, width: int, height: int
) -> tuple[float, float, float, float]:
    xmin = float(box.findtext("xmin", "0"))
    ymin = float(box.findtext("ymin", "0"))
    xmax = float(box.findtext("xmax", "0"))
    ymax = float(box.findtext("ymax", "0"))
    box_width = max(0.0, xmax - xmin)
    box_height = max(0.0, ymax - ymin)
    return (
        ((xmin + xmax) / 2.0) / width,
        ((ymin + ymax) / 2.0) / height,
        box_width / width,
        box_height / height,
    )


def parse_base_annotation(
    xml_path: Path, dataset_root: Path, class_to_id: dict[str, int]
) -> Record:
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename")
    if not filename:
        raise ValueError(f"Missing filename in {xml_path}")
    folder = root.findtext("folder") or xml_path.parent.name
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    image_path = dataset_root / "images" / folder / filename
    if not image_path.exists():
        matches = list((dataset_root / "images").rglob(filename))
        if not matches:
            raise FileNotFoundError(f"Missing image for {xml_path}: {filename}")
        image_path = matches[0]

    labels: list[tuple[int, float, float, float, float]] = []
    for obj in root.findall("object"):
        class_name = normalize_class(obj.findtext("name", ""))
        box = obj.find("bndbox")
        if box is None:
            continue
        x, y, width_norm, height_norm = voc_box_to_yolo(box, width, height)
        if width_norm > 0 and height_norm > 0:
            labels.append((class_to_id[class_name], x, y, width_norm, height_norm))
    return image_path, labels, folder


def find_matching_image(image_dir: Path, stem: str) -> Path | None:
    for suffix in sorted(IMAGE_SUFFIXES):
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    matches = [
        path
        for path in image_dir.glob(f"{stem}.*")
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]
    return matches[0] if matches else None


def parse_dspcbsd_label(
    label_path: Path, class_to_id: dict[str, int]
) -> tuple[list[tuple[int, float, float, float, float]], Counter]:
    labels: list[tuple[int, float, float, float, float]] = []
    skipped: Counter = Counter()
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            skipped["malformed"] += 1
            continue
        source_id = int(float(parts[0]))
        if source_id < 0 or source_id >= len(DSPCBSD_YOLO_CODES):
            skipped["unknown_id"] += 1
            continue
        source_code = DSPCBSD_YOLO_CODES[source_id]
        class_name = DSPCBSD_TO_PROJECT_CLASS.get(source_code)
        if class_name is None:
            skipped[source_code] += 1
            continue
        x, y, width_norm, height_norm = map(float, parts[1:])
        if width_norm > 0 and height_norm > 0:
            labels.append(
                (class_to_id[class_name], x, y, width_norm, height_norm)
            )
        else:
            skipped["invalid_box"] += 1
    return labels, skipped


def load_grouped_records(
    base_root: Path, dspcbsd_root: Path, class_names: list[str]
) -> tuple[dict[str, list[Record]], dict]:
    class_to_id = {name: index for index, name in enumerate(class_names)}
    records_by_group: dict[str, list[Record]] = defaultdict(list)
    source_counts: Counter = Counter()

    for xml_path in sorted((base_root / "Annotations").rglob("*.xml")):
        record = parse_base_annotation(xml_path, base_root, class_to_id)
        if record[1]:
            records_by_group[record[2]].append(record)
            source_counts["PCB-DATASET"] += 1

    skipped_boxes: Counter = Counter()
    for source_split in ("train", "val"):
        label_dir = dspcbsd_root / "labels" / source_split
        image_dir = dspcbsd_root / "images" / source_split
        if not label_dir.is_dir():
            continue
        for label_path in sorted(label_dir.glob("*.txt")):
            labels, skipped = parse_dspcbsd_label(label_path, class_to_id)
            skipped_boxes.update(skipped)
            if not labels:
                continue
            image_path = find_matching_image(image_dir, label_path.stem)
            if image_path is None:
                raise FileNotFoundError(f"Missing DsPCBSD+ image for {label_path}")
            primary_class = class_names[labels[0][0]]
            group = f"DsPCBSD_plus_{source_split}_{primary_class}"
            records_by_group[group].append((image_path, labels, group))
            source_counts["DsPCBSD+"] += 1

    return records_by_group, {
        "source_images": dict(source_counts),
        "dspcbsd_skipped_boxes": dict(skipped_boxes),
    }


def split_records(
    records_by_group: dict[str, list[Record]]
) -> dict[str, list[Record]]:
    split = {"train": [], "val": [], "test": []}
    rng = random.Random(SEED)
    for _, records in sorted(records_by_group.items()):
        records = list(records)
        rng.shuffle(records)
        validation_count = max(1, int(round(len(records) * VAL_RATIO)))
        test_count = max(1, int(round(len(records) * TEST_RATIO)))
        split["val"].extend(records[:validation_count])
        split["test"].extend(records[validation_count : validation_count + test_count])
        split["train"].extend(records[validation_count + test_count :])
    for records in split.values():
        rng.shuffle(records)
    return split


def reset_output(output_root: Path) -> None:
    if output_root.exists():
        shutil.rmtree(output_root)
    for split in ("train", "val", "test"):
        (output_root / split / "images").mkdir(parents=True, exist_ok=True)
        (output_root / split / "labels").mkdir(parents=True, exist_ok=True)


def write_split(records: list[Record], split: str, output_root: Path) -> None:
    for image_path, labels, folder in records:
        safe_name = f"{folder}_{image_path.name}"
        destination_image = output_root / split / "images" / safe_name
        destination_label = (
            output_root / split / "labels" / f"{Path(safe_name).stem}.txt"
        )
        shutil.copy2(image_path, destination_image)
        lines = [
            f"{class_id} {x:.6f} {y:.6f} {width:.6f} {height:.6f}"
            for class_id, x, y, width, height in labels
        ]
        destination_label.write_text("\n".join(lines) + "\n")


def read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append(
                (
                    int(float(parts[0])),
                    float(parts[1]),
                    float(parts[2]),
                    float(parts[3]),
                    float(parts[4]),
                )
            )
    return rows


def yolo_to_xyxy(
    row: tuple[int, float, float, float, float], width: int, height: int
) -> tuple[float, float, float, float]:
    _, x, y, box_width, box_height = row
    return (
        (x - box_width / 2.0) * width,
        (y - box_height / 2.0) * height,
        (x + box_width / 2.0) * width,
        (y + box_height / 2.0) * height,
    )


def xyxy_to_yolo(
    class_id: int,
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        class_id,
        ((x1 + x2) / 2.0) / width,
        ((y1 + y2) / 2.0) / height,
        max(0.0, x2 - x1) / width,
        max(0.0, y2 - y1) / height,
    )


def build_synthetic_images(
    output_root: Path, class_names: list[str]
) -> list[dict[str, object]]:
    try:
        import albumentations as transforms
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Synthetic augmentation requires albumentations, opencv-python-headless, and numpy"
        ) from exc

    random.seed(SEED)
    np.random.seed(SEED)
    rng = random.Random(SEED)
    class_to_id = {name: index for index, name in enumerate(class_names)}
    image_dir = output_root / "train" / "images"
    label_dir = output_root / "train" / "labels"
    image_paths = sorted(
        path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    pool: dict[int, list[tuple[Path, tuple[int, float, float, float, float]]]] = {
        index: [] for index in range(len(class_names))
    }
    for image_path in image_paths:
        for row in read_yolo_labels(label_dir / f"{image_path.stem}.txt"):
            pool[row[0]].append((image_path, row))

    try:
        noise = transforms.GaussNoise(
            std_range=(0.012, 0.028), mean_range=(0.0, 0.0), p=0.3
        )
    except TypeError:
        noise = transforms.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
    augmentation = transforms.Compose(
        [
            transforms.RandomBrightnessContrast(p=0.4),
            noise,
            transforms.MotionBlur(blur_limit=7, p=0.2),
            transforms.Rotate(limit=10, p=0.3),
            transforms.HorizontalFlip(p=0.5),
        ],
        bbox_params=transforms.BboxParams(
            format="pascal_voc", label_fields=["labels"], min_visibility=0.2
        ),
    )

    summary = []
    for class_name, requested in SYNTHETIC_TARGETS.items():
        class_id = class_to_id[class_name]
        generated = 0
        attempts = 0
        while generated < requested and attempts < requested * 20:
            attempts += 1
            source_path, source_label = rng.choice(pool[class_id])
            background_path = rng.choice(image_paths)
            source = cv2.imread(str(source_path))
            background = cv2.imread(str(background_path))
            if source is None or background is None:
                continue

            source_height, source_width = source.shape[:2]
            x1, y1, x2, y2 = yolo_to_xyxy(
                source_label, source_width, source_height
            )
            box_width = max(1.0, x2 - x1)
            box_height = max(1.0, y2 - y1)
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0
            context = rng.uniform(*SYNTHETIC_PATCH_CONTEXT_RANGE)
            crop_width = max(box_width + 4, int(box_width * context))
            crop_height = max(box_height + 4, int(box_height * context))
            crop_x1 = int(max(0, center_x - crop_width / 2.0))
            crop_y1 = int(max(0, center_y - crop_height / 2.0))
            crop_x2 = int(min(source_width, center_x + crop_width / 2.0))
            crop_y2 = int(min(source_height, center_y + crop_height / 2.0))
            patch = source[crop_y1:crop_y2, crop_x1:crop_x2].copy()
            if patch.size == 0 or min(patch.shape[:2]) < 6:
                continue
            patch_box = [
                max(0.0, x1 - crop_x1),
                max(0.0, y1 - crop_y1),
                min(float(patch.shape[1]), x2 - crop_x1),
                min(float(patch.shape[0]), y2 - crop_y1),
            ]
            if patch_box[2] <= patch_box[0] + 1 or patch_box[3] <= patch_box[1] + 1:
                continue
            transformed = augmentation(
                image=patch, bboxes=[patch_box], labels=[class_id]
            )
            if not transformed["bboxes"]:
                continue
            patch = transformed["image"]
            patch_box = list(transformed["bboxes"][0])
            original_patch_height, original_patch_width = patch.shape[:2]
            scale = rng.uniform(0.75, 1.35)
            new_width = max(4, int(original_patch_width * scale))
            new_height = max(4, int(original_patch_height * scale))
            if (
                new_width >= background.shape[1]
                or new_height >= background.shape[0]
            ):
                continue
            patch = cv2.resize(
                patch, (new_width, new_height), interpolation=cv2.INTER_LINEAR
            )
            paste_x = rng.randint(0, background.shape[1] - new_width)
            paste_y = rng.randint(0, background.shape[0] - new_height)
            background[
                paste_y : paste_y + new_height, paste_x : paste_x + new_width
            ] = patch
            relative_box = [
                patch_box[0] / max(1, original_patch_width),
                patch_box[1] / max(1, original_patch_height),
                patch_box[2] / max(1, original_patch_width),
                patch_box[3] / max(1, original_patch_height),
            ]
            pasted_box = (
                paste_x + relative_box[0] * new_width,
                paste_y + relative_box[1] * new_height,
                paste_x + relative_box[2] * new_width,
                paste_y + relative_box[3] * new_height,
            )
            if pasted_box[2] <= pasted_box[0] + 1 or pasted_box[3] <= pasted_box[1] + 1:
                continue
            base_labels = read_yolo_labels(
                label_dir / f"{background_path.stem}.txt"
            )
            synthetic_label = xyxy_to_yolo(
                class_id,
                pasted_box,
                background.shape[1],
                background.shape[0],
            )
            output_stem = (
                f"synthetic_{class_name}_{generated:04d}_{background_path.stem}"
            )
            if not cv2.imwrite(str(image_dir / f"{output_stem}.jpg"), background):
                raise OSError(f"Could not write synthetic image {output_stem}")
            lines = [
                f"{int(row[0])} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}"
                for row in base_labels + [synthetic_label]
            ]
            (label_dir / f"{output_stem}.txt").write_text(
                "\n".join(lines) + "\n"
            )
            generated += 1
        if generated != requested:
            raise RuntimeError(
                f"Generated {generated}/{requested} synthetic {class_name} images"
            )
        summary.append(
            {
                "class": class_name,
                "requested": requested,
                "generated": generated,
                "source_instances": len(pool[class_id]),
                "settings": {
                    "context_range": SYNTHETIC_PATCH_CONTEXT_RANGE,
                    "scale_range": [0.75, 1.35],
                    "brightness_contrast_probability": 0.4,
                    "noise_probability": 0.3,
                    "motion_blur_probability": 0.2,
                    "rotation_degrees": 10,
                    "horizontal_flip_probability": 0.5,
                },
            }
        )
    return summary


def write_data_yaml(output_root: Path, class_names: list[str]) -> Path:
    data_yaml = output_root / "data.yaml"
    names = "\n".join(
        f"  {index}: {name}" for index, name in enumerate(class_names)
    )
    data_yaml.write_text(
        f"path: {output_root.resolve()}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"names:\n{names}\n"
    )
    return data_yaml


def split_statistics(
    output_root: Path, class_names: list[str]
) -> dict[str, dict[str, object]]:
    statistics = {}
    for split in ("train", "val", "test"):
        image_paths = [
            path
            for path in (output_root / split / "images").iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        ]
        label_paths = sorted((output_root / split / "labels").glob("*.txt"))
        class_instances: Counter = Counter()
        instances = 0
        for label_path in label_paths:
            for row in read_yolo_labels(label_path):
                if row[0] < 0 or row[0] >= len(class_names):
                    raise ValueError(f"Invalid class ID in {label_path}: {row[0]}")
                class_instances[class_names[row[0]]] += 1
                instances += 1
        statistics[split] = {
            "images": len(image_paths),
            "labels": len(label_paths),
            "instances": instances,
            "instances_by_class": dict(class_instances),
            "filename_sha256": tree_fingerprint(
                image_paths, output_root, include_content=False
            ),
            "label_sha256": tree_fingerprint(
                label_paths, output_root, include_content=True
            ),
        }
    return statistics


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def validate_expected_counts(statistics: dict[str, dict[str, object]]) -> None:
    failures = []
    for split, expected in EXPECTED_IMAGES.items():
        actual = statistics[split]["images"]
        if actual != expected:
            failures.append(f"{split} images: expected {expected}, got {actual}")
    for split, expected in EXPECTED_INSTANCES.items():
        actual = statistics[split]["instances"]
        if actual != expected:
            failures.append(f"{split} instances: expected {expected}, got {actual}")
    if failures:
        raise RuntimeError("Dataset verification failed:\n- " + "\n- ".join(failures))


def build_dataset(args: argparse.Namespace) -> Path:
    class_names = LEGACY_CLASSES if args.label_mode == "legacy" else CANONICAL_CLASSES
    base_root = find_base_dataset_root(args.base_root)
    dspcbsd_root = prepare_dspcbsd_plus(args.cache_dir)
    records_by_group, source_summary = load_grouped_records(
        base_root, dspcbsd_root, class_names
    )
    splits = split_records(records_by_group)
    reset_output(args.output)
    for split, records in splits.items():
        write_split(records, split, args.output)
    synthetic_summary = build_synthetic_images(args.output, class_names)
    data_yaml = write_data_yaml(args.output, class_names)
    statistics = split_statistics(args.output, class_names)
    validate_expected_counts(statistics)

    manifest = {
        "schema_version": 1,
        "label_mode": args.label_mode,
        "seed": SEED,
        "validation_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "classes": class_names,
        "canonical_classes": CANONICAL_CLASSES,
        "legacy_mapping_warning": (
            "The accepted baseline used this historical ID layout. Do not use "
            "legacy per-class names for camera-ready semantic claims."
            if args.label_mode == "legacy"
            else None
        ),
        "base_dataset": {
            "root": str(base_root),
            "annotation_files": len(list((base_root / "Annotations").rglob("*.xml"))),
        },
        "dspcbsd_plus": {
            "root": str(dspcbsd_root),
            "download_url": DSPCBSD_PLUS_URL,
            "archive_md5": DSPCBSD_PLUS_MD5,
            "doi": DSPCBSD_PLUS_DOI,
        },
        "source_summary": source_summary,
        "synthetic_augmentation": synthetic_summary,
        "splits": statistics,
        "data_yaml": str(data_yaml),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "hostname": platform.node(),
            "numpy": package_version("numpy"),
            "opencv_python_headless": package_version("opencv-python-headless"),
            "albumentations": package_version("albumentations"),
            "ultralytics": package_version("ultralytics"),
        },
    }
    manifest_path = args.output / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"dataset": str(args.output), "splits": statistics}, indent=2))
    return manifest_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-root",
        type=Path,
        required=True,
        help="PCB-DATASET root or a directory containing it",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Download/extraction cache for DsPCBSD+",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--label-mode",
        choices=("legacy", "corrected"),
        required=True,
        help="Legacy reproduces accepted IDs; corrected uses canonical semantics",
    )
    return parser


def main() -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(SEED))
    args = build_parser().parse_args()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    manifest_path = build_dataset(args)
    print(f"Verified dataset manifest: {manifest_path}")


if __name__ == "__main__":
    main()
