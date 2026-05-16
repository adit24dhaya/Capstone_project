#!/usr/bin/env python3
"""Portable Nautilus experiment runner for PCB defect experiments.

This script avoids Kaggle-only paths and writes all generated artifacts under
an output directory, defaulting to ``~/outputs/nautilus``.
"""

from __future__ import annotations

import argparse
import csv
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


def placeholder(name: str) -> None:
    raise SystemExit(
        f"Experiment {name!r} is not implemented yet. "
        "Run --experiment smoke or --experiment yolo_smoke first."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        choices=["smoke", "yolo_smoke", "faster_rcnn", "cross_dataset", "segmentation_pilot", "research_batch"],
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
    parser.add_argument("--max-train-items", type=int, default=None)
    parser.add_argument("--max-eval-items", type=int, default=None)
    parser.add_argument("--frcnn-epochs", type=int, default=5)
    parser.add_argument("--frcnn-batch", type=int, default=2)
    parser.add_argument("--frcnn-lr", type=float, default=0.005)
    parser.add_argument("--eval-conf", type=float, default=0.25)
    parser.add_argument("--prediction-save-limit", type=int, default=20)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--external-data-yaml", default=None)
    parser.add_argument("--yolo-weights", default="yolo11n.pt")
    parser.add_argument("--split", default="test")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.experiment == "smoke":
        run_smoke(args)
    elif args.experiment == "yolo_smoke":
        run_yolo_smoke(args)
    elif args.experiment == "faster_rcnn":
        run_faster_rcnn(args)
    elif args.experiment == "cross_dataset":
        run_cross_dataset(args)
    elif args.experiment == "segmentation_pilot":
        run_segmentation_pilot(args)
    elif args.experiment == "research_batch":
        run_research_batch(args)
    else:
        placeholder(args.experiment)


if __name__ == "__main__":
    main()
