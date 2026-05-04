import json
from pathlib import Path


def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip("\n").splitlines(True),
    }


def markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip("\n").splitlines(True),
    }


cells = [
    markdown_cell(
        """
# Automated PCB Defect Detection with Deep Learning

This notebook is configured for Kaggle GPU. It prepares the PCB dataset, converts annotations to YOLO labels, trains a YOLOv8 detector, runs Albumentations-based robustness tests, benchmarks a transformer-style RT-DETR detector, measures inference latency, and exports deployment artifacts.
"""
    ),
    code_cell(
        """
import importlib.util
import subprocess
import sys

required_packages = {
    "ultralytics": "ultralytics",
    "albumentations": "albumentations",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
}
missing = [pip_name for module_name, pip_name in required_packages.items() if importlib.util.find_spec(module_name) is None]

if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
else:
    print("ultralytics, albumentations, onnx, and onnxruntime are already installed")
"""
    ),
    code_cell(
        """
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import random
import shutil
import time
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import HTML, display
from ultralytics import YOLO

os.environ.setdefault("WANDB_MODE", "disabled")

SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15
EPOCHS = 50
IMG_SIZE = 640
BATCH = 16
RUN_RTDETR_BENCHMARK = True
RTDETR_EPOCHS = 10
RTDETR_BATCH = 4
RTDETR_IMG_SIZE = 640
RUN_ROBUSTNESS_TESTS = True
ROBUSTNESS_SAMPLE_SIZE = 250
RUN_TENSORRT_EXPORT = False
USE_DSPCBSD_PLUS = True
MAX_DSPCBSD_PLUS_IMAGES = None

DSPCBSD_PLUS_URL = "https://ndownloader.figshare.com/files/44069552"
DSPCBSD_PLUS_MD5 = "508334b65bdaea7336f4c1b5d5a80a81"
DSPCBSD_PLUS_DOI = "10.6084/m9.figshare.24970329.v1"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

CLASSES = [
    "Mouse_bite",
    "Spur",
    "Open_circuit",
    "Short",
    "Missing_hole",
    "Spurious_copper",
]

CLASS_TO_ID = {name.lower(): i for i, name in enumerate(CLASSES)}

DSPCBSD_YOLO_CODES = ["SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO"]
DSPCBSD_TO_PROJECT_CLASS = {
    "SH": "Short",
    "SP": "Spur",
    "SC": "Spurious_copper",
    "OP": "Open_circuit",
    "MB": "Mouse_bite",
}

WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("working")
CACHE_DIR = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else WORK_DIR / "external"
YOLO_ROOT = WORK_DIR / "YOLO_PCB"
DATA_YAML = YOLO_ROOT / "data.yaml"
RUN_DIR = WORK_DIR / "runs/detect/train"
WEIGHTS = RUN_DIR / "weights/best.pt"
VAL_IMG_DIR = YOLO_ROOT / "val/images"
VAL_LBL_DIR = YOLO_ROOT / "val/labels"
TEST_IMG_DIR = YOLO_ROOT / "test/images"
TEST_LBL_DIR = YOLO_ROOT / "test/labels"
PRED_SAVE_DIR = WORK_DIR / "vis_predictions/val_preds"
SUMMARY_CSV = WORK_DIR / "project_metrics_summary.csv"
ROBUSTNESS_CSV = WORK_DIR / "robustness_metrics.csv"
ARCHITECTURE_CSV = WORK_DIR / "architecture_comparison.csv"
ROBUSTNESS_ROOT = CACHE_DIR / "robustness_eval"

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("No Kaggle GPU is active. In Kaggle, open Settings and set Accelerator to GPU.")
"""
    ),
    code_cell(
        """
def find_dataset_root():
    candidates = [
        Path("/kaggle/input/pcb-dataset/PCB-DATASET-master"),
        Path("/kaggle/input/datasets/aditya2402/pcb-dataset/PCB-DATASET-master"),
        Path("data/pcb-dataset/PCB-DATASET-master"),
    ]
    for candidate in candidates:
        if (candidate / "images").exists() and (candidate / "Annotations").exists():
            return candidate

    input_root = Path("/kaggle/input")
    if input_root.exists():
        for candidate in input_root.rglob("PCB-DATASET-master"):
            if (candidate / "images").exists() and (candidate / "Annotations").exists():
                return candidate

    raise FileNotFoundError("Could not find PCB-DATASET-master. Attach aditya2402/pcb-dataset to this notebook.")


DATASET_ROOT = find_dataset_root()
print("Dataset root:", DATASET_ROOT)
print("Image folders:", sorted(p.name for p in (DATASET_ROOT / "images").iterdir() if p.is_dir()))
"""
    ),
    code_cell(
        """
def file_md5(path: Path):
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_dspcbsd_yolo_root(extract_root: Path):
    candidates = [extract_root / "Data_YOLO"]
    candidates.extend(p for p in extract_root.rglob("Data_YOLO") if p.is_dir())
    for candidate in candidates:
        if (candidate / "images").exists() and (candidate / "labels").exists():
            return candidate
    return None


def prepare_dspcbsd_plus():
    if not USE_DSPCBSD_PLUS:
        print("DsPCBSD+ merge disabled.")
        return None

    cache_root = CACHE_DIR / "DsPCBSD_plus"
    cache_root.mkdir(parents=True, exist_ok=True)
    yolo_root = find_dspcbsd_yolo_root(cache_root)
    if yolo_root is not None:
        print("DsPCBSD+ already extracted:", yolo_root)
        return yolo_root

    zip_path = cache_root / "DsPCBSD_plus.zip"
    if zip_path.exists() and file_md5(zip_path) != DSPCBSD_PLUS_MD5:
        print("Existing DsPCBSD+ zip checksum mismatch; downloading a fresh copy.")
        zip_path.unlink()

    if not zip_path.exists():
        print(f"Downloading DsPCBSD+ from Figshare DOI {DSPCBSD_PLUS_DOI} ...")
        urllib.request.urlretrieve(DSPCBSD_PLUS_URL, zip_path)

    md5 = file_md5(zip_path)
    if md5 != DSPCBSD_PLUS_MD5:
        raise ValueError(f"DsPCBSD+ checksum mismatch: expected {DSPCBSD_PLUS_MD5}, got {md5}")

    print("DsPCBSD+ zip verified:", md5)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_root)

    yolo_root = find_dspcbsd_yolo_root(cache_root)
    if yolo_root is None:
        raise FileNotFoundError(f"Could not find Data_YOLO inside {cache_root}")
    print("DsPCBSD+ YOLO root:", yolo_root)
    return yolo_root


def parse_dspcbsd_label_file(label_path: Path):
    labels = []
    skipped_boxes = Counter()
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            skipped_boxes["malformed"] += 1
            continue

        src_id = int(float(parts[0]))
        if src_id < 0 or src_id >= len(DSPCBSD_YOLO_CODES):
            skipped_boxes["unknown_id"] += 1
            continue

        src_code = DSPCBSD_YOLO_CODES[src_id]
        target_class = DSPCBSD_TO_PROJECT_CLASS.get(src_code)
        if target_class is None:
            skipped_boxes[src_code] += 1
            continue

        x, y, w, h = map(float, parts[1:])
        if w > 0 and h > 0:
            labels.append((CLASS_TO_ID[target_class.lower()], x, y, w, h))
        else:
            skipped_boxes["invalid_box"] += 1
    return labels, skipped_boxes


def find_matching_image(image_dir: Path, stem: str):
    for suffix in [".jpg", ".jpeg", ".png", ".bmp"]:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    matches = list(image_dir.glob(f"{stem}.*"))
    return matches[0] if matches else None


def load_dspcbsd_plus_records(yolo_root: Path):
    records_by_group = {}
    skipped_boxes = Counter()
    kept_boxes = Counter()
    missing_images = 0

    for source_split in ["train", "val"]:
        label_dir = yolo_root / "labels" / source_split
        image_dir = yolo_root / "images" / source_split
        if not label_dir.exists():
            continue

        for label_path in sorted(label_dir.glob("*.txt")):
            labels, skipped = parse_dspcbsd_label_file(label_path)
            skipped_boxes.update(skipped)
            if not labels:
                continue

            image_path = find_matching_image(image_dir, label_path.stem)
            if image_path is None:
                missing_images += 1
                continue

            for cls_id, *_ in labels:
                kept_boxes[CLASSES[cls_id]] += 1

            primary_class = CLASSES[labels[0][0]]
            group = f"DsPCBSD_plus_{source_split}_{primary_class}"
            records_by_group.setdefault(group, []).append((image_path, labels, group))

    records = [record for group_records in records_by_group.values() for record in group_records]
    if MAX_DSPCBSD_PLUS_IMAGES is not None and len(records) > MAX_DSPCBSD_PLUS_IMAGES:
        rng = random.Random(SEED)
        rng.shuffle(records)
        records = records[:MAX_DSPCBSD_PLUS_IMAGES]
        records_by_group = {}
        for image_path, labels, group in records:
            records_by_group.setdefault(group, []).append((image_path, labels, group))

    print("DsPCBSD+ images with overlapping project classes:", sum(len(v) for v in records_by_group.values()))
    print("DsPCBSD+ kept boxes by project class:", dict(kept_boxes))
    print("DsPCBSD+ skipped non-project boxes:", dict(skipped_boxes))
    if missing_images:
        print("DsPCBSD+ labels missing matching images:", missing_images)

    return records_by_group
"""
    ),
    code_cell(
        """
def voc_box_to_yolo(box, img_w, img_h):
    xmin = float(box.findtext("xmin"))
    ymin = float(box.findtext("ymin"))
    xmax = float(box.findtext("xmax"))
    ymax = float(box.findtext("ymax"))

    xmin = max(0.0, min(xmin, img_w - 1))
    xmax = max(0.0, min(xmax, img_w - 1))
    ymin = max(0.0, min(ymin, img_h - 1))
    ymax = max(0.0, min(ymax, img_h - 1))

    x_center = ((xmin + xmax) / 2.0) / img_w
    y_center = ((ymin + ymax) / 2.0) / img_h
    width = (xmax - xmin) / img_w
    height = (ymax - ymin) / img_h
    return x_center, y_center, width, height


def parse_annotation(xml_path):
    root = ET.parse(xml_path).getroot()
    filename = root.findtext("filename")
    folder = root.findtext("folder") or xml_path.parent.name
    width = int(root.find("size/width").text)
    height = int(root.find("size/height").text)

    image_path = DATASET_ROOT / "images" / folder / filename
    if not image_path.exists():
        matches = list((DATASET_ROOT / "images").rglob(filename))
        image_path = matches[0] if matches else image_path

    labels = []
    for obj in root.findall("object"):
        raw_name = obj.findtext("name", "").strip().replace(" ", "_").lower()
        if raw_name not in CLASS_TO_ID:
            raise ValueError(f"Unknown class {raw_name!r} in {xml_path}")
        box = obj.find("bndbox")
        x, y, w, h = voc_box_to_yolo(box, width, height)
        if w > 0 and h > 0:
            labels.append((CLASS_TO_ID[raw_name], x, y, w, h))

    return image_path, labels, folder


records_by_folder = {}
for xml_path in sorted((DATASET_ROOT / "Annotations").rglob("*.xml")):
    image_path, labels, folder = parse_annotation(xml_path)
    if image_path.exists() and labels:
        records_by_folder.setdefault(folder, []).append((image_path, labels, folder))

dspcbsd_yolo_root = prepare_dspcbsd_plus()
if dspcbsd_yolo_root is not None:
    dspcbsd_records_by_group = load_dspcbsd_plus_records(dspcbsd_yolo_root)
    for group, records in dspcbsd_records_by_group.items():
        records_by_folder.setdefault(group, []).extend(records)

train_records = []
val_records = []
test_records = []
rng = random.Random(SEED)
for folder, records in sorted(records_by_folder.items()):
    rng.shuffle(records)
    n_val = max(1, int(round(len(records) * VAL_RATIO)))
    n_test = max(1, int(round(len(records) * TEST_RATIO)))
    val_records.extend(records[:n_val])
    test_records.extend(records[n_val:n_val + n_test])
    train_records.extend(records[n_val + n_test:])

rng.shuffle(train_records)
rng.shuffle(val_records)
rng.shuffle(test_records)

print("Train images:", len(train_records))
print("Val images:", len(val_records))
print("Test images:", len(test_records))
print("Total images:", len(train_records) + len(val_records) + len(test_records))

def class_counts(records):
    counts = Counter()
    for _, labels, _ in records:
        for cls_id, *_ in labels:
            counts[CLASSES[cls_id]] += 1
    return dict(counts)

print("Train boxes by class:", class_counts(train_records))
print("Val boxes by class:", class_counts(val_records))
print("Test boxes by class:", class_counts(test_records))
"""
    ),
    code_cell(
        """
def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


for split in ["train", "val", "test"]:
    reset_dir(YOLO_ROOT / split / "images")
    reset_dir(YOLO_ROOT / split / "labels")


def write_split(records, split):
    for image_path, labels, folder in records:
        safe_name = f"{folder}_{image_path.name}"
        dest_image = YOLO_ROOT / split / "images" / safe_name
        dest_label = YOLO_ROOT / split / "labels" / f"{Path(safe_name).stem}.txt"

        shutil.copy2(image_path, dest_image)
        label_lines = [
            f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
            for cls_id, x, y, w, h in labels
        ]
        dest_label.write_text("\\n".join(label_lines) + "\\n")


write_split(train_records, "train")
write_split(val_records, "val")
write_split(test_records, "test")

names_block = "\\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
DATA_YAML.write_text(
    f"path: {YOLO_ROOT}\\n"
    "train: train/images\\n"
    "val: val/images\\n"
    "test: test/images\\n"
    f"names:\\n{names_block}\\n"
)

print(DATA_YAML.read_text())
print("Prepared train images:", len(list((YOLO_ROOT / "train/images").glob("*"))))
print("Prepared val images:", len(list((YOLO_ROOT / "val/images").glob("*"))))
print("Prepared test images:", len(list((YOLO_ROOT / "test/images").glob("*"))))
"""
    ),
    code_cell(
        """
model = YOLO("yolov8n.pt")
train_results = model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=0,
    workers=2,
    project=str(WORK_DIR / "runs/detect"),
    name="train",
    exist_ok=True,
    patience=15,
    seed=SEED,
    optimizer="auto",
    cos_lr=True,
    close_mosaic=10,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    shear=2.0,
    perspective=0.0005,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
)

print("Best weights:", WEIGHTS)
print("Weights exist:", WEIGHTS.exists())
"""
    ),
    code_cell(
        """
results_csv = RUN_DIR / "results.csv"
if results_csv.exists():
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    def plot_series(df_, cols, title):
        plt.figure(figsize=(10, 4))
        ok = False
        for c in cols:
            if c in df_.columns:
                plt.plot(df_[c].values, label=c)
                ok = True
        plt.title(title)
        plt.xlabel("epoch")
        plt.grid(True)
        if ok:
            plt.legend()
        plt.show()

    plot_series(df, ["train/box_loss", "val/box_loss"], "Box Loss")
    plot_series(df, ["train/cls_loss", "val/cls_loss"], "Class Loss")
    plot_series(df, ["train/dfl_loss", "val/dfl_loss"], "DFL Loss")
    plot_series(df, ["metrics/precision(B)", "metrics/recall(B)"], "Precision / Recall")
    plot_series(df, ["metrics/mAP50(B)", "metrics/mAP50-95(B)"], "mAP")

    best_idx = df["metrics/mAP50-95(B)"].idxmax() if "metrics/mAP50-95(B)" in df.columns else len(df) - 1
    print("Best epoch:", best_idx + 1)
    print(df.iloc[best_idx].tail(12))
else:
    print("results.csv not found at:", results_csv)
"""
    ),
    code_cell(
        """
model = YOLO(str(WEIGHTS))

def metrics_to_dict(metrics, split_name):
    box = metrics.box
    row = {
        "split": split_name,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
    }
    for key, value in getattr(metrics, "speed", {}).items():
        row[f"speed_{key}_ms"] = float(value)
    return row


summary_rows = []
architecture_rows = []
for split_name in ["val", "test"]:
    metrics = model.val(data=str(DATA_YAML), imgsz=IMG_SIZE, device=0, split=split_name)
    row = metrics_to_dict(metrics, split_name)
    summary_rows.append(row)
    architecture_rows.append({
        "model": "YOLOv8n",
        "family": "CNN / YOLO baseline",
        **row,
    })
    print(split_name.upper(), row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)
display(summary_df)
print("Saved metric summary:", SUMMARY_CSV)

architecture_df = pd.DataFrame(architecture_rows)
architecture_df.to_csv(ARCHITECTURE_CSV, index=False)
display(architecture_df)
print("Saved architecture comparison:", ARCHITECTURE_CSV)
"""
    ),
    code_cell(
        """
def read_yolo_file(label_path: Path):
    boxes = []
    class_labels = []
    if not label_path.exists():
        return boxes, class_labels

    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
        if w > 0 and h > 0:
            boxes.append([x, y, w, h])
            class_labels.append(cls_id)
    return boxes, class_labels


def sanitize_yolo_boxes_for_albumentations(boxes, class_labels, eps=1e-3):
    # Clamp YOLO xywh so corners stay strictly inside [0, 1] (strict Albumentations bbox checks).
    out_boxes = []
    out_labels = []
    for box, cls_id in zip(boxes, class_labels):
        x, y, w, h = (float(v) for v in box)
        w = max(eps, min(w, 1.0 - 2 * eps))
        h = max(eps, min(h, 1.0 - 2 * eps))
        x = min(max(x, w / 2 + eps), 1.0 - w / 2 - eps)
        y = min(max(y, h / 2 + eps), 1.0 - h / 2 - eps)
        out_boxes.append([x, y, w, h])
        out_labels.append(cls_id)
    return out_boxes, out_labels


def write_yolo_file(label_path: Path, boxes, class_labels):
    lines = []
    for cls_id, box in zip(class_labels, boxes):
        x, y, w, h = [max(0.0, min(1.0, float(v))) for v in box]
        if w > 0 and h > 0:
            lines.append(f"{int(cls_id)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
    label_path.write_text("\\n".join(lines) + ("\\n" if lines else ""))


def albumentations_bbox_params():
    return A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.25,
    )


def make_robustness_transforms():
    return {
        "clean_subset": A.Compose([], bbox_params=albumentations_bbox_params()),
        "brightness_contrast": A.Compose(
            [A.RandomBrightnessContrast(brightness_limit=0.30, contrast_limit=0.25, p=1.0)],
            bbox_params=albumentations_bbox_params(),
        ),
        "gaussian_noise": A.Compose(
            [A.GaussNoise(p=1.0)],
            bbox_params=albumentations_bbox_params(),
        ),
        "motion_blur": A.Compose(
            [A.MotionBlur(blur_limit=5, p=1.0)],
            bbox_params=albumentations_bbox_params(),
        ),
        "rotate_10deg": A.Compose(
            [A.Rotate(limit=(10, 10), border_mode=cv2.BORDER_CONSTANT, p=1.0)],
            bbox_params=albumentations_bbox_params(),
        ),
    }


def write_data_yaml(root: Path, yaml_path: Path):
    names_block = "\\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    yaml_path.write_text(
        f"path: {root}\\n"
        "train: test/images\\n"
        "val: test/images\\n"
        "test: test/images\\n"
        f"names:\\n{names_block}\\n"
    )


def build_robustness_dataset(condition_name, transform, image_paths):
    condition_root = ROBUSTNESS_ROOT / condition_name
    image_out = condition_root / "test/images"
    label_out = condition_root / "test/labels"
    if condition_root.exists():
        shutil.rmtree(condition_root)
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    kept_images = 0
    kept_boxes = 0
    for image_path in image_paths:
        label_path = TEST_LBL_DIR / f"{image_path.stem}.txt"
        boxes, class_labels = read_yolo_file(label_path)
        boxes, class_labels = sanitize_yolo_boxes_for_albumentations(boxes, class_labels)
        if not boxes:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        augmented = transform(image=image, bboxes=boxes, class_labels=class_labels)
        aug_boxes = [list(map(float, b)) for b in augmented["bboxes"]]
        aug_labels = list(augmented["class_labels"])
        aug_boxes, aug_labels = sanitize_yolo_boxes_for_albumentations(aug_boxes, aug_labels)
        if not aug_boxes:
            continue

        out_image_path = image_out / image_path.name
        out_label_path = label_out / f"{image_path.stem}.txt"
        cv2.imwrite(str(out_image_path), augmented["image"])
        write_yolo_file(out_label_path, aug_boxes, aug_labels)
        kept_images += 1
        kept_boxes += len(aug_boxes)

    yaml_path = condition_root / "data.yaml"
    write_data_yaml(condition_root, yaml_path)
    return yaml_path, kept_images, kept_boxes


def run_robustness_tests():
    if not RUN_ROBUSTNESS_TESTS:
        print("Robustness tests disabled. Set RUN_ROBUSTNESS_TESTS=True to enable them.")
        return pd.DataFrame()

    rng = random.Random(SEED)
    image_paths = sorted(TEST_IMG_DIR.glob("*.*"))
    rng.shuffle(image_paths)
    image_paths = image_paths[: min(ROBUSTNESS_SAMPLE_SIZE, len(image_paths))]
    print("Robustness sample images:", len(image_paths))

    rows = []
    for condition_name, transform in make_robustness_transforms().items():
        yaml_path, kept_images, kept_boxes = build_robustness_dataset(condition_name, transform, image_paths)
        if kept_images == 0:
            print(f"Skipping {condition_name}: no transformed images with visible boxes.")
            continue
        metrics = model.val(data=str(yaml_path), imgsz=IMG_SIZE, device=0, split="test")
        row = metrics_to_dict(metrics, condition_name)
        row.update({"condition": condition_name, "images": kept_images, "boxes": kept_boxes})
        rows.append(row)
        print("ROBUSTNESS", row)

    robustness_df = pd.DataFrame(rows)
    robustness_df.to_csv(ROBUSTNESS_CSV, index=False)
    display(robustness_df)
    print("Saved robustness metrics:", ROBUSTNESS_CSV)
    return robustness_df


robustness_df = run_robustness_tests()
"""
    ),
    code_cell(
        """
if RUN_RTDETR_BENCHMARK:
    from ultralytics import RTDETR

    rtdetr_model = RTDETR("rtdetr-l.pt")
    rtdetr_results = rtdetr_model.train(
        data=str(DATA_YAML),
        epochs=RTDETR_EPOCHS,
        imgsz=RTDETR_IMG_SIZE,
        batch=RTDETR_BATCH,
        device=0,
        workers=2,
        project=str(WORK_DIR / "runs/rtdetr"),
        name="train",
        exist_ok=True,
        patience=10,
        seed=SEED,
    )
    rtdetr_best = WORK_DIR / "runs/rtdetr/train/weights/best.pt"
    rtdetr_model = RTDETR(str(rtdetr_best))
    rtdetr_rows = []
    for split_name in ["val", "test"]:
        rtdetr_metrics = rtdetr_model.val(data=str(DATA_YAML), imgsz=RTDETR_IMG_SIZE, device=0, split=split_name)
        rtdetr_rows.append({
            "model": "RT-DETR-L",
            "family": "Transformer-style detector",
            **metrics_to_dict(rtdetr_metrics, split_name),
        })
        print("RT-DETR", split_name.upper(), rtdetr_rows[-1])
    architecture_df = pd.read_csv(ARCHITECTURE_CSV) if ARCHITECTURE_CSV.exists() else pd.DataFrame()
    architecture_df = pd.concat([architecture_df, pd.DataFrame(rtdetr_rows)], ignore_index=True)
    architecture_df.to_csv(ARCHITECTURE_CSV, index=False)
    display(architecture_df)
    print("Saved architecture comparison:", ARCHITECTURE_CSV)
else:
    print("RT-DETR transformer benchmark is disabled. Set RUN_RTDETR_BENCHMARK=True for the architecture comparison phase.")
"""
    ),
    code_cell(
        """
def read_yolo_label(txt_path: Path):
    if not txt_path.exists():
        return []
    lines = txt_path.read_text().strip().splitlines()
    out = []
    for ln in lines:
        parts = ln.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        xc, yc, w, h = map(float, parts[1:])
        out.append((cls_id, xc, yc, w, h))
    return out


def yolo_to_xyxy(xc, yc, w, h, W, H):
    x1 = int((xc - w / 2) * W)
    y1 = int((yc - h / 2) * H)
    x2 = int((xc + w / 2) * W)
    y2 = int((yc + h / 2) * H)
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W - 1, x2))
    y2 = max(0, min(H - 1, y2))
    return x1, y1, x2, y2


def draw_boxes(img_bgr, boxes, color=(0, 255, 0), prefix=""):
    img = img_bgr.copy()
    for b in boxes:
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        text = prefix + b["text"]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img
"""
    ),
    code_cell(
        """
def predict_one(img_path: Path, conf=0.01, iou=0.7, imgsz=640):
    return model.predict(source=str(img_path), conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]


def get_pred_boxes(result):
    boxes = []
    if result.boxes is None or len(result.boxes) == 0:
        return boxes
    xyxy = result.boxes.xyxy.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()
    for (x1, y1, x2, y2), c, p in zip(xyxy, cls, conf):
        boxes.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "text": f"{CLASSES[c]} {p:.2f}",
        })
    return boxes
"""
    ),
    code_cell(
        """
def show_side_by_side(img_path: Path, conf=0.01, iou=0.7):
    img = cv2.imread(str(img_path))
    if img is None:
        print("Could not read image:", img_path)
        return
    H, W = img.shape[:2]

    lbl_path = VAL_LBL_DIR / (img_path.stem + ".txt")
    gt_boxes = []
    for cls_id, xc, yc, w, h in read_yolo_label(lbl_path):
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, w, h, W, H)
        gt_boxes.append({"x1": x1, "y1": y1, "x2": x2, "y2": y2, "text": CLASSES[cls_id]})

    pred_boxes = get_pred_boxes(predict_one(img_path, conf=conf, iou=iou))

    left = draw_boxes(img, gt_boxes, color=(0, 255, 0), prefix="GT:")
    right = draw_boxes(img, pred_boxes, color=(0, 0, 255), prefix="PRED:")

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.axis("off")
    plt.imshow(cv2.cvtColor(left, cv2.COLOR_BGR2RGB))
    plt.subplot(1, 2, 2)
    plt.title(f"Prediction (conf={conf})")
    plt.axis("off")
    plt.imshow(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
    plt.show()


val_imgs = sorted(VAL_IMG_DIR.glob("*.*"))
print("Validation images:", len(val_imgs))
for p in random.sample(val_imgs, min(8, len(val_imgs))):
    show_side_by_side(p, conf=0.01, iou=0.7)
"""
    ),
    code_cell(
        """
def count_preds_on_subset(conf, n=50):
    subset = random.sample(val_imgs, min(n, len(val_imgs)))
    total = 0
    for p in subset:
        r = predict_one(p, conf=conf)
        total += 0 if r.boxes is None else len(r.boxes)
    return total


if val_imgs:
    print("Pred count on validation subset at different confidence thresholds:")
    for conf in [0.25, 0.10, 0.05, 0.01, 0.005, 0.001]:
        print(f"  conf={conf:<6} -> total preds: {count_preds_on_subset(conf, n=50)}")
"""
    ),
    code_cell(
        """
def measure_latency(predict_model, image_paths, imgsz, n=80, warmup=10, conf=0.25):
    sample = image_paths[:]
    random.shuffle(sample)
    sample = sample[: min(n, len(sample))]
    if not sample:
        return {}

    for p in sample[: min(warmup, len(sample))]:
        _ = predict_model.predict(source=str(p), imgsz=imgsz, conf=conf, device=0, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    timings = []
    for p in sample:
        start = time.perf_counter()
        _ = predict_model.predict(source=str(p), imgsz=imgsz, conf=conf, device=0, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)

    return {
        "images": len(sample),
        "mean_ms": float(np.mean(timings)),
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "fps": float(1000.0 / np.mean(timings)),
        "under_100ms_target": bool(np.mean(timings) < 100.0),
    }


test_paths = sorted(TEST_IMG_DIR.glob("*.*"))
latency_summary = {
    "yolov8n": measure_latency(model, test_paths, IMG_SIZE, n=80, warmup=10, conf=0.25),
}
if RUN_RTDETR_BENCHMARK:
    rtdetr_m = globals().get("rtdetr_model")
    if rtdetr_m is not None:
        latency_summary["rtdetr_l"] = measure_latency(
            rtdetr_m, test_paths, RTDETR_IMG_SIZE, n=80, warmup=10, conf=0.25
        )

print("Latency summary:", latency_summary)
(WORK_DIR / "latency_summary.json").write_text(json.dumps(latency_summary, indent=2))
"""
    ),
    code_cell(
        """
exported = {}
try:
    onnx_path = model.export(format="onnx", imgsz=IMG_SIZE, opset=12, dynamic=True, simplify=True)
    exported["onnx"] = str(onnx_path)
    print("Exported ONNX:", onnx_path)
except Exception as exc:
    exported["onnx_error"] = repr(exc)
    print("ONNX export failed:", exc)

if RUN_TENSORRT_EXPORT:
    try:
        engine_path = model.export(format="engine", imgsz=IMG_SIZE, half=True, dynamic=False, device=0)
        exported["tensorrt_engine"] = str(engine_path)
        print("Exported TensorRT engine:", engine_path)
    except Exception as exc:
        exported["tensorrt_error"] = repr(exc)
        print("TensorRT export failed:", exc)
else:
    exported["tensorrt_note"] = "Disabled by default. Jetson TensorRT engines should be built on the Jetson target for final deployment."

(WORK_DIR / "deployment_exports.json").write_text(json.dumps(exported, indent=2))
print(exported)
"""
    ),
    code_cell(
        """
PRED_SAVE_DIR.mkdir(parents=True, exist_ok=True)
predict_results = model.predict(
    source=str(VAL_IMG_DIR),
    conf=0.01,
    iou=0.7,
    imgsz=IMG_SIZE,
    save=True,
    project=str(WORK_DIR / "vis_predictions"),
    name="val_preds",
    exist_ok=True,
    verbose=False,
)
print("Saved prediction images to:", PRED_SAVE_DIR)
"""
    ),
    code_cell(
        """
imgs = sorted(PRED_SAVE_DIR.glob("*.jpg"))[:36]
if not imgs:
    print("No .jpg files found in:", PRED_SAVE_DIR)
else:
    gallery_dir = WORK_DIR / "_gallery"
    gallery_dir.mkdir(exist_ok=True)

    for p in imgs:
        outp = gallery_dir / p.name
        if not outp.exists():
            img = cv2.imread(str(p))
            if img is not None:
                cv2.imwrite(str(outp), img)

    html = "<div style='display:flex;flex-wrap:wrap;gap:10px'>"
    for p in imgs:
        rel = f"_gallery/{p.name}"
        html += f"<a href='{rel}' target='_blank'><img src='{rel}' style='width:240px;border:1px solid #ddd'></a>"
    html += "</div>"
    display(HTML(html))
"""
    ),
    code_cell(
        """
traceability = pd.DataFrame([
    ["Detect and localize six PCB defect classes", "YOLO dataset conversion, DsPCBSD+ overlap merge, YOLOv8 training, prediction gallery"],
    ["Improve dataset realism and size", "Adds DsPCBSD+ real PCB images from Figshare DOI 10.6084/m9.figshare.24970329.v1 for overlapping classes"],
    ["Improve small-defect robustness", "Mosaic/MixUp/copy-paste, perspective, HSV, multi-scale-ready YOLO training, plus Albumentations robustness tests"],
    ["Report precision, recall, mAP", "Validation and held-out test metrics saved to project_metrics_summary.csv"],
    ["Evaluate robustness under imaging variation", "robustness_metrics.csv records clean, brightness/contrast, noise, blur, and rotation stress tests"],
    ["Meet real-time target under 100 ms/image", "latency_summary.json records mean, p50, p95 latency per model (YOLOv8n; RT-DETR-L when benchmark runs) and 100 ms pass/fail"],
    ["Provide deployment artifact", "ONNX export plus optional TensorRT export path"],
    ["Compare CNN and Transformer-style detectors", "architecture_comparison.csv records YOLOv8n and RT-DETR-L on val and test splits"],
    ["Provide interpretable visual output", "Ground-truth vs prediction plots and saved prediction gallery"],
])
traceability.columns = ["Proposal requirement", "Notebook implementation"]
display(traceability)
traceability.to_csv(WORK_DIR / "requirements_traceability.csv", index=False)
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.12.12",
        },
        "kaggle": {
            "accelerator": "gpu",
            "dataSources": [
                {
                    "sourceType": "datasetVersion",
                    "sourceId": 15003040,
                    "datasetId": 9603411,
                    "databundleVersionId": 15878469,
                }
            ],
            "dockerImageVersionId": 31286,
            "isInternetEnabled": True,
            "isGpuEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

repo_root = Path(__file__).resolve().parent.parent
text = json.dumps(notebook, indent=1)
for rel in ("kaggle_kernel/project.ipynb", "project.ipynb"):
    out = repo_root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"Wrote {out}")
