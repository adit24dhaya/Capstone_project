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

This notebook is configured for Kaggle GPU. It prepares the PCB dataset, converts annotations to YOLO labels, applies synthetic defect augmentation, trains a YOLOv8 detector, runs Albumentations-based robustness tests, benchmarks a transformer-style RT-DETR detector, evaluates YOLO-Transformer hybrid fusion, trains a custom CNN-Transformer feature refiner, measures inference latency, and exports deployment artifacts.
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
RUN_SYNTHETIC_DEFECT_AUGMENTATION = True
SYNTHETIC_TARGETS_PER_CLASS = {
    "Missing_hole": 320,
    "Mouse_bite": 240,
    "Spur": 240,
}
SYNTHETIC_PATCH_CONTEXT_RANGE = (1.4, 2.4)
RUN_HYBRID_FUSION = True
HYBRID_CONF = 0.01
HYBRID_FUSION_IOU = 0.55
HYBRID_EVAL_SAMPLE_SIZE = None
RUN_HYBRID_TUNING = True
HYBRID_TUNING_CONF_VALUES = [0.15, 0.20, 0.25, 0.30]
HYBRID_TUNING_IOU_VALUES = [0.40, 0.45, 0.55]
HYBRID_TUNING_MODES = ["agreement_only", "weighted_fusion", "single_high_conf_fallback", "class_weighted_fusion"]
HYBRID_TUNING_NMS_VALUES = [0.45, 0.60]
HYBRID_TUNING_SINGLE_MODEL_CONF_VALUES = [0.50, 0.60, 0.70]
HYBRID_TUNING_PER_CLASS_PROFILES = ["uniform", "precision_boost", "aggressive_precision"]
HYBRID_TUNING_PRECISION_FLOORS = [0.65, 0.70, 0.75]
HYBRID_MIN_PRECISION_FOR_SELECTION = 0.65
HYBRID_MIN_RECALL_FOR_SELECTION = 0.78
HYBRID_MIN_MAP50_FOR_SELECTION = 0.82
HYBRID_TUNING_SAMPLE_SIZE = 350
HYBRID_VISUAL_SAMPLE_SIZE = 12
RUN_CNN_TRANSFORMER_REFINER = True
REFINER_EPOCHS = 5
REFINER_BATCH = 64
REFINER_PATCH_SIZE = 96
REFINER_MAX_POSITIVE_PER_CLASS = 650
REFINER_NEGATIVE_SAMPLES = 1600
REFINER_KEEP_PROB = 0.45
REFINER_CANDIDATE_CONF = 0.03

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
PER_CLASS_CSV = WORK_DIR / "per_class_metrics.csv"
LATENCY_TABLE_CSV = WORK_DIR / "latency_comparison.csv"
FINAL_SUMMARY_CSV = WORK_DIR / "final_results_summary.csv"
HYBRID_FUSION_CSV = WORK_DIR / "hybrid_fusion_metrics.csv"
HYBRID_PER_CLASS_CSV = WORK_DIR / "hybrid_per_class_metrics.csv"
HYBRID_TUNING_GRID_CSV = WORK_DIR / "hybrid_tuning_grid.csv"
HYBRID_SELECTED_CONFIG_JSON = WORK_DIR / "hybrid_selected_config.json"
HYBRID_SELECTED_TEST_CSV = WORK_DIR / "hybrid_selected_test_metrics.csv"
HYBRID_SELECTED_PER_CLASS_CSV = WORK_DIR / "hybrid_selected_per_class_metrics.csv"
HYBRID_TUNING_GRID_V2_CSV = WORK_DIR / "hybrid_tuning_grid_v2.csv"
HYBRID_SELECTED_CONFIG_V2_JSON = WORK_DIR / "hybrid_selected_config_v2.json"
HYBRID_SELECTED_TEST_V2_CSV = WORK_DIR / "hybrid_selected_test_metrics_v2.csv"
HYBRID_ERROR_ANALYSIS_V2_CSV = WORK_DIR / "hybrid_error_analysis_v2.csv"
HYBRID_ROBUSTNESS_CSV = WORK_DIR / "hybrid_robustness_metrics.csv"
HYBRID_ROBUSTNESS_PER_CLASS_CSV = WORK_DIR / "hybrid_robustness_per_class_metrics.csv"
HYBRID_ERROR_ANALYSIS_CSV = WORK_DIR / "hybrid_error_analysis.csv"
HYBRID_ERROR_EXAMPLES_CSV = WORK_DIR / "hybrid_error_examples.csv"
HYBRID_CLASS_DELTA_CSV = WORK_DIR / "hybrid_class_delta.csv"
HYBRID_VIS_DIR = WORK_DIR / "hybrid_visual_evidence"
SYNTHETIC_AUGMENTATION_CSV = WORK_DIR / "synthetic_augmentation_summary.csv"
REFINER_TRAINING_CSV = WORK_DIR / "cnn_transformer_refiner_training.csv"
REFINER_METRICS_CSV = WORK_DIR / "cnn_transformer_refined_hybrid_metrics.csv"
REFINER_PER_CLASS_CSV = WORK_DIR / "cnn_transformer_refined_hybrid_per_class.csv"
JETSON_DEPLOYMENT_STATUS_JSON = WORK_DIR / "jetson_deployment_status.json"

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

def read_train_yolo_file(label_path: Path):
    rows = []
    if not label_path.exists():
        return rows
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(float(parts[0]))
        x, y, w, h = map(float, parts[1:])
        if 0 <= cls_id < len(CLASSES) and w > 0 and h > 0:
            rows.append((cls_id, x, y, w, h))
    return rows


def yolo_norm_to_xyxy_pixels(box, width, height):
    _, x, y, w, h = box
    x1 = int(max(0, (x - w / 2.0) * width))
    y1 = int(max(0, (y - h / 2.0) * height))
    x2 = int(min(width - 1, (x + w / 2.0) * width))
    y2 = int(min(height - 1, (y + h / 2.0) * height))
    return x1, y1, x2, y2


def xyxy_pixels_to_yolo_norm(cls_id, xyxy, width, height):
    x1, y1, x2, y2 = xyxy
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    xc = (x1 + bw / 2.0) / width
    yc = (y1 + bh / 2.0) / height
    return cls_id, xc, yc, bw / width, bh / height


def crop_defect_patch(image, label_row, context_scale):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = yolo_norm_to_xyxy_pixels(label_row, width, height)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    crop_w = max(bw + 4, int(bw * context_scale))
    crop_h = max(bh + 4, int(bh * context_scale))
    px1 = int(max(0, cx - crop_w / 2.0))
    py1 = int(max(0, cy - crop_h / 2.0))
    px2 = int(min(width, cx + crop_w / 2.0))
    py2 = int(min(height, cy + crop_h / 2.0))
    if px2 <= px1 or py2 <= py1:
        return None, None
    patch = image[py1:py2, px1:px2].copy()
    defect_xyxy_in_patch = [x1 - px1, y1 - py1, x2 - px1, y2 - py1]
    return patch, defect_xyxy_in_patch


def feathered_paste(background, patch, top_left):
    x, y = top_left
    h, w = patch.shape[:2]
    roi = background[y:y + h, x:x + w]
    if roi.shape[:2] != patch.shape[:2]:
        return background

    mask = np.full((h, w), 255, dtype=np.uint8)
    blur = max(3, (min(h, w) // 8) | 1)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0).astype(np.float32) / 255.0
    mask = mask[..., None]
    blended = (patch.astype(np.float32) * mask + roi.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
    background[y:y + h, x:x + w] = blended
    return background


def build_synthetic_augmentation_pool():
    train_img_dir = YOLO_ROOT / "train/images"
    train_lbl_dir = YOLO_ROOT / "train/labels"
    pool = {i: [] for i in range(len(CLASSES))}
    image_paths = sorted(train_img_dir.glob("*.*"))
    for image_path in image_paths:
        label_rows = read_train_yolo_file(train_lbl_dir / f"{image_path.stem}.txt")
        for row in label_rows:
            pool[row[0]].append((image_path, row))
    return pool, image_paths


def make_synthetic_defect_images():
    if not RUN_SYNTHETIC_DEFECT_AUGMENTATION:
        print("Synthetic defect augmentation disabled.")
        return pd.DataFrame()

    rng = random.Random(SEED)
    pool, background_paths = build_synthetic_augmentation_pool()
    train_img_dir = YOLO_ROOT / "train/images"
    train_lbl_dir = YOLO_ROOT / "train/labels"
    summary_rows = []

    try:
        synthetic_noise = A.GaussNoise(std_range=(0.01, 0.04), mean_range=(0.0, 0.0), p=0.35)
    except TypeError:
        synthetic_noise = A.GaussNoise(var_limit=(2.0, 12.0), p=0.35)

    transform = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.8),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=15, p=0.4),
        synthetic_noise,
        A.MotionBlur(blur_limit=3, p=0.2),
    ])

    for class_name, target_count in SYNTHETIC_TARGETS_PER_CLASS.items():
        cls_id = CLASS_TO_ID[class_name.lower()]
        generated = 0
        attempts = 0
        if not pool.get(cls_id):
            summary_rows.append({"class_name": class_name, "requested": target_count, "generated": 0, "note": "no source defects"})
            continue

        while generated < target_count and attempts < target_count * 20:
            attempts += 1
            source_image_path, source_label = rng.choice(pool[cls_id])
            background_path = rng.choice(background_paths)
            source = cv2.imread(str(source_image_path))
            background = cv2.imread(str(background_path))
            if source is None or background is None:
                continue

            context = rng.uniform(*SYNTHETIC_PATCH_CONTEXT_RANGE)
            patch, patch_box = crop_defect_patch(source, source_label, context)
            if patch is None or min(patch.shape[:2]) < 6:
                continue

            patch = transform(image=patch)["image"]
            orig_patch_h, orig_patch_w = patch.shape[:2]
            scale = rng.uniform(0.75, 1.35)
            new_w = max(4, int(patch.shape[1] * scale))
            new_h = max(4, int(patch.shape[0] * scale))
            if new_w >= background.shape[1] or new_h >= background.shape[0]:
                continue
            patch = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

            x = rng.randint(0, background.shape[1] - new_w)
            y = rng.randint(0, background.shape[0] - new_h)
            background = feathered_paste(background, patch, (x, y))

            # Use the source defect's relative location inside its crop after resizing.
            rel_x1 = max(0.0, min(1.0, patch_box[0] / max(1, orig_patch_w)))
            rel_y1 = max(0.0, min(1.0, patch_box[1] / max(1, orig_patch_h)))
            rel_x2 = max(0.0, min(1.0, patch_box[2] / max(1, orig_patch_w)))
            rel_y2 = max(0.0, min(1.0, patch_box[3] / max(1, orig_patch_h)))
            defect_xyxy = [
                x + rel_x1 * new_w,
                y + rel_y1 * new_h,
                x + rel_x2 * new_w,
                y + rel_y2 * new_h,
            ]
            if defect_xyxy[2] <= defect_xyxy[0] + 1 or defect_xyxy[3] <= defect_xyxy[1] + 1:
                continue

            base_labels = read_train_yolo_file(train_lbl_dir / f"{background_path.stem}.txt")
            synthetic_label = xyxy_pixels_to_yolo_norm(cls_id, defect_xyxy, background.shape[1], background.shape[0])
            output_stem = f"synthetic_{class_name}_{generated:04d}_{Path(background_path).stem}"
            cv2.imwrite(str(train_img_dir / f"{output_stem}.jpg"), background)
            labels = base_labels + [synthetic_label]
            label_lines = [f"{int(c)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for c, x, y, w, h in labels]
            (train_lbl_dir / f"{output_stem}.txt").write_text("\\n".join(label_lines) + "\\n")
            generated += 1

        summary_rows.append({
            "class_name": class_name,
            "requested": target_count,
            "generated": generated,
            "source_instances": len(pool.get(cls_id, [])),
            "note": "copy-paste synthetic augmentation; TransGAN-style class-balancing surrogate",
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SYNTHETIC_AUGMENTATION_CSV, index=False)
    display(summary_df)
    print("Saved synthetic augmentation summary:", SYNTHETIC_AUGMENTATION_CSV)
    return summary_df


synthetic_augmentation_df = make_synthetic_defect_images()

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


def per_class_metrics_to_rows(metrics, model_name, split_name):
    rows = []
    box = getattr(metrics, "box", None)
    if box is None:
        return rows
    class_indices = list(getattr(box, "ap_class_index", []))
    if not class_indices:
        class_indices = list(range(len(CLASSES)))
    for i, cls_id in enumerate(class_indices):
        try:
            p, r, ap50, ap = box.class_result(i)
        except Exception:
            continue
        cls_id = int(cls_id)
        cls_name = CLASSES[cls_id] if 0 <= cls_id < len(CLASSES) else str(cls_id)
        rows.append({
            "model": model_name,
            "split": split_name,
            "class_id": cls_id,
            "class_name": cls_name,
            "precision": float(p),
            "recall": float(r),
            "mAP50": float(ap50),
            "mAP50_95": float(ap),
        })
    return rows


summary_rows = []
architecture_rows = []
per_class_rows = []
for split_name in ["val", "test"]:
    metrics = model.val(data=str(DATA_YAML), imgsz=IMG_SIZE, device=0, split=split_name)
    row = metrics_to_dict(metrics, split_name)
    summary_rows.append(row)
    architecture_rows.append({
        "model": "YOLOv8n",
        "family": "CNN / YOLO baseline",
        **row,
    })
    per_class_rows.extend(per_class_metrics_to_rows(metrics, "YOLOv8n", split_name))
    print(split_name.upper(), row)


summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)
display(summary_df)
print("Saved metric summary:", SUMMARY_CSV)

architecture_df = pd.DataFrame(architecture_rows)
architecture_df.to_csv(ARCHITECTURE_CSV, index=False)
display(architecture_df)
print("Saved architecture comparison:", ARCHITECTURE_CSV)

per_class_df = pd.DataFrame(per_class_rows)
per_class_df.to_csv(PER_CLASS_CSV, index=False)
display(per_class_df)
print("Saved per-class metrics:", PER_CLASS_CSV)
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
    # Albumentations 2.x uses std_range/mean_range, while 1.x uses var_limit.
    try:
        gaussian_noise = A.GaussNoise(std_range=(0.02, 0.08), mean_range=(0.0, 0.0), p=1.0)
    except TypeError:
        gaussian_noise = A.GaussNoise(var_limit=(5.0, 25.0), p=1.0)

    return {
        "clean_subset": A.Compose([], bbox_params=albumentations_bbox_params()),
        "brightness_contrast": A.Compose(
            [A.RandomBrightnessContrast(brightness_limit=0.30, contrast_limit=0.25, p=1.0)],
            bbox_params=albumentations_bbox_params(),
        ),
        "gaussian_noise": A.Compose(
            [gaussian_noise],
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
        per_class_rows = per_class_metrics_to_rows(rtdetr_metrics, "RT-DETR-L", split_name)
        if per_class_rows:
            per_class_df = pd.read_csv(PER_CLASS_CSV) if PER_CLASS_CSV.exists() else pd.DataFrame()
            per_class_df = pd.concat([per_class_df, pd.DataFrame(per_class_rows)], ignore_index=True)
            per_class_df.to_csv(PER_CLASS_CSV, index=False)
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
def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def nms_class_aware(boxes, iou_thr=0.6):
    kept = []
    by_class = {}
    for b in boxes:
        by_class.setdefault(int(b["cls"]), []).append(b)
    for cls_id, cls_boxes in by_class.items():
        cls_boxes = sorted(cls_boxes, key=lambda x: x["conf"], reverse=True)
        while cls_boxes:
            top = cls_boxes.pop(0)
            kept.append(top)
            cls_boxes = [b for b in cls_boxes if iou_xyxy(top["xyxy"], b["xyxy"]) < iou_thr]
    return kept


def predict_xyxy(det_model, image_path: Path, imgsz, conf_thr):
    result = det_model.predict(source=str(image_path), conf=conf_thr, iou=0.7, imgsz=imgsz, device=0, verbose=False)[0]
    out = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    cls = result.boxes.cls.detach().cpu().numpy().astype(int)
    conf = result.boxes.conf.detach().cpu().numpy()
    for bb, c, p in zip(xyxy, cls, conf):
        out.append({"cls": int(c), "conf": float(p), "xyxy": [float(v) for v in bb]})
    return out


def default_hybrid_config():
    return {
        "conf": HYBRID_CONF,
        "fusion_iou": HYBRID_FUSION_IOU,
        "mode": "weighted_fusion",
        "nms_iou": 0.60,
        "single_model_conf": 0.60,
        "per_class_profile": "uniform",
        "per_class_conf": {},
        "model_class_weights": {},
    }


def load_validation_per_class_metrics():
    if not PER_CLASS_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(PER_CLASS_CSV)
    except Exception:
        return pd.DataFrame()
    if "split" not in df.columns:
        return pd.DataFrame()
    return df[df["split"] == "val"].copy()


def build_model_class_weights(val_per_class_df):
    weights = {
        "yolo": {str(i): 1.0 for i in range(len(CLASSES))},
        "rtdetr": {str(i): 1.0 for i in range(len(CLASSES))},
    }
    if val_per_class_df.empty:
        return weights

    model_map = {
        "YOLOv8n": "yolo",
        "RT-DETR-L": "rtdetr",
    }
    for model_name, key in model_map.items():
        rows = val_per_class_df[val_per_class_df["model"] == model_name]
        for _, row in rows.iterrows():
            cls_id = int(row["class_id"])
            score = float(row.get("mAP50_95", row.get("mAP50", 0.5)))
            weights[key][str(cls_id)] = float(np.clip(0.35 + score, 0.50, 1.35))
    return weights


def build_per_class_conf_profile(base_conf, profile_name, val_per_class_df):
    if profile_name == "uniform" or val_per_class_df.empty:
        return {str(i): float(base_conf) for i in range(len(CLASSES))}

    thresholds = {}
    for cls_id in range(len(CLASSES)):
        cls_rows = val_per_class_df[val_per_class_df["class_id"] == cls_id]
        if cls_rows.empty:
            thresholds[str(cls_id)] = float(base_conf)
            continue
        best_precision = float(cls_rows["precision"].max()) if "precision" in cls_rows else 0.65
        best_recall = float(cls_rows["recall"].max()) if "recall" in cls_rows else 0.65
        best_map = float(cls_rows["mAP50_95"].max()) if "mAP50_95" in cls_rows else 0.45
        false_positive_pressure = max(0.0, 0.82 - best_precision)
        weak_localization_pressure = max(0.0, 0.48 - best_map)
        recall_guard = -0.05 if best_recall < 0.70 else 0.0
        intensity = 0.20 if profile_name == "precision_boost" else 0.32
        threshold = base_conf + intensity * false_positive_pressure + 0.35 * weak_localization_pressure + recall_guard
        thresholds[str(cls_id)] = float(np.clip(threshold, base_conf, 0.85))
    return thresholds


def get_class_conf_threshold(config, cls_id):
    base_conf = float(config.get("conf", HYBRID_CONF))
    per_class_conf = config.get("per_class_conf") or {}
    return max(base_conf, float(per_class_conf.get(str(int(cls_id)), base_conf)))


def get_model_class_weight(config, model_key, cls_id):
    model_weights = config.get("model_class_weights") or {}
    return float(model_weights.get(model_key, {}).get(str(int(cls_id)), 1.0))


def weighted_box_merge(y, r, config=None):
    config = config or {}
    cls_id = int(y["cls"])
    wy = max(float(y["conf"]), 1e-6) * get_model_class_weight(config, "yolo", cls_id)
    wr = max(float(r["conf"]), 1e-6) * get_model_class_weight(config, "rtdetr", cls_id)
    denom = wy + wr
    fused_xyxy = [
        (wy * y["xyxy"][0] + wr * r["xyxy"][0]) / denom,
        (wy * y["xyxy"][1] + wr * r["xyxy"][1]) / denom,
        (wy * y["xyxy"][2] + wr * r["xyxy"][2]) / denom,
        (wy * y["xyxy"][3] + wr * r["xyxy"][3]) / denom,
    ]
    y_score = float(y["conf"]) * get_model_class_weight(config, "yolo", cls_id)
    r_score = float(r["conf"]) * get_model_class_weight(config, "rtdetr", cls_id)
    fused_conf = min(1.0, max(y_score, r_score, (y_score + r_score) / 2.0) + 0.05)
    return {"cls": cls_id, "conf": float(fused_conf), "xyxy": fused_xyxy, "sources": "yolo+rtdetr"}


def fuse_class_boxes_config(yolo_boxes, rtdetr_boxes, config):
    conf_thr = float(config.get("conf", HYBRID_CONF))
    iou_thr = float(config.get("fusion_iou", HYBRID_FUSION_IOU))
    mode = config.get("mode", "weighted_fusion")
    single_model_conf = float(config.get("single_model_conf", max(conf_thr, 0.60)))

    yolo_boxes = [b for b in yolo_boxes if float(b["conf"]) >= conf_thr]
    rtdetr_boxes = [b for b in rtdetr_boxes if float(b["conf"]) >= conf_thr]

    fused = []
    used_r = set()
    for y in sorted(yolo_boxes, key=lambda x: x["conf"], reverse=True):
        best_idx = -1
        best_iou = 0.0
        for idx, r in enumerate(rtdetr_boxes):
            if idx in used_r:
                continue
            ov = iou_xyxy(y["xyxy"], r["xyxy"])
            if ov > best_iou:
                best_iou = ov
                best_idx = idx
        if best_idx >= 0 and best_iou >= iou_thr:
            r = rtdetr_boxes[best_idx]
            used_r.add(best_idx)
            fused.append(weighted_box_merge(y, r, config))
        elif mode in {"weighted_fusion", "union_high_conf", "single_high_conf_fallback", "class_weighted_fusion"}:
            if mode == "weighted_fusion":
                keep_thr = conf_thr
            elif mode == "union_high_conf":
                keep_thr = max(conf_thr, 0.25)
            else:
                keep_thr = max(conf_thr, single_model_conf)
            if float(y["conf"]) >= keep_thr:
                out = dict(y)
                out["sources"] = "yolo"
                fused.append(out)

    if mode in {"weighted_fusion", "union_high_conf", "single_high_conf_fallback", "class_weighted_fusion"}:
        if mode == "weighted_fusion":
            keep_thr = conf_thr
        elif mode == "union_high_conf":
            keep_thr = max(conf_thr, 0.25)
        else:
            keep_thr = max(conf_thr, single_model_conf)
        for idx, r in enumerate(rtdetr_boxes):
            if idx not in used_r and float(r["conf"]) >= keep_thr:
                out = dict(r)
                out["sources"] = "rtdetr"
                fused.append(out)
    return [b for b in fused if float(b["conf"]) >= get_class_conf_threshold(config, b["cls"])]


def hybrid_fuse_predictions_config(yolo_preds, rtdetr_preds, config):
    out = []
    classes = sorted(set([b["cls"] for b in yolo_preds] + [b["cls"] for b in rtdetr_preds]))
    for cls_id in classes:
        y_cls = [b for b in yolo_preds if b["cls"] == cls_id]
        r_cls = [b for b in rtdetr_preds if b["cls"] == cls_id]
        out.extend(fuse_class_boxes_config(y_cls, r_cls, config))
    return nms_class_aware(out, iou_thr=float(config.get("nms_iou", 0.60)))


def hybrid_fuse_predictions(yolo_preds, rtdetr_preds, iou_thr=0.55):
    config = default_hybrid_config()
    config["fusion_iou"] = iou_thr
    return hybrid_fuse_predictions_config(yolo_preds, rtdetr_preds, config)


def ap_from_pr(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate_predictions(gt_by_class, pred_by_class, iou_thresholds=None):
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 0.96, 0.05)

    per_class_rows = []
    for cls_id in range(len(CLASSES)):
        gt_map = gt_by_class.get(cls_id, {})
        n_gt = sum(len(v) for v in gt_map.values())
        cls_preds = sorted(pred_by_class.get(cls_id, []), key=lambda x: x["conf"], reverse=True)
        aps = []
        p50 = 0.0
        r50 = 0.0
        ap50 = 0.0

        for thr in iou_thresholds:
            matched = {img_id: np.zeros(len(gt_map.get(img_id, [])), dtype=bool) for img_id in gt_map}
            tps, fps = [], []
            for pred in cls_preds:
                img_id = pred["image_id"]
                gts = gt_map.get(img_id, [])
                best_iou = 0.0
                best_j = -1
                for j, gt_box in enumerate(gts):
                    ov = iou_xyxy(pred["xyxy"], gt_box)
                    if ov > best_iou:
                        best_iou = ov
                        best_j = j
                if best_iou >= thr and best_j >= 0 and not matched[img_id][best_j]:
                    matched[img_id][best_j] = True
                    tps.append(1.0)
                    fps.append(0.0)
                else:
                    tps.append(0.0)
                    fps.append(1.0)

            if len(tps) == 0:
                aps.append(0.0)
                continue
            tps = np.cumsum(np.array(tps))
            fps = np.cumsum(np.array(fps))
            recalls = tps / max(n_gt, 1)
            precisions = tps / np.maximum(tps + fps, 1e-9)
            ap = ap_from_pr(recalls, precisions)
            aps.append(ap)

            if abs(float(thr) - 0.5) < 1e-9:
                ap50 = ap
                p50 = float(precisions[-1]) if len(precisions) else 0.0
                r50 = float(recalls[-1]) if len(recalls) else 0.0

        per_class_rows.append({
            "class_id": cls_id,
            "class_name": CLASSES[cls_id],
            "precision": p50,
            "recall": r50,
            "mAP50": ap50,
            "mAP50_95": float(np.mean(aps)) if aps else 0.0,
            "gt_boxes": n_gt,
        })
    return per_class_rows


def build_gt_for_image_paths(image_paths, lbl_dir):
    gt_by_class = {i: {} for i in range(len(CLASSES))}
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        label_path = lbl_dir / f"{image_path.stem}.txt"
        label_rows = []
        if label_path.exists():
            for ln in label_path.read_text().strip().splitlines():
                parts = ln.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(float(parts[0]))
                xc, yc, bw, bh = map(float, parts[1:])
                label_rows.append((cls_id, xc, yc, bw, bh))
        for cls_id, xc, yc, bw, bh in label_rows:
            x1 = max(0.0, (xc - bw / 2.0) * w)
            y1 = max(0.0, (yc - bh / 2.0) * h)
            x2 = min(float(w), (xc + bw / 2.0) * w)
            y2 = min(float(h), (yc + bh / 2.0) * h)
            gt_by_class.setdefault(int(cls_id), {}).setdefault(image_path.name, []).append([x1, y1, x2, y2])
    return gt_by_class


def build_gt_for_split(split_name, sample_size=None):
    img_dir = VAL_IMG_DIR if split_name == "val" else TEST_IMG_DIR
    lbl_dir = VAL_LBL_DIR if split_name == "val" else TEST_LBL_DIR
    image_paths = sorted(img_dir.glob("*.*"))
    limit = sample_size if sample_size is not None else HYBRID_EVAL_SAMPLE_SIZE
    if limit is not None:
        rng = random.Random(SEED)
        rng.shuffle(image_paths)
        image_paths = image_paths[: min(limit, len(image_paths))]

    gt_by_class = build_gt_for_image_paths(image_paths, lbl_dir)
    return image_paths, gt_by_class


def collect_hybrid_prediction_cache(image_paths, conf_thr):
    cache = []
    for image_path in image_paths:
        cache.append({
            "image_path": image_path,
            "yolo": predict_xyxy(model, image_path, IMG_SIZE, conf_thr),
            "rtdetr": predict_xyxy(rtdetr_model, image_path, RTDETR_IMG_SIZE, conf_thr),
        })
    return cache


def predictions_from_cache(cache, config):
    pred_by_class = {i: [] for i in range(len(CLASSES))}
    preds_by_image = {}
    for item in cache:
        image_path = item["image_path"]
        fused = hybrid_fuse_predictions_config(item["yolo"], item["rtdetr"], config)
        preds_by_image[image_path.name] = fused
        for pred in fused:
            pred_by_class[int(pred["cls"])].append({
                "image_id": image_path.name,
                "conf": float(pred["conf"]),
                "xyxy": pred["xyxy"],
            })
    return pred_by_class, preds_by_image


def summarize_per_class_rows(per_cls):
    precision = float(np.mean([r["precision"] for r in per_cls])) if per_cls else 0.0
    recall = float(np.mean([r["recall"] for r in per_cls])) if per_cls else 0.0
    mAP50 = float(np.mean([r["mAP50"] for r in per_cls])) if per_cls else 0.0
    mAP50_95 = float(np.mean([r["mAP50_95"] for r in per_cls])) if per_cls else 0.0
    f1 = float(2 * precision * recall / max(precision + recall, 1e-9))
    return precision, recall, mAP50, mAP50_95, f1


def f_beta_score(precision, recall, beta=0.5):
    beta_sq = beta * beta
    return float((1.0 + beta_sq) * precision * recall / max(beta_sq * precision + recall, 1e-9))


def count_detection_errors_at_iou(gt_by_class, pred_by_class, iou_thr=0.50):
    tp = fp = fn = 0
    for cls_id in range(len(CLASSES)):
        gt_map = gt_by_class.get(cls_id, {})
        cls_preds = sorted(pred_by_class.get(cls_id, []), key=lambda x: x["conf"], reverse=True)
        matched = {img_id: np.zeros(len(gt_map.get(img_id, [])), dtype=bool) for img_id in gt_map}
        for pred in cls_preds:
            img_id = pred["image_id"]
            gts = gt_map.get(img_id, [])
            best_iou = 0.0
            best_j = -1
            for j, gt_box in enumerate(gts):
                ov = iou_xyxy(pred["xyxy"], gt_box)
                if ov > best_iou:
                    best_iou = ov
                    best_j = j
            if best_iou >= iou_thr and best_j >= 0 and not matched[img_id][best_j]:
                matched[img_id][best_j] = True
                tp += 1
            else:
                fp += 1
        for flags in matched.values():
            fn += int((~flags).sum())
    return tp, fp, fn


def evaluate_cached_hybrid(cache, gt_by_class, config, split_name, model_name="Hybrid YOLOv8n + RT-DETR-L"):
    pred_by_class, preds_by_image = predictions_from_cache(cache, config)
    per_cls = evaluate_predictions(gt_by_class, pred_by_class)
    precision, recall, mAP50, mAP50_95, f1 = summarize_per_class_rows(per_cls)
    tp, fp, fn = count_detection_errors_at_iou(gt_by_class, pred_by_class)
    row = {
        "model": model_name,
        "family": "YOLO-Transformer tuned late fusion",
        "split": split_name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f0_5": f_beta_score(precision, recall, beta=0.5),
        "mAP50": mAP50,
        "mAP50_95": mAP50_95,
        "tp_iou50": tp,
        "fp_iou50": fp,
        "fn_iou50": fn,
        "false_positives_per_image": float(fp / max(len(cache), 1)),
        "images": len(cache),
        "conf": float(config.get("conf", HYBRID_CONF)),
        "fusion_iou": float(config.get("fusion_iou", HYBRID_FUSION_IOU)),
        "fusion_mode": config.get("mode", "weighted_fusion"),
        "nms_iou": float(config.get("nms_iou", 0.60)),
        "single_model_conf": float(config.get("single_model_conf", np.nan)),
        "per_class_profile": config.get("per_class_profile", "uniform"),
        "model_weight_profile": config.get("model_weight_profile", "equal"),
    }
    return row, per_cls, pred_by_class, preds_by_image


def tune_hybrid_config():
    if not RUN_HYBRID_TUNING:
        config = default_hybrid_config()
        HYBRID_SELECTED_CONFIG_JSON.write_text(json.dumps(config, indent=2))
        HYBRID_SELECTED_CONFIG_V2_JSON.write_text(json.dumps(config, indent=2))
        return config, pd.DataFrame()

    val_paths, gt_by_class = build_gt_for_split("val", sample_size=HYBRID_TUNING_SAMPLE_SIZE)
    min_conf = min(HYBRID_TUNING_CONF_VALUES)
    cache = collect_hybrid_prediction_cache(val_paths, min_conf)
    val_per_class_df = load_validation_per_class_metrics()
    model_class_weights = build_model_class_weights(val_per_class_df)
    rows = []
    for conf in HYBRID_TUNING_CONF_VALUES:
        for fusion_iou in HYBRID_TUNING_IOU_VALUES:
            for mode in HYBRID_TUNING_MODES:
                single_model_values = HYBRID_TUNING_SINGLE_MODEL_CONF_VALUES if mode in {"single_high_conf_fallback", "class_weighted_fusion"} else [0.60]
                for single_model_conf in single_model_values:
                    for profile_name in HYBRID_TUNING_PER_CLASS_PROFILES:
                        per_class_conf = build_per_class_conf_profile(conf, profile_name, val_per_class_df)
                        for nms_iou in HYBRID_TUNING_NMS_VALUES:
                            config = {
                                "conf": float(conf),
                                "fusion_iou": float(fusion_iou),
                                "mode": mode,
                                "nms_iou": float(nms_iou),
                                "single_model_conf": float(single_model_conf),
                                "per_class_profile": profile_name,
                                "per_class_conf": per_class_conf,
                                "model_weight_profile": "val_mAP50_95" if mode == "class_weighted_fusion" else "equal",
                                "model_class_weights": model_class_weights if mode == "class_weighted_fusion" else {},
                            }
                            row, _, _, _ = evaluate_cached_hybrid(cache, gt_by_class, config, "val_tuning")
                            row["per_class_conf_json"] = json.dumps(per_class_conf, sort_keys=True)
                            rows.append(row)
                            print("HYBRID_TUNE_V2", row)

    grid_df = pd.DataFrame(rows)
    grid_df.to_csv(HYBRID_TUNING_GRID_CSV, index=False)
    grid_df.to_csv(HYBRID_TUNING_GRID_V2_CSV, index=False)
    display(grid_df.sort_values(["precision", "f0_5", "mAP50_95"], ascending=False).head(12))
    print("Saved hybrid tuning grid:", HYBRID_TUNING_GRID_CSV)
    print("Saved hybrid tuning grid v2:", HYBRID_TUNING_GRID_V2_CSV)

    selected = None
    selection_floor = None
    selection_metric = ""
    sort_cols = ["f0_5", "mAP50_95", "f1", "precision", "recall", "false_positives_per_image"]
    sort_ascending = [False, False, False, False, False, True]
    for floor in sorted(HYBRID_TUNING_PRECISION_FLOORS, reverse=True):
        candidates = grid_df[
            (grid_df["precision"] >= floor)
            & (grid_df["recall"] >= HYBRID_MIN_RECALL_FOR_SELECTION)
            & (grid_df["mAP50"] >= HYBRID_MIN_MAP50_FOR_SELECTION)
        ]
        if not candidates.empty:
            selected = candidates.sort_values(sort_cols, ascending=sort_ascending).iloc[0]
            selection_floor = floor
            selection_metric = "f0_5_with_precision_recall_map_floor"
            break

    if selected is None:
        eligible = grid_df[grid_df["precision"] >= HYBRID_MIN_PRECISION_FOR_SELECTION]
        if eligible.empty:
            selected = grid_df.sort_values(sort_cols, ascending=sort_ascending).iloc[0]
            selection_floor = HYBRID_MIN_PRECISION_FOR_SELECTION
            selection_metric = "f0_5_fallback_no_precision_floor_match"
        else:
            selected = eligible.sort_values(sort_cols, ascending=sort_ascending).iloc[0]
            selection_floor = HYBRID_MIN_PRECISION_FOR_SELECTION
            selection_metric = "f0_5_with_precision_floor"

    per_class_conf = json.loads(str(selected.get("per_class_conf_json", "{}")))
    selected_config = {
        "conf": float(selected["conf"]),
        "fusion_iou": float(selected["fusion_iou"]),
        "mode": str(selected["fusion_mode"]),
        "nms_iou": float(selected["nms_iou"]),
        "single_model_conf": float(selected["single_model_conf"]),
        "per_class_profile": str(selected["per_class_profile"]),
        "per_class_conf": per_class_conf,
        "model_weight_profile": str(selected["model_weight_profile"]),
        "model_class_weights": model_class_weights if str(selected["fusion_mode"]) == "class_weighted_fusion" else {},
        "selection_split": "val",
        "selection_metric": selection_metric,
        "precision_floor": selection_floor,
        "minimum_recall": HYBRID_MIN_RECALL_FOR_SELECTION,
        "minimum_mAP50": HYBRID_MIN_MAP50_FOR_SELECTION,
        "selection_sample_size": len(val_paths),
        "validation_precision": float(selected["precision"]),
        "validation_recall": float(selected["recall"]),
        "validation_f0_5": float(selected["f0_5"]),
        "validation_mAP50": float(selected["mAP50"]),
        "validation_mAP50_95": float(selected["mAP50_95"]),
        "validation_false_positives_per_image": float(selected["false_positives_per_image"]),
    }
    HYBRID_SELECTED_CONFIG_JSON.write_text(json.dumps(selected_config, indent=2))
    HYBRID_SELECTED_CONFIG_V2_JSON.write_text(json.dumps(selected_config, indent=2))
    print("Selected hybrid config:", selected_config)
    return selected_config, grid_df


def run_selected_hybrid_eval(selected_config):
    rows = []
    per_class_rows = []
    selected_test_row = None
    selected_test_per_cls = []
    selected_test_pred_by_class = {}
    selected_test_preds_by_image = {}
    selected_test_gt_by_class = {}

    for split_name in ["val", "test"]:
        image_paths, gt_by_class = build_gt_for_split(split_name)
        cache = collect_hybrid_prediction_cache(image_paths, float(selected_config["conf"]))
        row, per_cls, pred_by_class, preds_by_image = evaluate_cached_hybrid(cache, gt_by_class, selected_config, split_name)
        rows.append(row)
        for cls_row in per_cls:
            per_class_rows.append({
                "model": "Hybrid YOLOv8n + RT-DETR-L",
                "family": "YOLO-Transformer tuned late fusion",
                "split": split_name,
                **cls_row,
                "conf": selected_config["conf"],
                "fusion_iou": selected_config["fusion_iou"],
                "fusion_mode": selected_config["mode"],
                "nms_iou": selected_config["nms_iou"],
                "single_model_conf": selected_config.get("single_model_conf"),
                "per_class_profile": selected_config.get("per_class_profile", "uniform"),
                "model_weight_profile": selected_config.get("model_weight_profile", "equal"),
            })
        if split_name == "test":
            selected_test_row = row
            selected_test_per_cls = per_cls
            selected_test_pred_by_class = pred_by_class
            selected_test_preds_by_image = preds_by_image
            selected_test_gt_by_class = gt_by_class
        print("HYBRID_SELECTED", split_name.upper(), row)

    fusion_df = pd.DataFrame(rows)
    fusion_per_class_df = pd.DataFrame(per_class_rows)
    fusion_df.to_csv(HYBRID_FUSION_CSV, index=False)
    fusion_per_class_df.to_csv(HYBRID_PER_CLASS_CSV, index=False)
    pd.DataFrame([selected_test_row]).to_csv(HYBRID_SELECTED_TEST_CSV, index=False)
    pd.DataFrame([selected_test_row]).to_csv(HYBRID_SELECTED_TEST_V2_CSV, index=False)
    pd.DataFrame([
        {
            "model": "Hybrid YOLOv8n + RT-DETR-L",
            "family": "YOLO-Transformer tuned late fusion",
            "split": "test",
            **row,
            "conf": selected_config["conf"],
            "fusion_iou": selected_config["fusion_iou"],
            "fusion_mode": selected_config["mode"],
            "nms_iou": selected_config["nms_iou"],
            "single_model_conf": selected_config.get("single_model_conf"),
            "per_class_profile": selected_config.get("per_class_profile", "uniform"),
            "model_weight_profile": selected_config.get("model_weight_profile", "equal"),
        }
        for row in selected_test_per_cls
    ]).to_csv(HYBRID_SELECTED_PER_CLASS_CSV, index=False)
    print("Saved hybrid fusion metrics:", HYBRID_FUSION_CSV)
    print("Saved selected hybrid test metrics:", HYBRID_SELECTED_TEST_CSV)
    print("Saved selected hybrid test metrics v2:", HYBRID_SELECTED_TEST_V2_CSV)
    print("Saved selected hybrid per-class metrics:", HYBRID_SELECTED_PER_CLASS_CSV)
    display(fusion_df)
    display(fusion_per_class_df)

    architecture_df = pd.read_csv(ARCHITECTURE_CSV) if ARCHITECTURE_CSV.exists() else pd.DataFrame()
    architecture_rows = fusion_df.drop(columns=["images", "conf", "fusion_iou", "fusion_mode", "nms_iou"], errors="ignore")
    architecture_df = pd.concat([architecture_df, architecture_rows], ignore_index=True)
    architecture_df.to_csv(ARCHITECTURE_CSV, index=False)
    print("Saved architecture comparison:", ARCHITECTURE_CSV)

    return (
        fusion_df,
        fusion_per_class_df,
        selected_test_pred_by_class,
        selected_test_preds_by_image,
        selected_test_gt_by_class,
    )


def run_hybrid_robustness_eval(selected_config):
    rows = []
    per_class_rows = []
    if not RUN_ROBUSTNESS_TESTS:
        return pd.DataFrame(), pd.DataFrame()

    for condition_name in make_robustness_transforms().keys():
        condition_root = ROBUSTNESS_ROOT / condition_name
        image_dir = condition_root / "test/images"
        label_dir = condition_root / "test/labels"
        if not image_dir.exists() or not label_dir.exists():
            continue
        image_paths = sorted(image_dir.glob("*.*"))
        gt_by_class = build_gt_for_image_paths(image_paths, label_dir)
        cache = collect_hybrid_prediction_cache(image_paths, float(selected_config["conf"]))
        row, per_cls, _, _ = evaluate_cached_hybrid(cache, gt_by_class, selected_config, condition_name)
        row.update({"condition": condition_name, "images": len(image_paths)})
        rows.append(row)
        for cls_row in per_cls:
            per_class_rows.append({
                "model": "Hybrid YOLOv8n + RT-DETR-L",
                "family": "YOLO-Transformer tuned late fusion",
                "condition": condition_name,
                **cls_row,
            })
        print("HYBRID_ROBUSTNESS", row)

    robustness_df = pd.DataFrame(rows)
    robustness_per_class_df = pd.DataFrame(per_class_rows)
    robustness_df.to_csv(HYBRID_ROBUSTNESS_CSV, index=False)
    robustness_per_class_df.to_csv(HYBRID_ROBUSTNESS_PER_CLASS_CSV, index=False)
    print("Saved hybrid robustness metrics:", HYBRID_ROBUSTNESS_CSV)
    print("Saved hybrid robustness per-class metrics:", HYBRID_ROBUSTNESS_PER_CLASS_CSV)
    display(robustness_df)
    return robustness_df, robustness_per_class_df


def cv_color_for_class(cls_id):
    palette = [
        (43, 130, 255),
        (60, 180, 75),
        (255, 225, 25),
        (245, 130, 48),
        (145, 30, 180),
        (70, 240, 240),
    ]
    return palette[int(cls_id) % len(palette)]


def draw_xyxy_boxes(image, boxes, label_prefix="", is_gt=False):
    out = image.copy()
    for box in boxes:
        cls_id = int(box.get("cls", box.get("class_id", 0)))
        x1, y1, x2, y2 = [int(round(v)) for v in box["xyxy"]]
        color = (0, 255, 0) if is_gt else cv_color_for_class(cls_id)
        caption = f"{label_prefix}{CLASSES[cls_id]}"
        if "conf" in box:
            caption += f" {box['conf']:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, caption, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def gt_boxes_for_image(gt_by_class, image_name):
    boxes = []
    for cls_id, per_image in gt_by_class.items():
        for xyxy in per_image.get(image_name, []):
            boxes.append({"cls": int(cls_id), "xyxy": xyxy})
    return boxes


def label_panel(image, title):
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(panel, title, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def save_hybrid_visual_evidence(selected_config, split_name="test"):
    image_paths, gt_by_class = build_gt_for_split(split_name)
    rng = random.Random(SEED)
    rng.shuffle(image_paths)
    image_paths = image_paths[: min(HYBRID_VISUAL_SAMPLE_SIZE, len(image_paths))]
    if HYBRID_VIS_DIR.exists():
        shutil.rmtree(HYBRID_VIS_DIR)
    HYBRID_VIS_DIR.mkdir(parents=True, exist_ok=True)

    saved = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        yolo_preds = predict_xyxy(model, image_path, IMG_SIZE, float(selected_config["conf"]))
        rtdetr_preds = predict_xyxy(rtdetr_model, image_path, RTDETR_IMG_SIZE, float(selected_config["conf"]))
        hybrid_preds = hybrid_fuse_predictions_config(yolo_preds, rtdetr_preds, selected_config)
        gt_boxes = gt_boxes_for_image(gt_by_class, image_path.name)

        panels = [
            label_panel(draw_xyxy_boxes(image, gt_boxes, "GT:", is_gt=True), "Ground Truth"),
            label_panel(draw_xyxy_boxes(image, yolo_preds, "YOLO:"), "YOLOv8n"),
            label_panel(draw_xyxy_boxes(image, rtdetr_preds, "RT-DETR:"), "RT-DETR-L"),
            label_panel(draw_xyxy_boxes(image, hybrid_preds, "HYB:"), "Tuned Hybrid"),
        ]
        resized = [cv2.resize(p, (480, 360), interpolation=cv2.INTER_AREA) for p in panels]
        top = np.concatenate(resized[:2], axis=1)
        bottom = np.concatenate(resized[2:], axis=1)
        canvas = np.concatenate([top, bottom], axis=0)
        out_path = HYBRID_VIS_DIR / f"{image_path.stem}_hybrid_comparison.jpg"
        cv2.imwrite(str(out_path), canvas)
        saved.append(out_path.name)

    html = "<div style='display:flex;flex-wrap:wrap;gap:12px'>"
    for name in saved:
        html += f"<a href='hybrid_visual_evidence/{name}' target='_blank'><img src='hybrid_visual_evidence/{name}' style='width:360px;border:1px solid #ccc'></a>"
    html += "</div>"
    (HYBRID_VIS_DIR / "index.html").write_text(html)
    display(HTML(html))
    print("Saved hybrid visual evidence to:", HYBRID_VIS_DIR)
    return saved


def detection_error_analysis(gt_by_class, pred_by_class, preds_by_image, per_class_rows):
    rows = []
    examples = []
    for cls_id in range(len(CLASSES)):
        gt_map = gt_by_class.get(cls_id, {})
        cls_preds = sorted(pred_by_class.get(cls_id, []), key=lambda x: x["conf"], reverse=True)
        matched = {img_id: np.zeros(len(gt_map.get(img_id, [])), dtype=bool) for img_id in gt_map}
        tp = fp = 0
        for pred in cls_preds:
            img_id = pred["image_id"]
            gts = gt_map.get(img_id, [])
            best_iou = 0.0
            best_j = -1
            for j, gt_box in enumerate(gts):
                ov = iou_xyxy(pred["xyxy"], gt_box)
                if ov > best_iou:
                    best_iou = ov
                    best_j = j
            if best_iou >= 0.5 and best_j >= 0 and not matched[img_id][best_j]:
                matched[img_id][best_j] = True
                tp += 1
            else:
                fp += 1
                if len(examples) < 120:
                    examples.append({
                        "type": "false_positive",
                        "image": img_id,
                        "class_name": CLASSES[cls_id],
                        "confidence": pred["conf"],
                        "best_iou": best_iou,
                    })
        fn = 0
        for img_id, flags in matched.items():
            missing = int((~flags).sum())
            fn += missing
            if missing and len(examples) < 120:
                examples.append({
                    "type": "missed_defect",
                    "image": img_id,
                    "class_name": CLASSES[cls_id],
                    "confidence": np.nan,
                    "best_iou": np.nan,
                })
        rows.append({
            "class_id": cls_id,
            "class_name": CLASSES[cls_id],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision_at_iou50": tp / max(tp + fp, 1),
            "recall_at_iou50": tp / max(tp + fn, 1),
        })

    error_df = pd.DataFrame(rows)
    examples_df = pd.DataFrame(examples)
    error_df.to_csv(HYBRID_ERROR_ANALYSIS_CSV, index=False)
    error_df.to_csv(HYBRID_ERROR_ANALYSIS_V2_CSV, index=False)
    examples_df.to_csv(HYBRID_ERROR_EXAMPLES_CSV, index=False)
    print("Saved hybrid error analysis:", HYBRID_ERROR_ANALYSIS_CSV)
    print("Saved hybrid error analysis v2:", HYBRID_ERROR_ANALYSIS_V2_CSV)
    print("Saved hybrid error examples:", HYBRID_ERROR_EXAMPLES_CSV)

    yolo_pc = pd.read_csv(PER_CLASS_CSV) if PER_CLASS_CSV.exists() else pd.DataFrame()
    hybrid_pc = pd.DataFrame(per_class_rows)
    if not yolo_pc.empty and not hybrid_pc.empty:
        yolo_test = yolo_pc[(yolo_pc["model"] == "YOLOv8n") & (yolo_pc["split"] == "test")]
        hybrid_test = hybrid_pc.copy()
        delta = hybrid_test.merge(
            yolo_test[["class_name", "precision", "recall", "mAP50", "mAP50_95"]],
            on="class_name",
            suffixes=("_hybrid", "_yolo"),
        )
        for metric in ["precision", "recall", "mAP50", "mAP50_95"]:
            delta[f"delta_{metric}"] = delta[f"{metric}_hybrid"] - delta[f"{metric}_yolo"]
        delta["hybrid_outcome"] = np.where(delta["delta_mAP50_95"] >= 0, "improved_or_equal", "worse")
        delta.to_csv(HYBRID_CLASS_DELTA_CSV, index=False)
        print("Saved hybrid class delta:", HYBRID_CLASS_DELTA_CSV)
        display(delta)
    return error_df, examples_df


def run_hybrid_fusion_eval():
    if not RUN_HYBRID_FUSION:
        print("Hybrid fusion disabled. Set RUN_HYBRID_FUSION=True to enable late-fusion evaluation.")
        return pd.DataFrame(), pd.DataFrame(), default_hybrid_config()
    if not RUN_RTDETR_BENCHMARK or "rtdetr_model" not in globals():
        print("Hybrid fusion skipped: RT-DETR model is unavailable in this run.")
        return pd.DataFrame(), pd.DataFrame(), default_hybrid_config()

    selected_config, tuning_grid_df = tune_hybrid_config()
    (
        fusion_df,
        fusion_per_class_df,
        selected_test_pred_by_class,
        selected_test_preds_by_image,
        selected_test_gt_by_class,
    ) = run_selected_hybrid_eval(selected_config)
    run_hybrid_robustness_eval(selected_config)
    save_hybrid_visual_evidence(selected_config, split_name="test")
    test_per_class_rows = fusion_per_class_df[fusion_per_class_df["split"] == "test"].to_dict("records")
    detection_error_analysis(
        selected_test_gt_by_class,
        selected_test_pred_by_class,
        selected_test_preds_by_image,
        test_per_class_rows,
    )
    return fusion_df, fusion_per_class_df, selected_config


def measure_hybrid_latency(image_paths, n=80, warmup=10, config=None):
    config = config or globals().get("selected_hybrid_config", default_hybrid_config())
    sample = image_paths[:]
    random.shuffle(sample)
    sample = sample[: min(n, len(sample))]
    if not sample:
        return {}

    for p in sample[: min(warmup, len(sample))]:
        y = predict_xyxy(model, p, IMG_SIZE, float(config["conf"]))
        r = predict_xyxy(rtdetr_model, p, RTDETR_IMG_SIZE, float(config["conf"]))
        _ = hybrid_fuse_predictions_config(y, r, config)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    timings = []
    for p in sample:
        start = time.perf_counter()
        y = predict_xyxy(model, p, IMG_SIZE, float(config["conf"]))
        r = predict_xyxy(rtdetr_model, p, RTDETR_IMG_SIZE, float(config["conf"]))
        _ = hybrid_fuse_predictions_config(y, r, config)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)

    return {
        "images": len(sample),
        "mean_ms": float(np.mean(timings)),
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "fps": float(1000.0 / np.mean(timings)),
        "under_100ms_target": bool(np.mean(timings) < 100.0),
    }


hybrid_fusion_df, hybrid_per_class_df, selected_hybrid_config = run_hybrid_fusion_eval()
"""
    ),
    code_cell(
        """
class PatchRefinerDataset(Dataset):
    def __init__(self, samples, patch_size):
        self.samples = samples
        self.patch_size = patch_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, xyxy, label = self.samples[idx]
        image = cv2.imread(str(image_path))
        if image is None:
            patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8)
        else:
            patch = crop_resize_patch(image, xyxy, self.patch_size)
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(patch.transpose(2, 0, 1))
        return tensor, torch.tensor(int(label), dtype=torch.long)


class CNNTransformerPatchRefiner(nn.Module):
    def __init__(self, num_classes, patch_size=96, d_model=128, nhead=4, depth=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(d_model),
            nn.SiLU(),
        )
        token_side = patch_size // 8
        max_tokens = token_side * token_side + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, max_tokens, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        feats = self.cnn(x)
        tokens = feats.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.shape[1], :]
        encoded = self.transformer(tokens)
        return self.head(encoded[:, 0])


def clamp_xyxy(xyxy, width, height):
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 + 1:
        x2 = min(float(width), x1 + 2)
    if y2 <= y1 + 1:
        y2 = min(float(height), y1 + 2)
    return [x1, y1, x2, y2]


def expand_xyxy(xyxy, width, height, scale=1.8):
    x1, y1, x2, y2 = xyxy
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    nw = bw * scale
    nh = bh * scale
    return clamp_xyxy([cx - nw / 2.0, cy - nh / 2.0, cx + nw / 2.0, cy + nh / 2.0], width, height)


def crop_resize_patch(image, xyxy, patch_size):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in clamp_xyxy(xyxy, w, h)]
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        crop = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    return cv2.resize(crop, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)


def yolo_row_to_expanded_xyxy(row, width, height, scale=1.8):
    cls_id, xc, yc, bw, bh = row
    x1 = (xc - bw / 2.0) * width
    y1 = (yc - bh / 2.0) * height
    x2 = (xc + bw / 2.0) * width
    y2 = (yc + bh / 2.0) * height
    return expand_xyxy([x1, y1, x2, y2], width, height, scale=scale)


def build_patch_refiner_samples():
    train_img_dir = YOLO_ROOT / "train/images"
    train_lbl_dir = YOLO_ROOT / "train/labels"
    rng = random.Random(SEED)

    positives_by_class = {i: [] for i in range(len(CLASSES))}
    image_paths = sorted(train_img_dir.glob("*.*"))
    labels_by_image = {}

    for image_path in image_paths:
        rows = read_train_yolo_file(train_lbl_dir / f"{image_path.stem}.txt")
        labels_by_image[image_path] = rows
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        for row in rows:
            cls_id = row[0]
            xyxy = yolo_row_to_expanded_xyxy(row, w, h, scale=1.8)
            positives_by_class[cls_id].append((image_path, xyxy, cls_id + 1))

    samples = []
    for cls_id, cls_samples in positives_by_class.items():
        rng.shuffle(cls_samples)
        samples.extend(cls_samples[: min(REFINER_MAX_POSITIVE_PER_CLASS, len(cls_samples))])

    negatives = []
    attempts = 0
    while len(negatives) < REFINER_NEGATIVE_SAMPLES and attempts < REFINER_NEGATIVE_SAMPLES * 30 and image_paths:
        attempts += 1
        image_path = rng.choice(image_paths)
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        h, w = image.shape[:2]
        side = rng.randint(max(24, min(w, h) // 20), max(32, min(w, h) // 4))
        x1 = rng.randint(0, max(0, w - side - 1))
        y1 = rng.randint(0, max(0, h - side - 1))
        cand = [x1, y1, x1 + side, y1 + side]
        gt_boxes = []
        for row in labels_by_image.get(image_path, []):
            gt_boxes.append(yolo_row_to_expanded_xyxy(row, w, h, scale=1.1))
        if all(iou_xyxy(cand, gt) < 0.05 for gt in gt_boxes):
            negatives.append((image_path, cand, 0))

    samples.extend(negatives)
    rng.shuffle(samples)
    counts = Counter(label for _, _, label in samples)
    print("CNN-Transformer refiner samples:", dict(counts))
    return samples


def train_cnn_transformer_refiner():
    if not RUN_CNN_TRANSFORMER_REFINER:
        print("CNN-Transformer feature refiner disabled.")
        return None, pd.DataFrame()

    samples = build_patch_refiner_samples()
    if not samples:
        print("No samples available for CNN-Transformer feature refiner.")
        return None, pd.DataFrame()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    split_idx = int(len(samples) * 0.9)
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]

    train_ds = PatchRefinerDataset(train_samples, REFINER_PATCH_SIZE)
    val_ds = PatchRefinerDataset(val_samples, REFINER_PATCH_SIZE)
    train_loader = DataLoader(train_ds, batch_size=REFINER_BATCH, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=REFINER_BATCH, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    refiner = CNNTransformerPatchRefiner(num_classes=len(CLASSES) + 1, patch_size=REFINER_PATCH_SIZE).to(device)
    optimizer = torch.optim.AdamW(refiner.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    history = []

    for epoch in range(1, REFINER_EPOCHS + 1):
        refiner.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = refiner(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * xb.shape[0]
            train_correct += int((logits.argmax(1) == yb).sum().item())
            train_total += int(xb.shape[0])

        refiner.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                logits = refiner(xb)
                loss = criterion(logits, yb)
                val_loss += float(loss.item()) * xb.shape[0]
                val_correct += int((logits.argmax(1) == yb).sum().item())
                val_total += int(xb.shape[0])

        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_total, 1),
            "train_accuracy": train_correct / max(train_total, 1),
            "val_loss": val_loss / max(val_total, 1),
            "val_accuracy": val_correct / max(val_total, 1),
            "train_samples": train_total,
            "val_samples": val_total,
        }
        history.append(row)
        print("REFINER", row)

    history_df = pd.DataFrame(history)
    history_df.to_csv(REFINER_TRAINING_CSV, index=False)
    display(history_df)
    print("Saved CNN-Transformer refiner training history:", REFINER_TRAINING_CSV)
    return refiner, history_df


def refiner_candidate_score(refiner, image_path, pred):
    image = cv2.imread(str(image_path))
    if image is None:
        return 0.0, 0
    device = next(refiner.parameters()).device
    patch = crop_resize_patch(image, pred["xyxy"], REFINER_PATCH_SIZE)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(refiner(tensor), dim=1)[0].detach().cpu().numpy()
    predicted_label = int(np.argmax(probs))
    expected_label = int(pred["cls"]) + 1
    return float(probs[expected_label]), predicted_label


def refiner_filter_predictions(refiner, image_path, preds):
    if refiner is None:
        return preds
    kept = []
    for pred in preds:
        class_prob, predicted_label = refiner_candidate_score(refiner, image_path, pred)
        expected_label = int(pred["cls"]) + 1
        if predicted_label == expected_label and class_prob >= REFINER_KEEP_PROB:
            out = dict(pred)
            out["conf"] = float(min(1.0, 0.5 * out["conf"] + 0.5 * class_prob))
            out["refiner_prob"] = class_prob
            kept.append(out)
    return nms_class_aware(kept, iou_thr=0.55)


def run_refined_hybrid_eval(refiner):
    if refiner is None:
        return pd.DataFrame(), pd.DataFrame()
    if not RUN_RTDETR_BENCHMARK or "rtdetr_model" not in globals():
        print("CNN-Transformer refined hybrid skipped: RT-DETR model is unavailable.")
        return pd.DataFrame(), pd.DataFrame()

    rows = []
    per_class_rows = []
    refiner.eval()

    for split_name in ["val", "test"]:
        image_paths, gt_by_class = build_gt_for_split(split_name)
        pred_by_class = {i: [] for i in range(len(CLASSES))}
        base_config = globals().get("selected_hybrid_config", default_hybrid_config())
        for image_path in image_paths:
            y_preds = predict_xyxy(model, image_path, IMG_SIZE, REFINER_CANDIDATE_CONF)
            r_preds = predict_xyxy(rtdetr_model, image_path, RTDETR_IMG_SIZE, REFINER_CANDIDATE_CONF)
            fused = hybrid_fuse_predictions_config(y_preds, r_preds, base_config)
            refined = refiner_filter_predictions(refiner, image_path, fused)
            for pred in refined:
                pred_by_class[pred["cls"]].append({
                    "image_id": image_path.name,
                    "conf": float(pred["conf"]),
                    "xyxy": pred["xyxy"],
                })

        per_cls = evaluate_predictions(gt_by_class, pred_by_class)
        for row in per_cls:
            per_class_rows.append({
                "model": "Custom CNN-Transformer refined hybrid",
                "family": "Feature-level CNN-Transformer refiner",
                "split": split_name,
                **row,
            })
        row = {
            "model": "Custom CNN-Transformer refined hybrid",
            "family": "Feature-level CNN-Transformer refiner",
            "split": split_name,
            "precision": float(np.mean([r["precision"] for r in per_cls])) if per_cls else 0.0,
            "recall": float(np.mean([r["recall"] for r in per_cls])) if per_cls else 0.0,
            "mAP50": float(np.mean([r["mAP50"] for r in per_cls])) if per_cls else 0.0,
            "mAP50_95": float(np.mean([r["mAP50_95"] for r in per_cls])) if per_cls else 0.0,
            "images": len(image_paths),
            "refiner_keep_prob": REFINER_KEEP_PROB,
            "candidate_conf": REFINER_CANDIDATE_CONF,
        }
        rows.append(row)
        print("CNN-TRANSFORMER REFINED HYBRID", split_name.upper(), row)

    metrics_df = pd.DataFrame(rows)
    per_class_df = pd.DataFrame(per_class_rows)
    metrics_df.to_csv(REFINER_METRICS_CSV, index=False)
    per_class_df.to_csv(REFINER_PER_CLASS_CSV, index=False)
    display(metrics_df)
    display(per_class_df)
    print("Saved CNN-Transformer refined hybrid metrics:", REFINER_METRICS_CSV)
    print("Saved CNN-Transformer refined hybrid per-class metrics:", REFINER_PER_CLASS_CSV)

    architecture_df = pd.read_csv(ARCHITECTURE_CSV) if ARCHITECTURE_CSV.exists() else pd.DataFrame()
    architecture_df = pd.concat([architecture_df, metrics_df.drop(columns=["images", "refiner_keep_prob", "candidate_conf"])], ignore_index=True)
    architecture_df.to_csv(ARCHITECTURE_CSV, index=False)
    print("Saved architecture comparison:", ARCHITECTURE_CSV)
    return metrics_df, per_class_df


def measure_refined_hybrid_latency(refiner, image_paths, n=80, warmup=10):
    if refiner is None or "rtdetr_model" not in globals():
        return {}
    sample = image_paths[:]
    random.shuffle(sample)
    sample = sample[: min(n, len(sample))]
    if not sample:
        return {}

    refiner.eval()
    base_config = globals().get("selected_hybrid_config", default_hybrid_config())
    for p in sample[: min(warmup, len(sample))]:
        y = predict_xyxy(model, p, IMG_SIZE, REFINER_CANDIDATE_CONF)
        r = predict_xyxy(rtdetr_model, p, RTDETR_IMG_SIZE, REFINER_CANDIDATE_CONF)
        fused = hybrid_fuse_predictions_config(y, r, base_config)
        _ = refiner_filter_predictions(refiner, p, fused)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    timings = []
    for p in sample:
        start = time.perf_counter()
        y = predict_xyxy(model, p, IMG_SIZE, REFINER_CANDIDATE_CONF)
        r = predict_xyxy(rtdetr_model, p, RTDETR_IMG_SIZE, REFINER_CANDIDATE_CONF)
        fused = hybrid_fuse_predictions_config(y, r, base_config)
        _ = refiner_filter_predictions(refiner, p, fused)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)

    return {
        "images": len(sample),
        "mean_ms": float(np.mean(timings)),
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "fps": float(1000.0 / np.mean(timings)),
        "under_100ms_target": bool(np.mean(timings) < 100.0),
    }


cnn_transformer_refiner, cnn_transformer_refiner_history = train_cnn_transformer_refiner()
refined_hybrid_df, refined_hybrid_per_class_df = run_refined_hybrid_eval(cnn_transformer_refiner)
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
if RUN_HYBRID_FUSION and RUN_RTDETR_BENCHMARK and "rtdetr_model" in globals():
    latency_summary["hybrid_yolo_rtdetr_fusion"] = measure_hybrid_latency(test_paths, n=80, warmup=10)
if RUN_CNN_TRANSFORMER_REFINER and "cnn_transformer_refiner" in globals() and cnn_transformer_refiner is not None:
    latency_summary["cnn_transformer_refined_hybrid"] = measure_refined_hybrid_latency(
        cnn_transformer_refiner, test_paths, n=80, warmup=10
    )

print("Latency summary:", latency_summary)
(WORK_DIR / "latency_summary.json").write_text(json.dumps(latency_summary, indent=2))

latency_rows = []
for model_name, stats in latency_summary.items():
    latency_rows.append({"model": model_name, **stats})
latency_df = pd.DataFrame(latency_rows)
latency_df.to_csv(LATENCY_TABLE_CSV, index=False)
display(latency_df)
print("Saved latency table:", LATENCY_TABLE_CSV)
"""
    ),
    code_cell(
        """
def load_csv_or_empty(path: Path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


summary_df = load_csv_or_empty(SUMMARY_CSV)
architecture_df = load_csv_or_empty(ARCHITECTURE_CSV)
robustness_df = load_csv_or_empty(ROBUSTNESS_CSV)
per_class_df = load_csv_or_empty(PER_CLASS_CSV)
latency_df = load_csv_or_empty(LATENCY_TABLE_CSV)

print("YOLOv8n val/test metrics")
display(summary_df)
print("Architecture comparison (YOLOv8n vs RT-DETR-L vs Hybrid)")
display(architecture_df)
print("Robustness metrics")
display(robustness_df)
print("Per-class metrics")
display(per_class_df)
print("Latency comparison")
display(latency_df)

final_summary_rows = []
if not architecture_df.empty:
    top_arch = architecture_df.sort_values("mAP50_95", ascending=False).iloc[0]
    final_summary_rows.append({
        "summary_item": "Best architecture row by mAP50-95",
        "model": top_arch.get("model", ""),
        "split": top_arch.get("split", ""),
        "mAP50": top_arch.get("mAP50", np.nan),
        "mAP50_95": top_arch.get("mAP50_95", np.nan),
    })
if not robustness_df.empty:
    weakest = robustness_df.sort_values("mAP50_95", ascending=True).iloc[0]
    final_summary_rows.append({
        "summary_item": "Weakest robustness condition",
        "model": "YOLOv8n",
        "split": weakest.get("condition", ""),
        "mAP50": weakest.get("mAP50", np.nan),
        "mAP50_95": weakest.get("mAP50_95", np.nan),
    })
if not latency_df.empty:
    fastest = latency_df.sort_values("mean_ms", ascending=True).iloc[0]
    final_summary_rows.append({
        "summary_item": "Fastest model by mean latency",
        "model": fastest.get("model", ""),
        "split": "test",
        "mAP50": np.nan,
        "mAP50_95": np.nan,
    })
if not architecture_df.empty and not latency_df.empty:
    hybrid_arch = architecture_df[architecture_df["model"] == "Hybrid YOLOv8n + RT-DETR-L"]
    hybrid_lat = latency_df[latency_df["model"] == "hybrid_yolo_rtdetr_fusion"]
    if not hybrid_arch.empty and not hybrid_lat.empty:
        h_arch = hybrid_arch.sort_values("mAP50_95", ascending=False).iloc[0]
        h_lat = hybrid_lat.iloc[0]
        final_summary_rows.append({
            "summary_item": "Hybrid accuracy-latency tradeoff",
            "model": h_arch.get("model", ""),
            "split": h_arch.get("split", ""),
            "mAP50": h_arch.get("mAP50", np.nan),
            "mAP50_95": h_arch.get("mAP50_95", np.nan),
            "mean_latency_ms": h_lat.get("mean_ms", np.nan),
            "p95_latency_ms": h_lat.get("p95_ms", np.nan),
            "under_100ms_target": h_lat.get("under_100ms_target", False),
        })

final_summary_df = pd.DataFrame(final_summary_rows)
final_summary_df.to_csv(FINAL_SUMMARY_CSV, index=False)
print("Saved final summary:", FINAL_SUMMARY_CSV)
display(final_summary_df)
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
jetson_status = {
    "onnx_export": exported.get("onnx"),
    "tensorrt_status": "prepared_not_built_on_kaggle" if not RUN_TENSORRT_EXPORT else "attempted_in_current_runtime",
    "jetson_note": "Final TensorRT engine should be built and benchmarked on the NVIDIA Jetson Orin target because TensorRT engines are hardware/runtime specific.",
    "recommended_jetson_command": "yolo export model=best.pt format=engine imgsz=640 half=True device=0",
    "mean_latency_requirement_ms": 100,
    "reporting_note": "Report ONNX as completed; report TensorRT as deployment path prepared unless a Jetson Orin run produces an engine and latency measurement.",
}
JETSON_DEPLOYMENT_STATUS_JSON.write_text(json.dumps(jetson_status, indent=2))
print(exported)
print("Saved Jetson/TensorRT deployment status:", JETSON_DEPLOYMENT_STATUS_JSON)
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
    ["Mitigate class imbalance with synthetic augmentation", "Generates copy-paste synthetic defect images for weak/rare classes as a practical TransGAN-style class-balancing surrogate; summary saved to synthetic_augmentation_summary.csv"],
    ["Improve small-defect robustness", "Mosaic/MixUp/copy-paste, perspective, HSV, synthetic defect augmentation, multi-scale-ready YOLO training, plus Albumentations robustness tests; mild synthetic Gaussian noise in robustness evaluation and documented training-noise limitation"],
    ["Report precision, recall, mAP", "Validation and held-out test metrics saved to project_metrics_summary.csv with per-class metrics in per_class_metrics.csv"],
    ["Evaluate robustness under imaging variation", "robustness_metrics.csv records clean, brightness/contrast, noise, blur, and rotation stress tests"],
    ["Evaluate hybrid robustness under imaging variation", "hybrid_robustness_metrics.csv records the selected tuned hybrid under clean, brightness/contrast, noise, blur, and rotation conditions"],
    ["Meet real-time target under 100 ms/image", "latency_summary.json and latency_comparison.csv record mean, p50, p95 latency per model (YOLOv8n, RT-DETR-L, tuned hybrid, and refined hybrid when phases run) and 100 ms pass/fail"],
    ["Provide deployment artifact", "ONNX export plus Jetson/TensorRT deployment status in jetson_deployment_status.json; final TensorRT engine should be built on Jetson Orin"],
    ["Compare CNN and Transformer-style detectors", "architecture_comparison.csv records YOLOv8n and RT-DETR-L on val and test splits"],
    ["Implement Hybrid YOLO-Transformer detection", "YOLOv8n proposal detector + RT-DETR-L transformer-style validation/refinement with tuned late-fusion evaluation."],
    ["Tune hybrid model using validation only", "hybrid_tuning_grid.csv and hybrid_tuning_grid_v2.csv sweep confidence, fusion IoU, fusion mode, single-model fallback confidence, per-class thresholds, class-aware NMS, and class-weighted fusion; hybrid_selected_config.json and hybrid_selected_config_v2.json store the validation-selected F0.5/precision-floor config; hybrid_selected_test_metrics_v2.csv evaluates that config on test"],
    ["Analyze hybrid errors", "hybrid_error_analysis.csv, hybrid_error_examples.csv, and hybrid_class_delta.csv summarize false positives, missed defects, and class-level gains/failures"],
    ["Implement custom feature-level CNN-Transformer model", "Trains a custom CNN-Transformer patch refiner on defect/background crops and evaluates refined hybrid predictions in cnn_transformer_refined_hybrid_metrics.csv"],
    ["Provide interpretable visual output", "Ground-truth vs YOLO vs RT-DETR vs tuned hybrid comparison panels saved under hybrid_visual_evidence plus standard prediction gallery"],
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
