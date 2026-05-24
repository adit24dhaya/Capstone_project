"""STEP 1 — Build YOLO_PCB on Kaggle. Paste this entire cell and Run.
Requires: GPU optional for prep, Internet ON, dataset aditya2402/pcb-dataset attached.
Success: Prepared train images: ~5551, val/test ~1016 each.
"""

# ===== notebook cell 2 =====
import importlib.util
import subprocess
import sys

required_packages = {
    "ultralytics": "ultralytics",
    "albumentations": "albumentations",
    "ensemble_boxes": "ensemble-boxes",
    "pandas": "pandas",
    "numpy": "numpy",
    "cv2": "opencv-python",
    "matplotlib": "matplotlib",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
    "requests": "requests",
}
missing = [pip_name for module_name, pip_name in required_packages.items() if importlib.util.find_spec(module_name) is None]

if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
else:
    print("All required packages are already installed")


def _gpu_compute_major() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True,
        ).strip()
        return int(float(out.splitlines()[0]))
    except Exception:
        return None


_gpu_major = _gpu_compute_major()
if _gpu_major is not None and _gpu_major < 7:
    print(
        f"Detected CUDA capability {_gpu_major}.x (e.g. Tesla P100). "
        "Kaggle's default torch 2.10+cu128 does not support sm_60; installing cu118 wheels..."
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--upgrade",
            "torch==2.5.1",
            "torchvision==0.20.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu118",
        ]
    )

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
from ensemble_boxes import weighted_boxes_fusion
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
from PIL import Image
import sklearn
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from IPython.display import HTML, display

if torch.cuda.is_available():
    _cap = torch.cuda.get_device_capability(0)
    if _cap[0] < 7 and "+cu128" in torch.__version__:
        raise RuntimeError(
            f"PyTorch {torch.__version__} is incompatible with "
            f"{torch.cuda.get_device_name(0)} (sm_{_cap[0]}{_cap[1]}). "
            "Re-run Cell 2 after the cu118 reinstall, or switch Kaggle GPU to T4."
        )
    print("CUDA device:", torch.cuda.get_device_name(0), "capability:", _cap, "torch:", torch.__version__)

from ultralytics import YOLO

os.environ.setdefault("WANDB_MODE", "disabled")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Reproducibility seed:", SEED)
print("Python:", sys.version.split()[0])
print("ultralytics:", __import__("ultralytics").__version__)
print("albumentations:", A.__version__)
print("ensemble-boxes:", __import__("ensemble_boxes").__version__ if hasattr(__import__("ensemble_boxes"), "__version__") else "installed")
print("pandas:", pd.__version__)
print("numpy:", np.__version__)
print("opencv-python:", cv2.__version__)
print("matplotlib:", matplotlib.__version__)
print("Pillow:", Image.__version__)
print("scikit-learn:", sklearn.__version__)
print("torch:", torch.__version__)
print("onnx:", onnx.__version__)
print("onnxruntime:", ort.__version__)
# ===== notebook cell 3 =====
EXPERIMENT_TAG = "v18d_timeout_safe_yolo_run"
VAL_RATIO = 0.15
TEST_RATIO = 0.15
EPOCHS = 65  # v18c peaked near epoch 64; 80 epochs plus RT-DETR exceeded Kaggle wall time.
IMG_SIZE = 1280
BATCH = 8  # P100 16GB @ imgsz 1280; use 12 on T4/V100 if assigned
RUN_RTDETR_BENCHMARK = False  # Run RT-DETR in a separate Kaggle/Nautilus job to avoid timeout.
RTDETR_EPOCHS = 10
RTDETR_BATCH = 4
RTDETR_IMG_SIZE = 640
RUN_ROBUSTNESS_TESTS = True
ROBUSTNESS_SAMPLE_SIZE = 150
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
RUN_HYBRID_FUSION = False  # Requires RT-DETR predictions; run separately after RT-DETR completes.
HYBRID_CONF = 0.01
HYBRID_FUSION_IOU = 0.55
HYBRID_EVAL_SAMPLE_SIZE = None
RUN_HYBRID_TUNING = False
HYBRID_TUNING_CONF_VALUES = [0.15, 0.25, 0.30]
HYBRID_TUNING_IOU_VALUES = [0.40, 0.45, 0.55]
HYBRID_TUNING_MODES = ["agreement_only", "weighted_fusion", "single_high_conf_fallback", "class_weighted_fusion"]
HYBRID_TUNING_NMS_VALUES = [0.45, 0.60]
HYBRID_TUNING_SINGLE_MODEL_CONF_VALUES = [0.50, 0.60, 0.70]
HYBRID_TUNING_PER_CLASS_PROFILES = ["uniform", "precision_boost", "aggressive_precision"]
HYBRID_TUNING_PRECISION_FLOORS = [0.65, 0.70, 0.75]
HYBRID_MIN_PRECISION_FOR_SELECTION = 0.65
HYBRID_MIN_RECALL_FOR_SELECTION = 0.78
HYBRID_MIN_MAP50_FOR_SELECTION = 0.82
HYBRID_TUNING_SAMPLE_SIZE = 250
HYBRID_VISUAL_SAMPLE_SIZE = 12
FINAL_PROFILE_NAME = "balanced"
RUN_FINAL_BALANCED_EVAL = False
RUN_VISUAL_GALLERY = True
RUN_CNN_TRANSFORMER_REFINER = False
REFINER_EPOCHS = 5
REFINER_BATCH = 64
REFINER_PATCH_SIZE = 96
REFINER_MAX_POSITIVE_PER_CLASS = 650
REFINER_NEGATIVE_SAMPLES = 1600
REFINER_KEEP_PROB = 0.45
REFINER_CANDIDATE_CONF = 0.03
RUN_PUBLICATION_ANALYSIS = False  # Keep Kaggle run focused; build paper package from saved artifacts locally.
ANALYSIS_SINGLE_MODEL_CONF = 0.05
V16_YOLO11S_REFERENCE = {
    "experiment_tag": "v16_final_full_analysis",
    "precision": 0.8908843529738135,
    "recall": 0.8611181145255921,
    "mAP50": 0.9054445661650888,
    "mAP50_95": 0.5068167219178348,
}

DSPCBSD_PLUS_URL = "https://ndownloader.figshare.com/files/44069552"
DSPCBSD_PLUS_MD5 = "508334b65bdaea7336f4c1b5d5a80a81"
DSPCBSD_PLUS_DOI = "10.6084/m9.figshare.24970329.v1"

CLASSES = [
    "Mouse_bite",
    "Spur",
    "Open_circuit",
    "Short",
    "Missing_hole",
    "Spurious_copper",
]

CLASS_TO_ID = {name.lower(): i for i, name in enumerate(CLASSES)}
PER_CLASS_CONF = {
    "Mouse_bite": 0.10,
    "Spur": 0.12,
    "Open_circuit": 0.15,
    "Short": 0.15,
    "Missing_hole": 0.10,
    "Spurious_copper": 0.15,
}

DSPCBSD_YOLO_CODES = ["SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO"]
DSPCBSD_TO_PROJECT_CLASS = {
    "SH": "Short",
    "SP": "Spur",
    "SC": "Spurious_copper",
    "OP": "Open_circuit",
    "MB": "Mouse_bite",
}

WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("working")
CACHE_DIR = Path("/kaggle/temp") if Path("/kaggle").exists() else WORK_DIR / "external"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
YOLO_ROOT = CACHE_DIR / "YOLO_PCB"
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
ACCURACY_TUNING_COMPARISON_CSV = WORK_DIR / "accuracy_tuning_comparison.csv"
HYBRID_FUSION_CSV = WORK_DIR / "hybrid_fusion_metrics.csv"
HYBRID_PER_CLASS_CSV = WORK_DIR / "hybrid_per_class_metrics.csv"
HYBRID_TUNING_GRID_CSV = WORK_DIR / "hybrid_tuning_grid.csv"
HYBRID_SELECTED_CONFIG_JSON = WORK_DIR / "hybrid_selected_config.json"
HYBRID_SELECTED_TEST_CSV = WORK_DIR / "hybrid_selected_test_metrics.csv"
HYBRID_SELECTED_PER_CLASS_CSV = WORK_DIR / "hybrid_selected_per_class_metrics.csv"
HYBRID_TUNING_GRID_V2_CSV = WORK_DIR / "hybrid_tuning_grid_v2.csv"
HYBRID_SELECTED_CONFIG_V2_JSON = WORK_DIR / "hybrid_selected_config_v2.json"
HYBRID_SELECTED_TEST_V2_CSV = WORK_DIR / "hybrid_selected_test_metrics_v2.csv"
HYBRID_PARETO_FRONTIER_CSV = WORK_DIR / "hybrid_pareto_frontier.csv"
HYBRID_SELECTED_PROFILES_CONFIG_JSON = WORK_DIR / "hybrid_selected_profiles_config.json"
HYBRID_SELECTED_PROFILES_TEST_CSV = WORK_DIR / "hybrid_selected_profiles_test_metrics.csv"
HYBRID_ERROR_ANALYSIS_V2_CSV = WORK_DIR / "hybrid_error_analysis_v2.csv"
HYBRID_ROBUSTNESS_CSV = WORK_DIR / "hybrid_robustness_metrics.csv"
HYBRID_ROBUSTNESS_PER_CLASS_CSV = WORK_DIR / "hybrid_robustness_per_class_metrics.csv"
HYBRID_ERROR_ANALYSIS_CSV = WORK_DIR / "hybrid_error_analysis.csv"
HYBRID_ERROR_EXAMPLES_CSV = WORK_DIR / "hybrid_error_examples.csv"
HYBRID_CLASS_DELTA_CSV = WORK_DIR / "hybrid_class_delta.csv"
HYBRID_VIS_DIR = WORK_DIR / "hybrid_visual_evidence"
HYBRID_FINAL_BALANCED_CONFIG_JSON = WORK_DIR / "hybrid_final_balanced_config.json"
HYBRID_FINAL_BALANCED_TEST_CSV = WORK_DIR / "hybrid_final_balanced_test_metrics.csv"
HYBRID_FINAL_BALANCED_PER_CLASS_CSV = WORK_DIR / "hybrid_final_balanced_per_class_metrics.csv"
HYBRID_FINAL_BALANCED_ROBUSTNESS_CSV = WORK_DIR / "hybrid_final_balanced_robustness_metrics.csv"
HYBRID_FINAL_BALANCED_ERROR_ANALYSIS_CSV = WORK_DIR / "hybrid_final_balanced_error_analysis.csv"
HYBRID_FINAL_BALANCED_ERROR_EXAMPLES_CSV = WORK_DIR / "hybrid_final_balanced_error_examples.csv"
HYBRID_FINAL_BALANCED_LATENCY_CSV = WORK_DIR / "hybrid_final_balanced_latency.csv"
HYBRID_FINAL_BALANCED_VIS_DIR = WORK_DIR / "hybrid_final_balanced_visual_evidence"
HYBRID_PARETO_PROFILE_PLOT = WORK_DIR / "hybrid_pareto_profile_plot.png"
HYBRID_FINAL_BALANCED_ROBUSTNESS_PLOT = WORK_DIR / "hybrid_final_balanced_robustness_plot.png"
LATENCY_COMPARISON_PLOT = WORK_DIR / "latency_comparison_plot.png"
SYNTHETIC_AUGMENTATION_CSV = WORK_DIR / "synthetic_augmentation_summary.csv"
REFINER_TRAINING_CSV = WORK_DIR / "cnn_transformer_refiner_training.csv"
REFINER_METRICS_CSV = WORK_DIR / "cnn_transformer_refined_hybrid_metrics.csv"
REFINER_PER_CLASS_CSV = WORK_DIR / "cnn_transformer_refined_hybrid_per_class.csv"
JETSON_DEPLOYMENT_STATUS_JSON = WORK_DIR / "jetson_deployment_status.json"
DEFECT_SIZE_ANALYSIS_CSV = WORK_DIR / "defect_size_analysis.csv"
ADAPTIVE_POLICY_JSON = WORK_DIR / "adaptive_defect_aware_policy.json"
ADAPTIVE_POLICY_CSV = WORK_DIR / "adaptive_defect_aware_policy.csv"
ADAPTIVE_TEST_CSV = WORK_DIR / "adaptive_defect_aware_hybrid_test_metrics.csv"
ADAPTIVE_PER_CLASS_CSV = WORK_DIR / "adaptive_defect_aware_hybrid_per_class_metrics.csv"
ADAPTIVE_SIZE_ANALYSIS_CSV = WORK_DIR / "adaptive_defect_aware_size_analysis.csv"
INSPECTION_COST_CSV = WORK_DIR / "industrial_inspection_cost_metrics.csv"
CALIBRATION_METRICS_CSV = WORK_DIR / "calibration_metrics.csv"
CALIBRATION_RELIABILITY_CSV = WORK_DIR / "calibration_reliability_bins.csv"
CALIBRATION_RELIABILITY_PLOT = WORK_DIR / "calibration_reliability_plot.png"

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    raise RuntimeError("No Kaggle GPU is active. In Kaggle, open Settings and set Accelerator to GPU.")

print("Experiment tag:", EXPERIMENT_TAG)
print("YOLO training config:", {
    "model": "yolo11s.pt",
    "epochs": EPOCHS,
    "imgsz": IMG_SIZE,
    "batch": BATCH,
    "rt_detr_enabled": RUN_RTDETR_BENCHMARK,
    "hybrid_enabled": RUN_HYBRID_FUSION,
    "publication_analysis_enabled": RUN_PUBLICATION_ANALYSIS,
})

# ===== notebook cell 4 =====
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
# ===== notebook cell 5 =====
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


def download_dspcbsd_plus_zip(zip_path: Path) -> None:
    """Download DsPCBSD+ with browser-like headers; Figshare often returns 403 to bare urllib."""
    import requests

    urls = [
        DSPCBSD_PLUS_URL,
        "https://ndownloader.figshare.com/articles/24970329/versions/1",
    ]
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; PCB-defect-capstone/1.0; +https://www.kaggle.com/)",
    }
    last_error = None
    for url in urls:
        try:
            print(f"Downloading DsPCBSD+ from {url} ...")
            with requests.get(url, headers=headers, stream=True, timeout=600) as response:
                response.raise_for_status()
                with zip_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size= 1 << 20):
                        if chunk:
                            handle.write(chunk)
            return
        except Exception as exc:
            last_error = exc
            print(f"Download failed for {url}: {exc}")
    raise RuntimeError(f"Could not download DsPCBSD+ zip: {last_error}")


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
        try:
            download_dspcbsd_plus_zip(zip_path)
        except Exception as exc:
            print(
                "WARNING: DsPCBSD+ download failed; continuing with PCB-DATASET only. "
                f"Reason: {exc}"
            )
            return None

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
# ===== notebook cell 6 =====
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
# ===== notebook cell 8 =====
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
        dest_label.write_text("\n".join(label_lines) + "\n")


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
        synthetic_noise = A.GaussNoise(std_range=(0.012, 0.028), mean_range=(0.0, 0.0), p=0.3)
    except TypeError:
        synthetic_noise = A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)

    transform = A.Compose(
        [
            A.RandomBrightnessContrast(p=0.4),
            synthetic_noise,
            A.MotionBlur(blur_limit=7, p=0.2),
            A.Rotate(limit=10, p=0.3),
            A.HorizontalFlip(p=0.5),
        ],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"], min_visibility=0.2),
    )

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

            src_h, src_w = patch.shape[:2]
            clipped_box = [
                max(0.0, min(patch_box[0], src_w - 1.0)),
                max(0.0, min(patch_box[1], src_h - 1.0)),
                max(1.0, min(patch_box[2], float(src_w))),
                max(1.0, min(patch_box[3], float(src_h))),
            ]
            if clipped_box[2] <= clipped_box[0] + 1 or clipped_box[3] <= clipped_box[1] + 1:
                continue
            transformed = transform(image=patch, bboxes=[clipped_box], labels=[cls_id])
            if not transformed.get("bboxes"):
                continue
            patch = transformed["image"]
            patch_box = list(transformed["bboxes"][0])
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
            (train_lbl_dir / f"{output_stem}.txt").write_text("\n".join(label_lines) + "\n")
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

names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
DATA_YAML.write_text(
    f"path: {YOLO_ROOT}\n"
    "train: train/images\n"
    "val: val/images\n"
    "test: test/images\n"
    f"names:\n{names_block}\n"
)

print(DATA_YAML.read_text())
print("Prepared train images:", len(list((YOLO_ROOT / "train/images").glob("*"))))
print("Prepared val images:", len(list((YOLO_ROOT / "val/images").glob("*"))))
print("Prepared test images:", len(list((YOLO_ROOT / "test/images").glob("*"))))
# ===== STEP 1 DONE — verify =====
for split in ("train", "val", "test"):
    n = len(list((YOLO_ROOT / split / "images").glob("*")))
    print(f"{split}: {n} images")
assert len(list((YOLO_ROOT / "train/images").glob("*"))) > 5000, "Train count too low — check prep"
print("YOLO_ROOT:", YOLO_ROOT)
print("DATA_YAML:", DATA_YAML)
