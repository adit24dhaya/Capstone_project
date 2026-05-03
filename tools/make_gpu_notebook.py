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

This notebook is configured for Kaggle GPU. It prepares the PCB dataset, converts Pascal VOC XML annotations to YOLO labels, trains a YOLOv8 detector, evaluates validation/test performance, measures inference latency, and exports deployment artifacts.
"""
    ),
    code_cell(
        """
import importlib.util
import subprocess
import sys

if importlib.util.find_spec("ultralytics") is None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics", "onnx", "onnxruntime"])
else:
    print("ultralytics is already installed")
"""
    ),
    code_cell(
        """
from pathlib import Path
from collections import Counter
import json
import random
import shutil
import time
import xml.etree.ElementTree as ET

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import HTML, display
from ultralytics import YOLO

SEED = 42
VAL_RATIO = 0.15
TEST_RATIO = 0.15
EPOCHS = 50
IMG_SIZE = 640
BATCH = 16
RUN_RTDETR_BENCHMARK = False
RUN_TENSORRT_EXPORT = False

random.seed(SEED)
np.random.seed(SEED)

CLASSES = [
    "Mouse_bite",
    "Spur",
    "Open_circuit",
    "Short",
    "Missing_hole",
    "Spurious_copper",
]

CLASS_TO_ID = {name.lower(): i for i, name in enumerate(CLASSES)}

WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("working")
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
for split_name in ["val", "test"]:
    metrics = model.val(data=str(DATA_YAML), imgsz=IMG_SIZE, device=0, split=split_name)
    row = metrics_to_dict(metrics, split_name)
    summary_rows.append(row)
    print(split_name.upper(), row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(SUMMARY_CSV, index=False)
display(summary_df)
print("Saved metric summary:", SUMMARY_CSV)
"""
    ),
    code_cell(
        """
if RUN_RTDETR_BENCHMARK:
    from ultralytics import RTDETR

    rtdetr_model = RTDETR("rtdetr-l.pt")
    rtdetr_results = rtdetr_model.train(
        data=str(DATA_YAML),
        epochs=max(10, EPOCHS // 2),
        imgsz=IMG_SIZE,
        batch=max(2, BATCH // 2),
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
    rtdetr_metrics = rtdetr_model.val(data=str(DATA_YAML), imgsz=IMG_SIZE, device=0, split="test")
    print("RT-DETR transformer benchmark:", metrics_to_dict(rtdetr_metrics, "test_rtdetr"))
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
def measure_latency(image_paths, n=80, warmup=10, conf=0.25):
    sample = image_paths[:]
    random.shuffle(sample)
    sample = sample[: min(n, len(sample))]
    if not sample:
        return {}

    for p in sample[: min(warmup, len(sample))]:
        _ = model.predict(source=str(p), imgsz=IMG_SIZE, conf=conf, device=0, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    timings = []
    for p in sample:
        start = time.perf_counter()
        _ = model.predict(source=str(p), imgsz=IMG_SIZE, conf=conf, device=0, verbose=False)
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


latency = measure_latency(sorted(TEST_IMG_DIR.glob("*.*")), n=80, warmup=10, conf=0.25)
print("Latency summary:", latency)
(WORK_DIR / "latency_summary.json").write_text(json.dumps(latency, indent=2))
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
    ["Detect and localize six PCB defect classes", "YOLO dataset conversion, YOLOv8 training, prediction gallery"],
    ["Improve small-defect robustness", "Mosaic/MixUp/copy-paste, perspective, HSV, multi-scale-ready YOLO training"],
    ["Report precision, recall, mAP", "Validation and held-out test metrics saved to project_metrics_summary.csv"],
    ["Meet real-time target under 100 ms/image", "latency_summary.json records mean, p50, p95 latency and target pass/fail"],
    ["Provide deployment artifact", "ONNX export plus optional TensorRT export path"],
    ["Compare CNN and Transformer-style detectors", "Optional RT-DETR benchmark switch for transformer comparison"],
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

out = Path("kaggle_kernel/project.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(notebook, indent=1))
print(f"Wrote {out}")
