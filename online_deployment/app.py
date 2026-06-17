from __future__ import annotations

import json
import os
import tempfile
import traceback
from collections import Counter
from functools import lru_cache
from html import escape
from pathlib import Path
from time import perf_counter, time
from uuid import uuid4

import gradio as gr
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


GRADIO_MAJOR = int(gr.__version__.split(".", maxsplit=1)[0])

MODEL_CANDIDATES = (
    Path(os.environ.get("MODEL_PATH", "")) if os.environ.get("MODEL_PATH") else None,
    Path("models/best.pt"),
    Path("models/best.onnx"),
    Path("best.pt"),
    Path("best.onnx"),
)

CLASS_NAMES = {
    0: "Missing_hole",
    1: "Mouse_bite",
    2: "Open_circuit",
    3: "Short",
    4: "Spur",
    5: "Spurious_copper",
}

COLORS = {
    "Missing_hole": "#0072B2",
    "Mouse_bite": "#D55E00",
    "Open_circuit": "#009E73",
    "Short": "#CC79A7",
    "Spur": "#E69F00",
    "Spurious_copper": "#56B4E9",
}

CLASS_DISPLAY_NAMES = {
    class_name: class_name.replace("_", " ") for class_name in CLASS_NAMES.values()
}
DISPLAY_TO_CLASS = {display: class_name for class_name, display in CLASS_DISPLAY_NAMES.items()}
CLASS_FILTER_CHOICES = list(CLASS_DISPLAY_NAMES.values())
DATASET_IMAGE_URL = "https://github.com/Ironbrotherstyle/PCB-DATASET/tree/master/images"
DATASET_REPO_URL = "https://github.com/Ironbrotherstyle/PCB-DATASET"

MODEL_LABEL = os.environ.get("MODEL_LABEL", "YOLO11s")
DEFAULT_IMGSZ = int(os.environ.get("DEFAULT_IMGSZ", "1280"))
BENCHMARK_MS = os.environ.get("BENCHMARK_MS", "12.8")
STUDENT_NAME = os.environ.get("STUDENT_NAME", "Aditya Dhayapulay")
INSTITUTION = os.environ.get("INSTITUTION", "California State University, Fullerton")
CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL", "aditdhayapulay@gmail.com")

# Reference metrics shown in About (from saved YOLO_PCB unified eval artifacts).
REFERENCE_METRICS = {
    "dataset": "YOLO_PCB (5,551 / 1,016 / 1,016 train / val / test)",
    "test_map50": "0.902",
    "test_map50_95": "0.502",
    "test_recall": "0.866",
    "v100_inference_ms": BENCHMARK_MS,
}

APP_CSS = """
html, body, .gradio-container {
    background: #f4f7f5 !important;
    color: #17251f !important;
}

*::selection {
    background: #dff3ec !important;
    color: #10251f !important;
}

*::-moz-selection {
    background: #dff3ec !important;
    color: #10251f !important;
}

.gradio-container [role="tablist"] {
    border-bottom-color: #4b5f58 !important;
}

.gradio-container [role="tab"],
.gradio-container button[role="tab"] {
    color: #31443c !important;
    opacity: 1 !important;
    font-weight: 700 !important;
    background: transparent !important;
    user-select: none;
}

.gradio-container [role="tab"] *,
.gradio-container button[role="tab"] * {
    color: inherit !important;
    opacity: 1 !important;
}

.gradio-container [role="tab"][aria-selected="true"],
.gradio-container button[role="tab"][aria-selected="true"] {
    color: #0f766e !important;
    border-bottom-color: #14b8a6 !important;
}

.gradio-container [role="tab"][aria-selected="false"],
.gradio-container button[role="tab"][aria-selected="false"] {
    color: #31443c !important;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 20px 18px 36px !important;
}

#hero {
    background: linear-gradient(135deg, #0f2f29 0%, #0f766e 55%, #5b8c4a 140%);
    border-radius: 14px;
    padding: 28px 30px 22px;
    margin-bottom: 14px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: 0 18px 36px rgba(15, 63, 53, 0.16);
}

#hero, #hero * { color: #ffffff !important; }

#hero h1 {
    font-size: 31px !important;
    line-height: 1.15 !important;
    margin: 0 0 8px !important;
}

#hero p {
    font-size: 14px !important;
    opacity: 0.92;
    margin: 0 !important;
    max-width: 820px;
}

#hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 14px;
}

.badge {
    background: rgba(255, 255, 255, 0.14);
    border: 1px solid rgba(255, 255, 255, 0.22);
    border-radius: 999px;
    color: #ffffff;
    font-size: 12px;
    font-weight: 700;
    padding: 5px 11px;
}

#model-status {
    margin-bottom: 14px;
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 13px;
    font-weight: 600;
}

#model-status,
#model-status * {
    color: #065f46 !important;
}

#model-status.ok {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    color: #065f46;
}

#model-status.error {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b;
}

#verdict-banner {
    border-radius: 10px;
    padding: 12px 14px;
    margin: 0 0 12px;
    font-size: 14px;
    font-weight: 700;
}

#verdict-banner.pass {
    background: #ecfdf5;
    border: 1px solid #6ee7b7;
    color: #065f46;
}

#verdict-banner.review {
    background: #fffbeb;
    border: 1px solid #fcd34d;
    color: #92400e;
}

.panel {
    background: #ffffff !important;
    border: 1px solid #dfe8e3 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    box-shadow: 0 10px 28px rgba(26, 48, 40, 0.07) !important;
}

.panel label, .panel span, .panel p, .panel textarea,
.panel input, .panel button, .panel th, .panel td {
    color: #17251f !important;
}

.section-title h3 {
    margin: 0 !important;
    color: #17251f !important;
    font-size: 17px !important;
}

.section-title p {
    margin: 4px 0 0 !important;
    color: #5d6d65 !important;
    font-size: 13px !important;
}

#input-image, #output-image {
    border-radius: 10px !important;
    overflow: hidden !important;
}

#input-image .image-container, #output-image .image-container {
    background: #eef3f1 !important;
}

#status-card {
    background: #eef7f2 !important;
    border: 1px solid #cfe3d9 !important;
    border-radius: 10px !important;
    padding: 12px 14px !important;
}

#status-card, #status-card * { color: #16352f !important; }

#metric-cards {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin: 10px 0 12px;
}

.metric-card {
    background: #f7fbf9;
    border: 1px solid #d7e7df;
    border-radius: 10px;
    padding: 10px 12px;
}

.metric-card .label {
    color: #405a50 !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.metric-card .value {
    color: #12352e;
    font-size: 22px;
    font-weight: 800;
    line-height: 1.1;
    margin-top: 4px;
}

.metric-card .sub {
    color: #51675f !important;
    font-size: 11px;
    margin-top: 2px;
}

#class-legend {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 8px;
    margin: 8px 0 2px;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 8px;
    background: #f8fbf9;
    border: 1px solid #e3ece7;
    border-radius: 8px;
    padding: 7px 9px;
    font-size: 12px;
    color: #17352f;
}

.legend-swatch {
    width: 12px;
    height: 12px;
    border-radius: 3px;
    flex-shrink: 0;
}

#detection-preview {
    background: #ffffff !important;
    border: 1px solid #d9e5df !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

#detection-preview .empty-preview {
    color: #5d6d65;
    padding: 14px 16px;
    font-size: 13px;
}

#detection-preview .preview-scroll {
    max-height: 280px;
    overflow: auto;
}

#detection-preview table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    table-layout: fixed;
}

#detection-preview th {
    background: #12352e;
    color: #ffffff;
    font-weight: 700;
    padding: 9px 8px;
    position: sticky;
    top: 0;
    text-align: left;
    z-index: 1;
}

#detection-preview td {
    border-bottom: 1px solid #e6eee9;
    color: #17251f;
    padding: 8px;
    overflow-wrap: anywhere;
}

#detection-preview tr:nth-child(even) td { background: #f5faf7; }

#detection-preview .preview-note {
    color: #5d6d65;
    font-size: 12px;
    margin: 9px 10px 10px;
}

#class-breakdown { margin-top: 8px; }

#class-breakdown .bar-row {
    display: grid;
    grid-template-columns: 120px 1fr 34px;
    gap: 8px;
    align-items: center;
    margin-bottom: 6px;
    font-size: 12px;
    color: #17352f;
}

#class-breakdown .bar-row span {
    color: #17352f !important;
    font-weight: 600;
}

#class-breakdown .bar-track {
    background: #e8f0ec;
    border-radius: 999px;
    height: 8px;
    overflow: hidden;
}

#class-breakdown .bar-fill {
    height: 100%;
    border-radius: 999px;
}

#sample-buttons button, #action-row button {
    min-height: 38px !important;
    border-radius: 9px !important;
}

#sample-buttons button {
    background: #eef7f2 !important;
    border: 1px solid #cfe3d9 !important;
    color: #164139 !important;
    font-weight: 700 !important;
}

button.primary {
    background: #0f766e !important;
    border: 1px solid #0f766e !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 10px !important;
    min-height: 46px !important;
    box-shadow: 0 10px 18px rgba(15, 118, 110, 0.22) !important;
}

button.primary:hover {
    background: #115e59 !important;
    border-color: #115e59 !important;
}

#sample-note, .muted-note {
    color: #31443c !important;
    font-size: 13px !important;
    margin: 6px 0 2px !important;
    opacity: 1 !important;
    user-select: none;
}

#sample-note *,
.muted-note * {
    color: #31443c !important;
    opacity: 1 !important;
}

.about-panel h3 {
    color: #12352e !important;
    margin-top: 0 !important;
}

.about-panel,
.about-panel *,
.about-panel p,
.about-panel li,
.about-panel strong {
    color: #31443c !important;
    opacity: 1 !important;
}

.about-panel p, .about-panel li, .about-panel strong {
    font-size: 14px !important;
    line-height: 1.55 !important;
}

.about-panel a {
    color: #0f766e !important;
    font-weight: 700 !important;
}

.about-panel table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    margin: 10px 0 14px;
}

.about-panel th, .about-panel td {
    border: 1px solid #dbe6e0;
    color: #17352f !important;
    padding: 8px 10px;
    text-align: left;
}

.about-panel td {
    background: #ffffff !important;
}

.about-panel th {
    background: #e7f3ee !important;
    color: #12352e !important;
}

footer { display: none !important; }

@media (max-width: 900px) {
    #metric-cards { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    #class-legend { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}

@media (max-width: 760px) {
    .gradio-container { padding: 16px 12px 28px !important; }
    #hero { padding: 22px 18px; }
    #hero h1 { font-size: 25px !important; }
    #metric-cards, #class-legend { grid-template-columns: 1fr; }
}
"""

READY_MESSAGE = (
    "**Ready.** Upload a clean PCB image or choose one of the clean sample inputs."
)
DETECTION_COLUMNS = ("class", "confidence", "x1", "y1", "x2", "y2")
MAX_PREVIEW_ROWS = 15
MAX_DISPLAY_SIDE = 1400
DOWNLOAD_DIR = Path(tempfile.gettempdir()) / "pcb_capstone_downloads"
DOWNLOAD_RETENTION_SECONDS = 60 * 60

SAMPLE_IMAGES = [
    ("Missing hole", "examples/clean_missing_hole.jpg"),
    ("Mouse bite", "examples/clean_mouse_bite.jpg"),
    ("Open circuit", "examples/clean_open_circuit.jpg"),
    ("Short", "examples/clean_short.jpg"),
    ("Spur", "examples/clean_spur.jpg"),
    ("Spurious copper", "examples/clean_spurious_copper.jpg"),
]

def event_kwargs(*, api_name: str | None = None) -> dict:
    """Gradio 5/6 compatible event listener options."""
    if GRADIO_MAJOR >= 6:
        return {"api_name": api_name, "api_visibility": "undocumented"}
    return {"api_name": api_name, "show_api": False}


def blocks_kwargs() -> dict:
    theme = gr.themes.Soft(primary_hue="teal", neutral_hue="slate")
    if GRADIO_MAJOR >= 6:
        return {}
    return {"theme": theme, "css": APP_CSS}


def launch_kwargs() -> dict:
    theme = gr.themes.Soft(primary_hue="teal", neutral_hue="slate")
    kwargs = {"server_name": "0.0.0.0"}
    if GRADIO_MAJOR >= 6:
        kwargs.update({"theme": theme, "css": APP_CSS})
    return kwargs


def find_model_path() -> Path | None:
    for candidate in MODEL_CANDIDATES:
        if candidate and candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def load_model() -> tuple[YOLO | None, str, str]:
    model_path = find_model_path()
    if model_path is None:
        return None, "unavailable", (
            "Model file not found. Add the trained detector as "
            "`online_deployment/models/best.pt` or `online_deployment/models/best.onnx`."
        )

    model = YOLO(str(model_path))
    backend = "ONNX Runtime" if model_path.suffix.lower() == ".onnx" else "PyTorch"
    return model, backend, f"{MODEL_LABEL} ({model_path.name}, {backend})"


def runtime_device_label() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "Apple MPS"
    except Exception:
        pass
    return "CPU"


def render_model_status() -> str:
    model_path = find_model_path()
    if model_path is None:
        return (
            '<div id="model-status" class="error">'
            "Model not found. Place <code>best.pt</code> or <code>best.onnx</code> in "
            "<code>models/</code> before running detection."
            "</div>"
        )
    _, backend, status = load_model()
    device = runtime_device_label()
    return (
        f'<div id="model-status" class="ok">'
        f"Model ready: {escape(status)} · device: {escape(device)} · backend: {escape(backend)}"
        f"</div>"
    )


def render_verdict(rows: list[dict]) -> str:
    if not rows:
        return (
            '<div id="verdict-banner" class="pass">'
            "No defect candidates above the selected threshold."
            "</div>"
        )
    classes = sorted({display_class_name(row["class"]) for row in rows})
    class_text = ", ".join(classes[:4])
    if len(classes) > 4:
        class_text += f" (+{len(classes) - 4} more)"
    return (
        '<div id="verdict-banner" class="review">'
        f"Review recommended: {len(rows)} defect candidate(s) detected "
        f"({escape(class_text)})."
        "</div>"
    )


def display_class_name(class_name: str) -> str:
    return CLASS_DISPLAY_NAMES.get(str(class_name), str(class_name).replace("_", " "))


def canonical_class_name(class_name: str) -> str:
    raw_name = str(class_name)
    if raw_name in CLASS_DISPLAY_NAMES:
        return raw_name
    return DISPLAY_TO_CLASS.get(raw_name, raw_name.replace(" ", "_"))


def canonical_class_filter(class_filter: list[str] | None) -> set[str]:
    return {canonical_class_name(class_name) for class_name in (class_filter or [])}


def render_hero() -> str:
    return f"""
<div id="hero">
  <p><strong>CSUF Master's Capstone — Online Deployment Demo</strong></p>
  <h1>Automated PCB Defect Detection</h1>
  <p>
    Upload a PCB inspection image and run the exported <strong>{escape(MODEL_LABEL)}</strong>
    detector for six defect classes. The demo returns annotated boxes, per-class counts,
    latency, and downloadable results for presentation or QA review.
  </p>
  <div id="hero-badges">
    <span class="badge">6 defect classes</span>
    <span class="badge">{escape(MODEL_LABEL)} @ {DEFAULT_IMGSZ}px</span>
    <span class="badge">PyTorch + ONNX export</span>
    <span class="badge">Gradio web demo</span>
  </div>
</div>
"""


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 16)
    except OSError:
        return ImageFont.load_default()


def draw_detections(image: Image.Image, rows: list[dict]) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    font = _font()

    for row in rows:
        x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
        label = f"{display_class_name(row['class'])} {row['confidence']:.2f}"
        color = COLORS.get(row["class"], "#FF0000")

        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        text_bbox = draw.textbbox((x1, y1), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]
        label_y = max(0, y1 - text_h - 6)
        draw.rectangle((x1, label_y, x1 + text_w + 8, label_y + text_h + 6), fill=color)
        draw.text((x1 + 4, label_y + 3), label, fill="white", font=font)

    return annotated


def class_name_for(model: YOLO, class_id: int) -> str:
    names = getattr(model, "names", {})
    if isinstance(names, dict):
        return names.get(class_id, CLASS_NAMES.get(class_id, str(class_id)))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return CLASS_NAMES.get(class_id, str(class_id))


def resize_for_browser(image: Image.Image, max_side: int = MAX_DISPLAY_SIDE) -> Image.Image:
    width, height = image.size
    longest_side = max(width, height)
    if longest_side <= max_side:
        return image

    scale = max_side / longest_side
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def blank_detection_preview() -> str:
    return '<div class="empty-preview">No detections yet.</div>'


def render_metric_cards(
    *,
    detections: int,
    elapsed_ms: float,
    image_size: int,
    classes_found: int,
) -> str:
    return f"""
    <div id="metric-cards">
        <div class="metric-card">
            <div class="label">Detections</div>
            <div class="value">{detections}</div>
            <div class="sub">above threshold</div>
        </div>
        <div class="metric-card">
            <div class="label">Latency</div>
            <div class="value">{elapsed_ms:.0f}</div>
            <div class="sub">milliseconds</div>
        </div>
        <div class="metric-card">
            <div class="label">Input size</div>
            <div class="value">{image_size}</div>
            <div class="sub">pixels</div>
        </div>
        <div class="metric-card">
            <div class="label">Classes hit</div>
            <div class="value">{classes_found}</div>
            <div class="sub">of 6 defect types</div>
        </div>
    </div>
    """


def render_class_legend() -> str:
    items = []
    for class_name, color in COLORS.items():
        label = display_class_name(class_name)
        items.append(
            f'<div class="legend-item"><span class="legend-swatch" style="background:{color}"></span>{escape(label)}</div>'
        )
    return f'<div id="class-legend">{"".join(items)}</div>'


def render_class_breakdown(rows: list[dict]) -> str:
    if not rows:
        return '<div class="muted-note">Class breakdown appears after detection.</div>'

    counts = Counter(row["class"] for row in rows)
    max_count = max(counts.values())
    bars = []
    for class_name in CLASS_NAMES.values():
        count = counts.get(class_name, 0)
        width = int((count / max_count) * 100) if max_count else 0
        color = COLORS.get(class_name, "#999999")
        label = display_class_name(class_name)
        bars.append(
            f"""
            <div class="bar-row">
                <span>{escape(label)}</span>
                <div class="bar-track"><div class="bar-fill" style="width:{width}%; background:{color};"></div></div>
                <span>{count}</span>
            </div>
            """
        )
    return f'<div id="class-breakdown">{"".join(bars)}</div>'


def render_detection_preview(rows: list[dict], class_filter: list[str] | None = None) -> str:
    if class_filter:
        allowed_classes = canonical_class_filter(class_filter)
        rows = [row for row in rows if canonical_class_name(row["class"]) in allowed_classes]

    if not rows:
        return '<div class="empty-preview">No defect candidates above the selected threshold.</div>'

    sorted_rows = sorted(rows, key=lambda row: row["confidence"], reverse=True)
    visible_rows = sorted_rows[:MAX_PREVIEW_ROWS]
    header = "".join(f"<th>{escape(column)}</th>" for column in DETECTION_COLUMNS)
    body = []

    for row in visible_rows:
        cells = [
            escape(display_class_name(row["class"])),
            f"{row['confidence']:.3f}",
            f"{row['x1']:.1f}",
            f"{row['y1']:.1f}",
            f"{row['x2']:.1f}",
            f"{row['y2']:.1f}",
        ]
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")

    note = ""
    if len(rows) > MAX_PREVIEW_ROWS:
        note = (
            f'<p class="preview-note">Showing top {MAX_PREVIEW_ROWS} of '
            f'{len(rows)} detections.</p>'
        )

    return (
        '<div class="preview-scroll"><table>'
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        f"</table>{note}</div>"
    )


def _prepare_download_dir() -> Path:
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time() - DOWNLOAD_RETENTION_SECONDS
    for old_file in DOWNLOAD_DIR.glob("pcb_detections_*"):
        try:
            if old_file.is_file() and old_file.stat().st_mtime < cutoff:
                old_file.unlink()
        except OSError:
            pass
    return DOWNLOAD_DIR


def write_download_artifacts(rows: list[dict], annotated: Image.Image | None) -> tuple[str | None, str | None, str | None]:
    download_dir = _prepare_download_dir()
    run_id = uuid4().hex[:12]
    image_path = None
    json_path = None
    csv_path = None

    if annotated is not None:
        image_path = str(download_dir / f"pcb_detections_{run_id}.jpg")
        annotated.save(image_path, format="JPEG", quality=92)

    payload = {
        "model": MODEL_LABEL,
        "classes": CLASS_FILTER_CHOICES,
        "detections": [
            {**row, "class": display_class_name(row["class"])} for row in rows
        ],
        "count": len(rows),
    }
    json_path = str(download_dir / f"pcb_detections_{run_id}.json")
    Path(json_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = str(download_dir / f"pcb_detections_{run_id}.csv")
    csv_lines = ["class,confidence,x1,y1,x2,y2"]
    for row in rows:
        csv_lines.append(
            f"{display_class_name(row['class'])},{row['confidence']:.4f},{row['x1']},{row['y1']},{row['x2']},{row['y2']}"
        )
    Path(csv_path).write_text("\n".join(csv_lines) + "\n", encoding="utf-8")

    return image_path, json_path, csv_path


def empty_predict_result(image_size: int = DEFAULT_IMGSZ):
    return (
        None,
        READY_MESSAGE,
        blank_detection_preview(),
        render_metric_cards(detections=0, elapsed_ms=0.0, image_size=image_size, classes_found=0),
        render_class_breakdown([]),
        render_verdict([]),
        [],
        None,
        None,
        None,
        [],
        None,
        0.0,
    )


def clear_workspace():
    return (None,) + empty_predict_result()


def predict(
    image: Image.Image | None,
    confidence: float,
    iou: float,
    image_size: int,
    class_filter: list[str],
    cached_rows: list[dict] | None = None,
    cached_annotated: Image.Image | None = None,
    cached_elapsed_ms: float = 0.0,
    rerun: bool = True,
):
    if image is None:
        return empty_predict_result(image_size)

    rows = cached_rows or []
    annotated = cached_annotated
    elapsed_ms = cached_elapsed_ms
    model_line = MODEL_LABEL

    if rerun:
        model, backend, status = load_model()
        model_line = status
        if model is None:
            return (
                None,
                f"**Model unavailable.** {status}",
                blank_detection_preview(),
                render_metric_cards(detections=0, elapsed_ms=0.0, image_size=image_size, classes_found=0),
                render_class_breakdown([]),
                render_verdict([]),
                [],
                None,
                None,
                None,
                [],
                None,
                0.0,
            )

        try:
            start = perf_counter()
            results = model.predict(
                source=image.convert("RGB"),
                imgsz=int(image_size),
                conf=float(confidence),
                iou=float(iou),
                verbose=False,
            )
            elapsed_ms = (perf_counter() - start) * 1000

            rows = []
            result = results[0]
            boxes = result.boxes

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    class_id = int(box.cls.item())
                    class_name = class_name_for(model, class_id)
                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    rows.append(
                        {
                            "class": class_name,
                            "confidence": float(box.conf.item()),
                            "x1": round(x1, 1),
                            "y1": round(y1, 1),
                            "x2": round(x2, 1),
                            "y2": round(y2, 1),
                        }
                    )

            annotated = draw_detections(image, rows)
            annotated = resize_for_browser(annotated)
            model_line = f"{MODEL_LABEL} ({backend}) on {runtime_device_label()}"
        except Exception as exc:
            err = escape(str(exc))
            trace = escape(traceback.format_exc(limit=2))
            return (
                None,
                f"**Detection failed.** {err}\n\n```\n{trace}\n```",
                blank_detection_preview(),
                render_metric_cards(detections=0, elapsed_ms=0.0, image_size=image_size, classes_found=0),
                render_class_breakdown([]),
                render_verdict([]),
                [],
                None,
                None,
                None,
                [],
                None,
                0.0,
            )

    display_image = resize_for_browser(image)
    detection_preview = render_detection_preview(rows, class_filter or None)
    class_counts = Counter(row["class"] for row in rows)
    classes_found = len(class_counts)

    if rows:
        top_classes = ", ".join(
            f"{display_class_name(name)}: {count}"
            for name, count in class_counts.most_common(3)
        )
    else:
        top_classes = "No defect candidates above the selected threshold"

    summary = (
        f"**{model_line}**  \n"
        f"**Detections:** {len(rows)} candidate(s)  \n"
        f"**Inference time:** {elapsed_ms:.1f} ms  \n"
        f"**Top classes:** {top_classes}"
    )

    image_download, json_download, csv_download = write_download_artifacts(rows, annotated)
    comparison = [display_image, annotated] if annotated is not None else []

    return (
        annotated,
        summary,
        detection_preview,
        render_metric_cards(
            detections=len(rows),
            elapsed_ms=elapsed_ms,
            image_size=image_size,
            classes_found=classes_found,
        ),
        render_class_breakdown(rows),
        render_verdict(rows),
        comparison,
        image_download,
        json_download,
        csv_download,
        rows,
        annotated,
        elapsed_ms,
    )


def filter_existing_results(
    rows: list[dict] | None,
    annotated: Image.Image | None,
    elapsed_ms: float,
    image: Image.Image | None,
    confidence: float,
    iou: float,
    image_size: int,
    class_filter: list[str],
):
    return predict(
        image=image,
        confidence=confidence,
        iou=iou,
        image_size=image_size,
        class_filter=class_filter,
        cached_rows=rows or [],
        cached_annotated=annotated,
        cached_elapsed_ms=elapsed_ms,
        rerun=False,
    )


def load_sample_inputs(sample_file: str):
    if not Path(sample_file).exists():
        raise gr.Error(f"Sample image not found: {sample_file}")
    image = Image.open(sample_file).convert("RGB")
    image = resize_for_browser(image)
    return image, 0.25, 0.45, DEFAULT_IMGSZ, CLASS_FILTER_CHOICES


def load_sample_and_detect(sample_file: str):
    image, confidence, iou, image_size, class_filter = load_sample_inputs(sample_file)
    predict_outputs = predict(image, confidence, iou, image_size, class_filter, rerun=True)
    return (image, confidence, iou, image_size, class_filter, *predict_outputs)


def build_about_markdown() -> str:
    _, backend, model_status = load_model()
    device = runtime_device_label()
    return f"""
### Capstone deployment overview

This Hugging Face Space is the **online deployment artifact** for the CSUF master's capstone project on automated PCB defect inspection. It demonstrates that the trained detector can be exported and served through a browser-based workflow suitable for manufacturing triage demos.

**Student:** {STUDENT_NAME}<br>
**Institution:** {INSTITUTION}<br>
**Contact:** {CONTACT_EMAIL}<br>
**Runtime model:** {model_status}<br>
**Inference device:** {device}

### What the demo does

1. Accept a clean uploaded PCB inspection image or bundled raw PCB sample.
2. Run the exported detector with adjustable confidence and NMS thresholds.
3. Return an annotated image, per-detection table, class breakdown, inspection verdict, and downloadable results.

### Reference offline metrics

These numbers come from the saved YOLO_PCB unified evaluation used in the capstone report. Space latency will differ from the offline V100 GPU benchmark.

| Metric | Value |
|---|---|
| Dataset | {REFERENCE_METRICS['dataset']} |
| Test mAP50 | {REFERENCE_METRICS['test_map50']} |
| Test mAP50-95 | {REFERENCE_METRICS['test_map50_95']} |
| Test recall | {REFERENCE_METRICS['test_recall']} |
| V100 batch-1 inference latency | {REFERENCE_METRICS['v100_inference_ms']} ms/image |

### Defect classes

| Class | Typical appearance |
|---|---|
| Missing hole | Drill hole absent from pad region |
| Mouse bite | Small edge erosion on copper trace |
| Open circuit | Broken or interrupted conductor |
| Short | Unintended bridge between traces |
| Spur | Thin protrusion from a trace |
| Spurious copper | Extra copper not in the design |

### What this demo claims

- The project trained a YOLO-family detector for six PCB defect classes.
- The best checkpoint was exported to **PyTorch (`best.pt`)** and **ONNX (`best.onnx`)** for deployment evidence.
- The online demo satisfies the **web deployment requirement** for the capstone presentation.

### What this demo does not claim

- Embedded Jetson / TensorRT production latency (hardware access was unavailable).
- Full factory-line integration or live camera streaming.
- State-of-the-art benchmark leadership across all public PCB datasets.

### Reproducibility

Artifacts, metrics tables, and training logs are archived in the project repository and Kaggle/Nautilus export bundles used to build the final ESCS paper tables.

For additional manual tests, use clean raw PCB images from the public [PCB-DATASET image folders]({DATASET_IMAGE_URL}). Avoid annotated mosaics, screenshots, or files with text overlays.
"""


with gr.Blocks(title="Automated PCB Defect Detection", **blocks_kwargs()) as demo:
    rows_state = gr.State([])
    annotated_state = gr.State(None)
    elapsed_state = gr.State(0.0)

    gr.HTML(render_hero())
    model_status = gr.HTML(render_model_status())

    with gr.Tabs():
        with gr.Tab("Detect"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1, elem_classes="panel"):
                    gr.Markdown(
                        (
                            "### Input\nUpload a clean PCB image or load a clean sample. Do not use screenshots, YOLO "
                            "validation mosaics, or images that already contain text labels."
                        ),
                        elem_classes="section-title",
                    )
                    image_input = gr.Image(
                        type="pil",
                        label="Upload PCB image",
                        height=320,
                        elem_id="input-image",
                        format="jpeg",
                        sources=["upload", "clipboard"],
                    )
                    confidence = gr.Slider(
                        minimum=0.05,
                        maximum=0.95,
                        value=0.25,
                        step=0.05,
                        label="Confidence threshold",
                    )
                    iou = gr.Slider(
                        minimum=0.20,
                        maximum=0.90,
                        value=0.45,
                        step=0.05,
                        label="NMS IoU threshold",
                    )
                    image_size = gr.Dropdown(
                        choices=[640, 960, 1280],
                        value=DEFAULT_IMGSZ,
                        label="Inference image size",
                    )
                    class_filter = gr.CheckboxGroup(
                        choices=CLASS_FILTER_CHOICES,
                        value=CLASS_FILTER_CHOICES,
                        label="Show classes in results table",
                    )

                    with gr.Row(elem_id="action-row"):
                        detect_button = gr.Button("Run detection", variant="primary", scale=2)
                        clear_button = gr.Button("Clear", scale=1)

                    gr.Markdown("Clean sample inputs (auto-run)", elem_id="sample-note")
                    sample_buttons = []
                    with gr.Row(elem_id="sample-buttons"):
                        for sample_label, _sample_file in SAMPLE_IMAGES:
                            sample_buttons.append(gr.Button(sample_label))
                    gr.Markdown(
                        (
                            "Need more test images? Browse the public "
                            f"[PCB-DATASET image folders]({DATASET_IMAGE_URL}) and upload a clean "
                            "raw `.jpg` from an `images/<class>/` folder."
                        ),
                        elem_id="dataset-link",
                    )

                    gr.HTML(render_class_legend())

                with gr.Column(scale=1, elem_classes="panel"):
                    gr.Markdown(
                        "### Output\nAnnotated result, metrics, and exports.",
                        elem_classes="section-title",
                    )
                    verdict_banner = gr.HTML(render_verdict([]))
                    image_output = gr.Image(
                        type="pil",
                        label="Annotated detections",
                        height=320,
                        elem_id="output-image",
                        format="jpeg",
                    )
                    metric_cards = gr.HTML(
                        render_metric_cards(
                            detections=0,
                            elapsed_ms=0.0,
                            image_size=DEFAULT_IMGSZ,
                            classes_found=0,
                        )
                    )
                    summary_output = gr.Markdown(READY_MESSAGE, elem_id="status-card")
                    class_breakdown = gr.HTML(render_class_breakdown([]))
                    detection_preview = gr.HTML(blank_detection_preview(), elem_id="detection-preview")

                    comparison_gallery = gr.Gallery(
                        label="Before / after comparison (original, annotated)",
                        columns=2,
                        height=280,
                        object_fit="contain",
                    )
                    with gr.Row():
                        image_download = gr.DownloadButton("Download annotated image", value=None)
                        json_download = gr.DownloadButton("Download JSON results", value=None)
                        csv_download = gr.DownloadButton("Download CSV results", value=None)

        with gr.Tab("About"):
            gr.Markdown(build_about_markdown(), elem_classes="about-panel")

    predict_outputs = [
        image_output,
        summary_output,
        detection_preview,
        metric_cards,
        class_breakdown,
        verdict_banner,
        comparison_gallery,
        image_download,
        json_download,
        csv_download,
        rows_state,
        annotated_state,
        elapsed_state,
    ]

    detect_button.click(
        fn=predict,
        inputs=[image_input, confidence, iou, image_size, class_filter],
        outputs=predict_outputs,
        **event_kwargs(api_name="predict"),
    )

    class_filter.change(
        fn=filter_existing_results,
        inputs=[
            rows_state,
            annotated_state,
            elapsed_state,
            image_input,
            confidence,
            iou,
            image_size,
            class_filter,
        ],
        outputs=predict_outputs,
        **event_kwargs(),
    )

    clear_button.click(
        fn=clear_workspace,
        inputs=None,
        outputs=[image_input, *predict_outputs],
        **event_kwargs(),
    )

    for sample_button, (_, sample_file) in zip(sample_buttons, SAMPLE_IMAGES):
        sample_button.click(
            fn=lambda sf=sample_file: load_sample_and_detect(sf),
            inputs=None,
            outputs=[
                image_input,
                confidence,
                iou,
                image_size,
                class_filter,
                *predict_outputs,
            ],
            **event_kwargs(),
        )


if __name__ == "__main__":
    demo.launch(**launch_kwargs())
