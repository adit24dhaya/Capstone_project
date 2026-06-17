#!/usr/bin/env python3
"""Generate class-balanced success and failure figures from saved predictions."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

workspace_tools = Path("/workspace/repo/tools")
if workspace_tools.is_dir():
    sys.path.insert(0, str(workspace_tools))

from prepare_resolution_control_dataset import CANONICAL_CLASSES
from run_nautilus_experiments import (
    box_iou,
    load_ultralytics_model,
    load_yolo_items_from_data_yaml,
)

COLORS = [
    (20, 184, 166),
    (37, 99, 235),
    (239, 68, 68),
    (245, 158, 11),
    (139, 92, 246),
    (236, 72, 153),
]


def named_counts(counts: Counter) -> dict[str, int]:
    return {
        CANONICAL_CLASSES[class_id]: count
        for class_id, count in sorted(counts.items())
    }


def match_boxes(
    ground_truth_boxes: list[list[float]],
    ground_truth_labels: list[int],
    prediction_boxes: list[list[float]],
    prediction_labels: list[int],
    prediction_scores: list[float],
) -> dict:
    candidates = []
    for prediction_index, prediction_box in enumerate(prediction_boxes):
        for target_index, target_box in enumerate(ground_truth_boxes):
            if prediction_labels[prediction_index] != ground_truth_labels[target_index]:
                continue
            overlap = box_iou(prediction_box, target_box)
            if overlap >= 0.5:
                candidates.append(
                    (
                        overlap,
                        prediction_scores[prediction_index],
                        prediction_index,
                        target_index,
                    )
                )
    matched_predictions = set()
    matched_targets = set()
    for _, _, prediction_index, target_index in sorted(candidates, reverse=True):
        if (
            prediction_index in matched_predictions
            or target_index in matched_targets
        ):
            continue
        matched_predictions.add(prediction_index)
        matched_targets.add(target_index)
    matched_labels = Counter(
        ground_truth_labels[index] for index in matched_targets
    )
    false_positive_labels = Counter(
        prediction_labels[index]
        for index in range(len(prediction_boxes))
        if index not in matched_predictions
    )
    false_negative_labels = Counter(
        ground_truth_labels[index]
        for index in range(len(ground_truth_boxes))
        if index not in matched_targets
    )
    true_positives = len(matched_predictions)
    return {
        "true_positives": true_positives,
        "false_positives": len(prediction_boxes) - true_positives,
        "false_negatives": len(ground_truth_boxes) - true_positives,
        "true_positive_labels": matched_labels,
        "false_positive_labels": false_positive_labels,
        "false_negative_labels": false_negative_labels,
    }


def predict_items(model, items: list[dict], imgsz: int, confidence: float) -> list[dict]:
    evaluated = []
    for item in items:
        result = model.predict(
            source=str(item["image_path"]),
            imgsz=imgsz,
            conf=confidence,
            batch=1,
            device=0,
            verbose=False,
        )[0]
        boxes = result.boxes
        prediction_boxes = (
            boxes.xyxy.detach().cpu().tolist() if boxes is not None else []
        )
        prediction_labels = (
            [int(value) for value in boxes.cls.detach().cpu().tolist()]
            if boxes is not None
            else []
        )
        prediction_scores = (
            boxes.conf.detach().cpu().tolist() if boxes is not None else []
        )
        match_summary = match_boxes(
            item["boxes"],
            item["labels"],
            prediction_boxes,
            prediction_labels,
            prediction_scores,
        )
        evaluated.append(
            {
                **item,
                "prediction_boxes": prediction_boxes,
                "prediction_labels": prediction_labels,
                "prediction_scores": prediction_scores,
                **match_summary,
            }
        )
    return evaluated


def select_successes(evaluated: list[dict]) -> list[dict]:
    selected = []
    used_paths = set()
    for class_id in range(len(CANONICAL_CLASSES)):
        candidates = [
            item
            for item in evaluated
            if class_id in item["labels"]
            and item["true_positive_labels"][class_id] > 0
            and item["false_negative_labels"][class_id] == 0
            and item["false_negatives"] == 0
        ]
        candidates.sort(
            key=lambda item: (
                item["false_positives"],
                -item["true_positives"],
                str(item["image_path"]),
            )
        )
        choice = next(
            (
                item
                for item in candidates
                if str(item["image_path"]) not in used_paths
            ),
            candidates[0] if candidates else None,
        )
        if choice is not None:
            selected.append(choice)
            used_paths.add(str(choice["image_path"]))
    return selected


def select_failures(evaluated: list[dict], limit: int = 6) -> list[dict]:
    failures = [
        item
        for item in evaluated
        if item["false_negatives"] > 0 or item["false_positives"] > 0
    ]
    failures.sort(
        key=lambda item: (
            -(5 * item["false_negatives"] + item["false_positives"]),
            -item["false_negatives"],
            -item["false_positives"],
            str(item["image_path"]),
        )
    )
    selected = []
    represented_classes = set()
    for item in failures:
        item_classes = set(item["labels"])
        if item_classes - represented_classes or len(selected) >= len(
            CANONICAL_CLASSES
        ):
            selected.append(item)
            represented_classes.update(item_classes)
        if len(selected) == limit:
            break
    if len(selected) < limit:
        for item in failures:
            if item not in selected:
                selected.append(item)
            if len(selected) == limit:
                break
    return selected


def annotated_tile(item: dict, title: str, tile_size: tuple[int, int]):
    from PIL import Image, ImageDraw, ImageFont

    image = Image.open(item["image_path"]).convert("RGB")
    content_width = tile_size[0] - 20
    content_height = tile_size[1] - 72
    scale = min(content_width / image.width, content_height / image.height)
    resized_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    image = image.resize(resized_size, Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(image)
    try:
        label_font = ImageFont.truetype("DejaVuSans.ttf", 13)
        title_font = ImageFont.truetype("DejaVuSans.ttf", 14)
        footer_font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        label_font = title_font = footer_font = ImageFont.load_default()
    line_width = 2

    def scaled_box(box: list[float]) -> list[float]:
        return [coordinate * scale for coordinate in box]

    for box, class_id in zip(item["boxes"], item["labels"]):
        box = scaled_box(box)
        draw.rectangle(box, outline=(250, 204, 21), width=line_width)
        label = f"GT {CANONICAL_CLASSES[class_id]}"
        text_box = draw.textbbox((0, 0), label, font=label_font)
        text_width = text_box[2] - text_box[0] + 6
        text_height = text_box[3] - text_box[1] + 4
        x = min(max(0, box[0]), max(0, image.width - text_width))
        y = min(image.height - text_height, box[3])
        draw.rectangle((x, y, x + text_width, y + text_height), fill=(113, 63, 18))
        draw.text((x + 3, y + 2), label, fill="white", font=label_font)
    for box, class_id, score in zip(
        item["prediction_boxes"],
        item["prediction_labels"],
        item["prediction_scores"],
    ):
        box = scaled_box(box)
        color = COLORS[class_id % len(COLORS)]
        draw.rectangle(box, outline=color, width=line_width)
        label = f"{CANONICAL_CLASSES[class_id]} {score:.2f}"
        text_box = draw.textbbox((0, 0), label, font=label_font)
        text_width = text_box[2] - text_box[0] + 6
        text_height = text_box[3] - text_box[1] + 4
        x = min(max(0, box[0]), max(0, image.width - text_width))
        y = max(0, box[1] - text_height)
        draw.rectangle((x, y, x + text_width, y + text_height), fill=color)
        draw.text((x + 3, y + 2), label, fill="white", font=label_font)

    tile = Image.new("RGB", tile_size, "white")
    tile.paste(
        image,
        (
            (tile_size[0] - image.width) // 2,
            30 + (content_height - image.height) // 2,
        ),
    )
    tile_draw = ImageDraw.Draw(tile)
    tile_draw.text((10, 7), title, fill=(18, 55, 48), font=title_font)
    classes = ", ".join(
        CANONICAL_CLASSES[class_id] for class_id in sorted(set(item["labels"]))
    )
    footer = (
        f"GT yellow | predictions colored | TP {item['true_positives']} "
        f"FP {item['false_positives']} FN {item['false_negatives']} | {classes}"
    )
    tile_draw.text(
        (10, tile_size[1] - 20),
        footer,
        fill=(18, 55, 48),
        font=footer_font,
    )
    return tile


def save_grid(items: list[dict], output: Path, heading: str) -> None:
    from PIL import Image, ImageDraw, ImageFont

    tile_size = (720, 430)
    columns = 2
    rows = max(1, (len(items) + columns - 1) // columns)
    heading_height = 54
    canvas = Image.new(
        "RGB",
        (tile_size[0] * columns, tile_size[1] * rows + heading_height),
        (242, 247, 245),
    )
    draw = ImageDraw.Draw(canvas)
    try:
        heading_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        heading_font = ImageFont.load_default()
    draw.text((18, 15), heading, fill=(18, 55, 48), font=heading_font)
    for index, item in enumerate(items):
        tile = annotated_tile(
            item,
            Path(item["image_path"]).name,
            tile_size,
        )
        x = (index % columns) * tile_size[0]
        y = heading_height + (index // columns) * tile_size[1]
        canvas.paste(tile, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=95)


def write_manifest(path: Path, evaluated: list[dict], selected: dict[str, list[dict]]) -> None:
    selected_lookup = {
        str(item["image_path"]): kind
        for kind, items in selected.items()
        for item in items
    }
    rows = []
    for item in evaluated:
        ground_truth_counts = Counter(
            CANONICAL_CLASSES[class_id] for class_id in item["labels"]
        )
        prediction_counts = Counter(
            CANONICAL_CLASSES[class_id] for class_id in item["prediction_labels"]
        )
        rows.append(
            {
                "image": str(item["image_path"]),
                "selection": selected_lookup.get(str(item["image_path"]), ""),
                "ground_truth_boxes": len(item["boxes"]),
                "prediction_boxes": len(item["prediction_boxes"]),
                "true_positives": item["true_positives"],
                "false_positives": item["false_positives"],
                "false_negatives": item["false_negatives"],
                "ground_truth_classes": dict(ground_truth_counts),
                "prediction_classes": dict(prediction_counts),
                "true_positive_classes": named_counts(
                    item["true_positive_labels"]
                ),
                "false_positive_classes": named_counts(
                    item["false_positive_labels"]
                ),
                "false_negative_classes": named_counts(
                    item["false_negative_labels"]
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--engine", choices=("yolo", "rtdetr"), default="yolo")
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    items = load_yolo_items_from_data_yaml(args.data_yaml, "test")
    model = load_ultralytics_model(str(args.weights), engine=args.engine)
    evaluated = predict_items(model, items, args.imgsz, args.confidence)
    successes = select_successes(evaluated)
    failures = select_failures(evaluated)
    save_grid(
        successes,
        args.output_dir / "class_balanced_success_examples.png",
        "Class-balanced representative successes",
    )
    save_grid(
        failures,
        args.output_dir / "representative_failure_cases.png",
        "Representative failure cases at IoU >= 0.50",
    )
    write_manifest(
        args.output_dir / "prediction_example_manifest.csv",
        evaluated,
        {"success": successes, "failure": failures},
    )
    print(f"Saved figures and manifest to {args.output_dir}")


if __name__ == "__main__":
    main()
