#!/usr/bin/env bash
# Install YOLO_PCB from a manual zip (Kaggle Output download), fix data.yaml paths.
# Usage: bash tools/install_yolo_pcb_from_zip.sh ~/data/yolo_pcb_dataset.zip
set -euo pipefail

ZIP="${1:?Usage: $0 /path/to/yolo_pcb_dataset.zip}"
DATA_ROOT="${DATA_ROOT:-$HOME/data}"
DEST="${DEST:-$DATA_ROOT/YOLO_PCB}"
TMP="${DATA_ROOT}/.yolo_pcb_unzip_$$"

mkdir -p "$DATA_ROOT"
rm -rf "$TMP"
mkdir -p "$TMP"
unzip -q -o "$ZIP" -d "$TMP"

ROOT=""
if [[ -d "$TMP/YOLO_PCB" ]]; then
  ROOT="$TMP/YOLO_PCB"
elif [[ -d "$TMP/train" ]]; then
  ROOT="$TMP"
else
  echo "Could not find YOLO_PCB layout in zip. Contents:"
  find "$TMP" -maxdepth 2 -type d
  exit 1
fi

rm -rf "$DEST"
mkdir -p "$DEST"
cp -a "$ROOT/." "$DEST/"

python3 - <<PY
from pathlib import Path

dest = Path("${DEST}").expanduser().resolve()
for split in ("train", "val", "test"):
    imgs = dest / split / "images"
    n = len([p for p in imgs.iterdir() if p.is_file()]) if imgs.is_dir() else 0
    print(f"{split}: {n} images")

(dest / "data.yaml").write_text(
    f"path: {dest}\n"
    "train: train/images\n"
    "val: val/images\n"
    "test: test/images\n"
    "names:\n"
    "  0: Missing_hole\n"
    "  1: Mouse_bite\n"
    "  2: Open_circuit\n"
    "  3: Short\n"
    "  4: Spur\n"
    "  5: Spurious_copper\n"
)
train_n = len(list((dest / "train/images").glob("*")))
if train_n < 5000:
    raise SystemExit(f"ERROR: train has {train_n} images (expected ~5551). Wrong zip?")
print("OK:", dest / "data.yaml")
PY

rm -rf "$TMP"
echo "export KAGGLE_YAML=$DEST/data.yaml"
