#!/usr/bin/env bash
# Download prepared YOLO_PCB from Kaggle and fix data.yaml for Nautilus paths.
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/data}"
DATASET_SLUG="${DATASET_SLUG:-aditya2402/pcb-yolo-prepared}"
DEST="${DEST:-$DATA_ROOT/YOLO_PCB}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Installing kaggle CLI..."
  python3 -m pip install -q --user kaggle
  export PATH="$HOME/.local/bin:$PATH"
fi

if [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
  echo "Missing ~/.kaggle/kaggle.json on this machine."
  echo "Copy your Kaggle API token from https://www.kaggle.com/settings"
  exit 1
fi

mkdir -p "$DATA_ROOT"
TMP="$DATA_ROOT/.yolo_pcb_download_$$"
mkdir -p "$TMP"

echo "Downloading $DATASET_SLUG ..."
kaggle datasets download -d "$DATASET_SLUG" -p "$TMP" --unzip

# Find extracted root (zip may contain YOLO_PCB/ or flat train/val/test)
ROOT=""
if [[ -d "$TMP/YOLO_PCB" ]]; then
  ROOT="$TMP/YOLO_PCB"
elif [[ -f "$TMP/data.yaml" ]] || [[ -d "$TMP/train" ]]; then
  ROOT="$TMP"
else
  echo "Could not find YOLO_PCB layout under $TMP"
  ls -la "$TMP"
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
    if not imgs.is_dir():
        raise SystemExit(f"Missing {imgs}")
    n = len([p for p in imgs.iterdir() if p.is_file()])
    print(f"{split}: {n} images")

text = (
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
(dest / "data.yaml").write_text(text)
print("Wrote", dest / "data.yaml")
train_n = len(list((dest / "train/images").glob("*")))
if train_n < 1000:
    print("WARNING: train images < 1000 — likely wrong dataset (expected ~4751).")
PY

rm -rf "$TMP"
echo ""
echo "Ready: export KAGGLE_YAML=$DEST/data.yaml"
