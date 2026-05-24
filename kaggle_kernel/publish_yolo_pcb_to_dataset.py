# Paste this entire file into ONE Kaggle notebook cell AFTER YOLO_PCB prep finishes.
# Prerequisite: /kaggle/temp/YOLO_PCB exists with train/val/test and data.yaml

from pathlib import Path
import json
import shutil

YOLO_ROOT = Path("/kaggle/temp/YOLO_PCB")
WORK = Path("/kaggle/working")
ZIP_PATH = WORK / "yolo_pcb_dataset.zip"
META_DIR = WORK / "pcb_yolo_prepared_upload"
META_PATH = META_DIR / "dataset-metadata.json"

if not (YOLO_ROOT / "data.yaml").exists():
    raise FileNotFoundError(f"Missing {YOLO_ROOT / 'data.yaml'} — run dataset prep cells first.")

for split in ("train", "val", "test"):
    n = len(list((YOLO_ROOT / split / "images").glob("*")))
    print(f"{split}: {n} images")

# Rewrite data.yaml so paths work after unzip anywhere
names_block = "\n".join(
    f"  {i}: {name}"
    for i, name in enumerate(
        ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
    )
)
fixed_yaml = (
    "path: .\n"
    "train: train/images\n"
    "val: val/images\n"
    "test: test/images\n"
    f"names:\n{names_block}\n"
)
(WORK / "YOLO_PCB").mkdir(exist_ok=True)
staging = WORK / "YOLO_PCB"
if staging.exists():
  shutil.rmtree(staging)
shutil.copytree(YOLO_ROOT, staging)
(staging / "data.yaml").write_text(fixed_yaml)

if ZIP_PATH.exists():
    ZIP_PATH.unlink()
shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", staging.parent, staging.name)
size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
print(f"Created {ZIP_PATH} ({size_mb:.1f} MiB)")

META_DIR.mkdir(exist_ok=True)
META_PATH.write_text(
    json.dumps(
        {
            "title": "PCB YOLO Prepared (YOLO_PCB)",
            "id": "aditya2402/pcb-yolo-prepared",
            "licenses": [{"name": "CC0-1.0"}],
        },
        indent=2,
    )
    + "\n"
)
print(f"Wrote {META_PATH}")
print(
    "\nNext steps:\n"
    "1) Download yolo_pcb_dataset.zip from Output, OR\n"
    "2) On laptop: unzip, put zip in META_DIR, run:\n"
    "     kaggle datasets create -p pcb_yolo_prepared_upload\n"
    "   OR use Kaggle UI → New Dataset → upload the zip.\n"
    "3) On Nautilus: bash tools/fetch_yolo_pcb_on_nautilus.sh\n"
)
