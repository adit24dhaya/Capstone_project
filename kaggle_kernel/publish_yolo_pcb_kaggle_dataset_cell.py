# =============================================================================
# Paste this ONE cell on Kaggle AFTER yolo_pcb_dataset.zip exists (~1.1 GB).
# Publishes to: aditya2402/pcb-yolo-prepared
#
# ONE-TIME setup (Kaggle notebook sidebar):
#   Add-ons → Secrets → Add secret
#     KAGGLE_USERNAME = your Kaggle username
#     KAGGLE_KEY      = API key from https://www.kaggle.com/settings
# =============================================================================

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_SLUG = "aditya2402/pcb-yolo-prepared"
ZIP_PATH = Path("/kaggle/working/yolo_pcb_dataset.zip")
UPLOAD_DIR = Path("/kaggle/working/pcb_yolo_prepared_upload")

if not ZIP_PATH.exists():
    raise FileNotFoundError(
        f"Missing {ZIP_PATH}. Run the zip cell first."
    )

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])

try:
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
    os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")
except Exception as exc:
    raise RuntimeError(
        "Add KAGGLE_USERNAME and KAGGLE_KEY in Add-ons → Secrets, then re-run."
    ) from exc

if UPLOAD_DIR.exists():
    shutil.rmtree(UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True)
shutil.copy2(ZIP_PATH, UPLOAD_DIR / ZIP_PATH.name)

(UPLOAD_DIR / "dataset-metadata.json").write_text(
    json.dumps(
        {
            "title": "PCB YOLO Prepared (YOLO_PCB zip)",
            "id": DATASET_SLUG,
            "licenses": [{"name": "CC0-1.0"}],
        },
        indent=2,
    )
    + "\n"
)

print(f"Upload folder ready ({ZIP_PATH.stat().st_size / 1e9:.2f} GB zip)")

# Create dataset first time; add version if it already exists
create = subprocess.run(
    ["kaggle", "datasets", "create", "-p", str(UPLOAD_DIR), "-m", "YOLO_PCB zip from step1 prep"],
    capture_output=True,
    text=True,
)
if create.returncode == 0:
    print("Created dataset:", DATASET_SLUG)
    print(create.stdout)
else:
    print("Create returned (may already exist):", create.stderr.strip() or create.stdout)
    version = subprocess.run(
        [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(UPLOAD_DIR),
            "-m",
            "Updated YOLO_PCB zip (5551 train / 1016 val / 1016 test)",
            "--dir-mode",
            "zip",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    print("New dataset version published.")
    print(version.stdout)

print("\n=== On Nautilus (no Mac download) ===")
print("1) Copy ~/.kaggle/kaggle.json to the pod (same Kaggle account)")
print("2) git pull && bash tools/fetch_yolo_pcb_on_nautilus.sh")
print(f"   (downloads {DATASET_SLUG})")
