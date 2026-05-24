# Paste into ONE Kaggle cell. Works if ANY of these exist:
#   A) /kaggle/temp/YOLO_PCB  (same session as step1 prep)
#   B) /kaggle/working/yolo_pcb_dataset.zip  (already zipped)
#   C) /kaggle/input/.../yolo_pcb_dataset.zip  (attached dataset / uploaded zip)
#   D) /kaggle/input/.../YOLO_PCB  (attached extracted folder)
#
# Then publishes to aditya2402/pcb-yolo-prepared (needs Secrets: KAGGLE_USERNAME, KAGGLE_KEY)

from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

DATASET_SLUG = "aditya2402/pcb-yolo-prepared"
WORK = Path("/kaggle/working")
ZIP_PATH = WORK / "yolo_pcb_dataset.zip"
META_DIR = WORK / "pcb_yolo_prepared_upload"
YOLO_ROOT = Path("/kaggle/temp/YOLO_PCB")
INPUT_ROOT = Path("/kaggle/input")

RAW_PCB_CANDIDATES = [
    Path("/kaggle/input/datasets/aditya2402/pcb-dataset/PCB-DATASET-master"),
    Path("/kaggle/input/pcb-dataset/PCB-DATASET-master"),
]


def find_raw_pcb_dataset() -> Path | None:
    for candidate in RAW_PCB_CANDIDATES:
        if (candidate / "images").exists() and (candidate / "Annotations").exists():
            return candidate
    if INPUT_ROOT.exists():
        for candidate in INPUT_ROOT.rglob("PCB-DATASET-master"):
            if (candidate / "images").exists() and (candidate / "Annotations").exists():
                return candidate
    return None


def find_existing_zip() -> Path | None:
    if ZIP_PATH.exists():
        return ZIP_PATH
    if INPUT_ROOT.exists():
        for p in INPUT_ROOT.rglob("yolo_pcb_dataset.zip"):
            return p
    return None


def find_yolo_root() -> Path | None:
    if (YOLO_ROOT / "data.yaml").exists():
        return YOLO_ROOT
    if INPUT_ROOT.exists():
        for p in INPUT_ROOT.rglob("YOLO_PCB"):
            if (p / "data.yaml").exists() or (p / "train" / "images").exists():
                return p
    staging = WORK / "YOLO_PCB"
    if (staging / "train" / "images").exists():
        return staging
    return None


def build_zip_from_root(root: Path) -> Path:
    names_block = "\n".join(
        f"  {i}: {name}"
        for i, name in enumerate(
            ["Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"]
        )
    )
    staging = WORK / "YOLO_PCB"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(root, staging)
    (staging / "data.yaml").write_text(
        "path: .\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        f"names:\n{names_block}\n"
    )
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", staging.parent, staging.name)
    shutil.rmtree(staging)
    return ZIP_PATH


existing_zip = find_existing_zip()
yolo_root = find_yolo_root()

if existing_zip and existing_zip != ZIP_PATH:
    print(f"Using attached zip: {existing_zip}")
    shutil.copy2(existing_zip, ZIP_PATH)
elif yolo_root is not None:
    print(f"Building zip from: {yolo_root}")
    for split in ("train", "val", "test"):
        n = len(list((yolo_root / split / "images").glob("*")))
        print(f"  {split}: {n} images")
    build_zip_from_root(yolo_root)
elif ZIP_PATH.exists():
    print(f"Using existing: {ZIP_PATH}")
else:
    raw = find_raw_pcb_dataset()
    msg = "No prepared YOLO_PCB or yolo_pcb_dataset.zip found.\n\n"
    if raw is not None:
        msg += (
            f"You attached RAW pcb-dataset at:\n  {raw}\n\n"
            "That is NOT the prepared split (5551 train). It is only the Ding XML dataset.\n\n"
            "In THIS notebook, run ONE cell with the full file:\n"
            "  step1_yolo_pcb_prep_paste.py\n"
            "(Internet ON, ~15 min) — then re-run this publish cell.\n"
        )
    else:
        msg += (
            "Fix (pick one):\n"
            "  1) Add Data → aditya2402/pcb-dataset + run step1_yolo_pcb_prep_paste.py, OR\n"
            "  2) Add Data → upload yolo_pcb_dataset.zip from your earlier notebook, OR\n"
            "  3) Run publish in the notebook where prep already finished (do not restart kernel).\n"
        )
    raise FileNotFoundError(msg)

size_mb = ZIP_PATH.stat().st_size / (1024 * 1024)
print(f"Ready to publish: {ZIP_PATH} ({size_mb:.1f} MiB)")

# --- publish to Kaggle Dataset (Kaggle API credentials) ---
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])


def load_kaggle_credentials() -> None:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return
    try:
        from kaggle_secrets import UserSecretsClient

        secrets = UserSecretsClient()
        os.environ["KAGGLE_USERNAME"] = secrets.get_secret("KAGGLE_USERNAME")
        os.environ["KAGGLE_KEY"] = secrets.get_secret("KAGGLE_KEY")
        return
    except Exception as exc:
        print("Notebook Secrets not set:", exc)

    kaggle_json = Path("/kaggle/input/kaggle-json/kaggle.json")
    if kaggle_json.exists():
        cfg = json.loads(kaggle_json.read_text())
        os.environ["KAGGLE_USERNAME"] = cfg["username"]
        os.environ["KAGGLE_KEY"] = cfg["key"]
        print("Loaded credentials from /kaggle/input/kaggle-json/kaggle.json")
        return

    import getpass

    print(
        "Add Secrets (recommended): Add-ons → Secrets →\n"
        "  KAGGLE_USERNAME = aditya2402\n"
        "  KAGGLE_KEY = <from kaggle.com/settings>\n"
        "Or enter once below (input hidden for key):\n"
    )
    os.environ["KAGGLE_USERNAME"] = input("KAGGLE_USERNAME [aditya2402]: ").strip() or "aditya2402"
    os.environ["KAGGLE_KEY"] = getpass.getpass("KAGGLE_KEY: ").strip()
    if not os.environ["KAGGLE_KEY"]:
        raise RuntimeError("KAGGLE_KEY is required to publish the dataset.")


load_kaggle_credentials()

if META_DIR.exists():
    shutil.rmtree(META_DIR)
META_DIR.mkdir(parents=True)
shutil.copy2(ZIP_PATH, META_DIR / ZIP_PATH.name)
(META_DIR / "dataset-metadata.json").write_text(
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

create = subprocess.run(
    ["kaggle", "datasets", "create", "-p", str(META_DIR), "-m", "YOLO_PCB zip"],
    capture_output=True,
    text=True,
)
if create.returncode == 0:
    print("Created dataset:", DATASET_SLUG)
else:
    subprocess.run(
        [
            "kaggle",
            "datasets",
            "version",
            "-p",
            str(META_DIR),
            "-m",
            "Updated YOLO_PCB zip",
            "--dir-mode",
            "zip",
        ],
        check=True,
    )
    print("Published new version:", DATASET_SLUG)

print("\nOn Nautilus: git pull && bash tools/fetch_yolo_pcb_on_nautilus.sh")
