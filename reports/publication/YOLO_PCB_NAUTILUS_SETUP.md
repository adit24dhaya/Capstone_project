# Publish and Copy `YOLO_PCB` to Nautilus

Prepared dataset layout (after Kaggle notebook prep):

```text
YOLO_PCB/
  data.yaml
  train/images/  train/labels/
  val/images/    val/labels/
  test/images/   test/labels/
```

Expected counts (v16-style run): **~4751 train**, **~1016 val**, **~1016 test** images.

---

## Option A — Publish from Kaggle (recommended)

### A1. Build `YOLO_PCB` on Kaggle

1. Open [automated-pcb-defect-detection-with-deep-learning](https://www.kaggle.com/code/aditya2402/automated-pcb-defect-detection-with-deep-learning).
2. **Settings:** GPU on, Internet on, dataset `aditya2402/pcb-dataset` attached.
3. Run cells through **dataset prep** until you see:

   ```text
   Prepared train images: 4751  (approx)
   Prepared val images:   1016
   Prepared test images:  1016
   ```

   You do **not** need to finish YOLO training for this step.

4. Confirm folder exists: `/kaggle/temp/YOLO_PCB/data.yaml`

### A2. Zip and publish as a Kaggle Dataset

Add a **new code cell** after prep (or run `kaggle_kernel/publish_yolo_pcb_to_dataset.py` logic) and execute:

```python
# See kaggle_kernel/publish_yolo_pcb_to_dataset.py in the repo (copy into one Kaggle cell).
```

Or paste the cell from that file in the repo.

Then on your **laptop** (with `~/.kaggle/kaggle.json`):

```bash
# After the notebook writes /kaggle/working/yolo_pcb_dataset.zip and dataset-metadata.json
kaggle datasets download -d aditya2402/pcb-yolo-prepared -p /tmp/yolo_check --unzip
```

First time you publish from inside the notebook, use **New Dataset** in the Kaggle UI:

1. Kaggle → **Your Work** → **Datasets** → **New Dataset**.
2. Upload `yolo_pcb_dataset.zip` from the notebook output / working folder.
3. Title: `PCB YOLO Prepared (YOLO_PCB)`  
4. Slug: `pcb-yolo-prepared` → full id **`aditya2402/pcb-yolo-prepared`**
5. Set **Public** or **Private** (private works with API token on Nautilus).

Record the slug; Nautilus scripts use `aditya2402/pcb-yolo-prepared`.

### A3. Download on Nautilus

```bash
export DATA_ROOT=~/data
bash ~/Capstone_project/tools/fetch_yolo_pcb_on_nautilus.sh
```

Verify:

```bash
wc -l <(find ~/data/YOLO_PCB/train/images -type f)
cat ~/data/YOLO_PCB/data.yaml
```

Train count should be ~4751, not ~483.

---

## Option B — Copy zip manually (no new Kaggle dataset)

### B1. Create zip on Kaggle

Same prep as A1, then in a notebook cell:

```python
import shutil
from pathlib import Path
shutil.make_archive("/kaggle/working/yolo_pcb_dataset", "zip", "/kaggle/temp/YOLO_PCB")
print("Zip size (MB):", Path("/kaggle/working/yolo_pcb_dataset.zip").stat().st_size / 1e6)
```

**Save Version** → **Save output** → download `yolo_pcb_dataset.zip` from the run’s **Output** tab.

### B2. Upload to Nautilus

From your Mac:

```bash
scp yolo_pcb_dataset.zip jovyan@<nautilus-host>:~/data/
```

On Nautilus:

```bash
mkdir -p ~/data/YOLO_PCB
cd ~/data
unzip -o yolo_pcb_dataset.zip -d YOLO_PCB_tmp
# zip root may be YOLO_PCB/ or flat — adjust:
if [ -d YOLO_PCB_tmp/YOLO_PCB ]; then
  mv YOLO_PCB_tmp/YOLO_PCB/* ~/data/YOLO_PCB/
else
  mv YOLO_PCB_tmp/* ~/data/YOLO_PCB/
fi
rm -rf YOLO_PCB_tmp

# Fix path inside data.yaml for Nautilus
python3 - <<'PY'
from pathlib import Path
root = Path.home() / "data" / "YOLO_PCB"
yaml_path = root / "data.yaml"
names = (root / "train" / "images").exists()
text = (
    f"path: {root}\n"
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
yaml_path.write_text(text)
print(yaml_path.read_text())
print("train images:", len(list((root / "train/images").glob('*'))))
PY
```

---

## Option C — Build on Nautilus (heavy)

Only if A and B are impossible: port full prep from `project.ipynb` (DsPCBSD+ download, merge, synthetic aug) on a Nautilus pod with **large disk** and **hours** of CPU/GPU time. Not documented here; prefer A or B.

---

## After `YOLO_PCB` is on Nautilus

```bash
export REPO=~/Capstone_project
export DATA_ROOT=~/data
export OUTPUT_DIR=~/outputs/nautilus
export LOG_DIR=~/logs
export KAGGLE_YAML=~/data/YOLO_PCB/data.yaml

cd "$REPO" && git pull
bash reports/publication/nautilus_kaggle_fair_pipeline.sh
```

Extend that pipeline later with YOLO11l + full `paper_unified_eval` on the same yaml.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `data.yaml` points to `/kaggle/temp/...` | Rewrite `path:` to `~/data/YOLO_PCB` (see B2 script) |
| Train images ≈ 483 | Wrong folder — you have `current_pcb_yolo`, not `YOLO_PCB` |
| `kaggle: command not found` on Nautilus | `pip install kaggle` and `~/.kaggle/kaggle.json` |
| Dataset private | Use same Kaggle account token on Nautilus as dataset owner |
| Zip too large for UI | Use `kaggle datasets create` from a machine with the zip, or split upload |
