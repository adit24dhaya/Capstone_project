# Kaggle notebook: publish YOLO_PCB (2 cells)

## Settings

- **Internet:** ON (DsPCBSD+ download)
- **GPU:** optional for prep
- **Add Data:** `aditya2402/pcb-dataset` (raw Ding dataset — correct for Step 1)

`pcb-dataset` is **not** the finished 5551-image YOLO folder. Step 1 builds that.

---

## Cell 1 — Build YOLO_PCB (~15 min)

Paste and run **entire file:** `step1_yolo_pcb_prep_paste.py`

Success:

```text
Prepared train images: 5551
Prepared val images: 1016
Prepared test images: 1016
```

---

## Cell 2 — Publish to Kaggle Dataset

**Add-ons → Secrets:**

- `KAGGLE_USERNAME`
- `KAGGLE_KEY`

Paste and run **entire file:** `publish_yolo_pcb_to_dataset.py`

Success:

```text
Created dataset: aditya2402/pcb-yolo-prepared
```
or `Published new version: ...`

---

## Nautilus

```bash
bash tools/fetch_yolo_pcb_on_nautilus.sh
```
