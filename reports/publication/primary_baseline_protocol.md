# Primary Baseline Protocol (`current_pcb_yolo`)

This document defines how Nautilus experiments align with Kaggle v16 settings and where they intentionally differ. Use it in the paper Methods section (Sec. 3–4).

## Benchmark

| Item | Value |
|------|--------|
| Split name | `current_pcb_yolo` |
| Source | Ding et al. PCB-DATASET via Nautilus conversion |
| Train / val / test images | 483 / 102 / 108 |
| `data.yaml` | `$OUTPUT_DIR/datasets/current_pcb_yolo/data.yaml` |
| Seed | 42 |

**Not included on this split:** DsPCBSD+ merge, copy-paste synthetic augmentation (Kaggle notebook only). State this explicitly when comparing to Kaggle v16 rows.

## Model training matrix (Nautilus primary)

| Model | Pretrained | imgsz | Epochs | Batch | Patience | Notes |
|-------|------------|------:|-------:|------:|---------:|-------|
| YOLO11l champion | yolo11l.pt | 1280 | 100 (early stop) | 1–4 | 20 | Headline model (already trained) |
| YOLO11m ablation | yolo11m.pt | 960 | 100 (early stop) | 2 | 20 | Speed ablation (already trained) |
| YOLO11s baseline | yolo11s.pt | 1280 | 50 | 12 | 25 | Matches Kaggle v16 epoch/batch/imgsz |
| RT-DETR-L | rtdetr-l.pt | 640 | 10 | 4 | 10 | Matches Kaggle v16 |
| Faster R-CNN | COCO ResNet50-FPN | native | 5 | 2 | — | Short classical baseline (completed) |

Common Ultralytics defaults unless overridden: `cos_lr=True`, `close_mosaic=15`, `seed=42`, `workers=0` on Nautilus (avoids `/dev/shm` bus errors during `val`).

## Evaluation protocol

| Setting | Value | Purpose |
|---------|-------|---------|
| Evaluator | Ultralytics `model.val` | Same as training stack |
| `batch` | **1** for paper tables | Fair per-image latency |
| `augment` | False | No TTA in main tables |
| TTA / 1536 sweep | Separate `champion_eval_sweep.csv` | Champion only |

## Adaptive fusion

| Setting | Value |
|---------|-------|
| YOLO weights | `yolo11s_1280_current_pcb/weights/best.pt` |
| RT-DETR weights | `rtdetr_l_current_pcb/weights/best.pt` |
| YOLO `imgsz` | 1280 |
| RT-DETR `imgsz` | 640 |
| Data | `current_pcb_yolo` via conversion summary (**no** `--external-data-yaml`) |
| Policy | Validation-calibrated class-wise WBF (`adaptive_defect_aware_policy.json`) |

## Kaggle v16 vs Nautilus primary (paper footnote)

| Aspect | Kaggle v16 | Nautilus primary |
|--------|------------|------------------|
| Train images | ~4,751 | 483 |
| Synthetic copy-paste aug | Yes | No |
| DsPCBSD+ merge | Yes | No |
| YOLO11s / RT-DETR hyperparameters | 1280/50ep/b12; 640/10ep/b4 | **Same** |
| Hardware | Kaggle P100/T4 | Tesla V100 32GB |

Do **not** rank Kaggle fusion rows against Nautilus YOLO11l in one leaderboard without this footnote.

## Commands

Full pipeline:

```bash
export REPO=~/Capstone_project
export DATA_ROOT=~/data
export OUTPUT_DIR=~/outputs/nautilus
export LOG_DIR=~/logs
bash reports/publication/nautilus_primary_baselines_pipeline.sh
```

Individual steps: see `nautilus_primary_baselines_pipeline.sh`.

## Expected artifacts

- `runs/paper_unified_eval_batch1_existing/` — four YOLO checkpoints @ batch=1
- `runs/detector_train/yolo11s_1280_current_pcb/`
- `runs/rtdetr/rtdetr_l_current_pcb/`
- `runs/adaptive_fusion/adaptive_defect_aware_hybrid_test_metrics.csv`
- `runs/paper_unified_eval_batch1_all/` — six models @ batch=1
