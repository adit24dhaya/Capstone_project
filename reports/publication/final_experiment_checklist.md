# Final Experiment Checklist

## Completed Evidence

- YOLO11s baseline from Kaggle v16.
- RT-DETR-L transformer-style baseline from Kaggle v16.
- Hybrid YOLO11s + RT-DETR-L fusion experiments from Kaggle v16.
- Adaptive defect-aware fusion, defect-size, calibration, robustness, and inspection-cost artifacts from Kaggle v16.
- YOLO11m 960 Nautilus/RTX 2080 Ti result with saved `best.pt`, logs, curves, and metrics.
- YOLO11l 1280 Nautilus/RTX 2080 Ti result in `local_artifacts/yolo11l_1280_publication_outputs_20260522_092005/`.
- YOLO11l 1280 ONNX export in `local_artifacts/yolo11l_1280_onnx_export_20260522_102315/`.
- YOLO11l 1280 champion eval sweep on Nautilus V100, saved in `reports/publication/champion_eval_sweep.csv`; 1280/no-TTA remains best.
- Final unified same-split/same-evaluator package in `local_artifacts/paper_unified_eval_20260524_114254/`, copied into `reports/publication/paper_unified_eval_*`.
- Six selected prediction examples in `local_artifacts/outputs/nautilus/runs/paper_figures/yolo11l_1280_selected_examples/`.

## Still Worth Running

1. Faster R-CNN baseline on the current split.
2. Optional cross-dataset test if DeepPCB/DsPCBSD+/Mendeley YOLO data is ready.
3. No more Nautilus experiments are required for ESCS. Only rerun if a file is missing or a paper reviewer/advisor asks for a specific check.

## Repeatable Export Command

```bash
python tools/run_nautilus_experiments.py \
  --experiment detector_export \
  --data-root ~/data \
  --output-dir ~/outputs/nautilus \
  --yolo-weights ~/outputs/nautilus/runs/detector_train/yolo11l_1280_publication/weights/best.pt \
  --imgsz 1280 \
  --export-format onnx
```

## Repeatable Visual Examples Command

```bash
python tools/run_nautilus_experiments.py \
  --experiment visual_examples \
  --data-root ~/data \
  --output-dir ~/outputs/nautilus \
  --yolo-weights ~/outputs/nautilus/runs/detector_train/yolo11l_1280_publication/weights/best.pt \
  --run-name yolo11l_1280_selected_examples \
  --imgsz 1280 \
  --split test \
  --prediction-save-limit 6
```

## Faster R-CNN Command

```bash
cd ~/Capstone_project
mkdir -p ~/logs ~/backups

nohup bash -lc '
python tools/run_nautilus_experiments.py \
  --experiment faster_rcnn \
  --data-root ~/data \
  --output-dir ~/outputs/nautilus \
  --frcnn-epochs 5 \
  --frcnn-batch 2 \
  --workers 0

EXIT_CODE=$?

cd ~
zip -r backups/faster_rcnn_outputs_$(date +%Y%m%d_%H%M%S).zip \
  outputs/nautilus/runs/faster_rcnn \
  logs/faster_rcnn.log

echo "EXIT_CODE=$EXIT_CODE"
echo "Backup finished at $(date)"
exit $EXIT_CODE
' > ~/logs/faster_rcnn.log 2>&1 &
```

## Unified Paper Evaluation Command

Run this after restoring the saved checkpoint folders under `~/outputs/nautilus/runs/detector_train/`. It re-evaluates every available YOLO checkpoint with the same dataset YAML, split, evaluator, and workers setting, then writes CSV/JSON summaries plus paper figures.

```bash
cd ~/Capstone_project
git pull
mkdir -p ~/logs ~/backups

python tools/run_nautilus_experiments.py \
  --experiment paper_unified_eval \
  --data-root ~/data \
  --output-dir ~/outputs/nautilus \
  --workers 0 \
  --device 0 \
  --prediction-save-limit 6 \
  2>&1 | tee ~/logs/paper_unified_eval.log

cd ~
zip -r backups/paper_unified_eval_$(date +%Y%m%d_%H%M%S).zip \
  outputs/nautilus/runs/paper_unified_eval \
  logs/paper_unified_eval.log
```

Expected outputs:

- `~/outputs/nautilus/runs/paper_unified_eval/paper_unified_eval_metrics.csv`
- `~/outputs/nautilus/runs/paper_unified_eval/paper_unified_eval_per_class.csv`
- `~/outputs/nautilus/runs/paper_unified_eval/paper_unified_eval_summary.json`
- `~/outputs/nautilus/runs/paper_unified_eval/paper_unified_eval_summary.md`
- `~/outputs/nautilus/runs/paper_unified_eval/figures/*.png`
- `~/outputs/nautilus/runs/paper_unified_eval/qualitative_examples_*/`

## Paper Priority

Start writing now using YOLO11l @ 1280 as the main detector. Use YOLO11m-960 and Kaggle fusion as ablations. Treat Faster R-CNN and cross-dataset as optional before ESCS (May 27).

## Required Acknowledgments

- Thank Dr. Paul Salvador Inventado for guidance and feedback throughout the project.
- Thank Professor Ryu for support with Nautilus/NRP access and research computing resources.
- Include the required NRP/Nautilus acknowledgment exactly in the paper acknowledgments section:

```text
This work used resources available through the National Research Platform (NRP) at the University of California, San Diego. NRP has been developed, and is supported in part, by funding from National Science Foundation, from awards 1730158, 1540112, 1541349, 1826967, 2112167, 2100237, and 2120019, as well as additional funding from community partners. The CSUF Titan Supercomputing Center is one of the collaborative partners to contribute to NRP resources.
```

- Add the NRP citation: The National Research Platform: Stretched, Multi-Tenant, Scientific Kubernetes Cluster.
