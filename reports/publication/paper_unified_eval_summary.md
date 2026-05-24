# Paper Unified Evaluation

All rows were produced with the same converted Current PCB YOLO dataset, the same train/val/test split, and Ultralytics `model.val` evaluator.

| Rank | Model | Precision | Recall | mAP50 | mAP50-95 | Inference ms |
|---:|---|---:|---:|---:|---:|---:|
| 1 | YOLO11l_1280_champion | 0.9832 | 0.9881 | 0.9898 | 0.5769 | 30.7 |
| 2 | YOLO11l_1280_v100_repeat | 0.9827 | 0.9907 | 0.9866 | 0.5629 | 30.4 |
| 3 | YOLO11l_1280_refine_lowaug_v1 | 0.9817 | 0.9903 | 0.9866 | 0.5604 | 30.5 |
| 4 | YOLO11m_960_ablation | 0.9819 | 0.9648 | 0.9787 | 0.5347 | 14.6 |
