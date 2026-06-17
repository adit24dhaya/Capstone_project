# ESCS'26 Resolution-Control Runbook

This runbook reproduces the camera-ready YOLO11s/RT-DETR-L experiments
on Nautilus without committing credentials or model weights.

## Protocol

- Namespace: `csuf-titans`
- GPU: `Tesla-V100-SXM2-32GB`
- Seed: `42`
- Evaluation: batch 1, workers 0, no test-time augmentation
- Matched resolution: 640 pixels
- High-resolution RT-DETR-L: 1280 pixels
- 960 pixels is allowed only after a logged CUDA out-of-memory failure

## Credentials

Create the Kaggle secret from the local token. Never add the token to
Git or a ConfigMap.

```bash
kubectl create secret generic kaggle-access-token \
  -n csuf-titans \
  --from-file=access_token="$HOME/.kaggle/access_token" \
  --dry-run=client -o yaml | kubectl apply -f -
```

## Main Workspace and YOLO

Set `YOLO_CHECKPOINT` and `RTDETR_CHECKPOINT` to the two archived local
checkpoint paths before staging.

```bash
export YOLO_CHECKPOINT=/absolute/path/to/yolo11s_1280_kaggle_fair_best.pt
export RTDETR_CHECKPOINT=/absolute/path/to/rtdetr_l_kaggle_fair_best.pt

kubectl apply -f k8s/pcb-resolution-control-pvc.yaml
kubectl apply -f k8s/pcb-resolution-control-upload-pod.yaml
kubectl wait -n csuf-titans \
  --for=condition=Ready pod/pcb-resolution-control-upload \
  --timeout=5m

kubectl exec -n csuf-titans pcb-resolution-control-upload -- \
  mkdir -p /workspace/repo /workspace/checkpoints

tar -cf - \
  reports/publication/nautilus_resolution_control_pipeline.sh \
  reports/publication/resolution_control_corrected_models.json \
  reports/publication/resolution_control_diagnostic_models.json \
  reports/publication/resolution_control_legacy_models.json \
  reports/publication/archived_baseline_metrics.csv \
  tools/prepare_resolution_control_dataset.py \
  tools/run_nautilus_experiments.py \
  tools/verify_baseline_gate.py \
  tools/generate_resolution_control_examples.py \
  tools/render_resolution_control_tables.py \
  tools/build_resolution_control_manifest.py \
  | kubectl exec -i -n csuf-titans pcb-resolution-control-upload -- \
      tar -xf - -C /workspace/repo

kubectl cp "$YOLO_CHECKPOINT" \
  csuf-titans/pcb-resolution-control-upload:/workspace/checkpoints/yolo11s_1280_kaggle_fair_best.pt
kubectl cp "$RTDETR_CHECKPOINT" \
  csuf-titans/pcb-resolution-control-upload:/workspace/checkpoints/rtdetr_l_kaggle_fair_best.pt

kubectl exec -n csuf-titans pcb-resolution-control-upload -- \
  sha256sum /workspace/checkpoints/*.pt

kubectl delete pod -n csuf-titans pcb-resolution-control-upload
kubectl apply -f k8s/pcb-resolution-control-job.yaml
```

The archived checkpoints must exist on `pcb-publication-workspace` at:

```text
/workspace/checkpoints/yolo11s_1280_kaggle_fair_best.pt
/workspace/checkpoints/rtdetr_l_kaggle_fair_best.pt
```

Verify their hashes against the experiment record before running.

## Parallel RT-DETR Workers

```bash
kubectl create configmap pcb-resolution-control-worker \
  -n csuf-titans \
  --from-file=worker.sh=reports/publication/nautilus_rtdetr_worker.sh \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create configmap pcb-resolution-control-tools \
  -n csuf-titans \
  --from-file=tools/prepare_resolution_control_dataset.py \
  --from-file=tools/run_nautilus_experiments.py \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/pcb-resolution-control-parallel-rtdetr.yaml
```

## YOLO Finalization

After the YOLO11s-640 run has 50 epochs and
`detector_train_summary.json` exists, stop the sequential job before it
starts duplicate RT-DETR work:

```bash
kubectl delete job -n csuf-titans pcb-resolution-control

kubectl create configmap pcb-resolution-control-finalizer \
  -n csuf-titans \
  --from-file=reports/publication/nautilus_yolo_finalizer.sh \
  --from-file=tools/run_nautilus_experiments.py \
  --from-file=tools/generate_resolution_control_examples.py \
  --from-file=tools/prepare_resolution_control_dataset.py \
  --from-file=tools/build_resolution_control_manifest.py \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl apply -f k8s/pcb-resolution-control-yolo-finalizer.yaml
```

## Monitoring

```bash
kubectl get jobs,pods -n csuf-titans | rg 'pcb-(resolution-control|rtdetr|yolo)'
kubectl logs -n csuf-titans job/pcb-rtdetr-640 --tail=40
kubectl logs -n csuf-titans job/pcb-rtdetr-highres --tail=40
```

Confirm every pod is assigned to a V100:

```bash
kubectl get pods -n csuf-titans \
  -o custom-columns='NAME:.metadata.name,NODE:.spec.nodeName,GPU:.spec.containers[0].resources.limits.nvidia\.com/gpu'
```

## Collection and Paper Build

After all three final jobs report `Complete`:

```bash
bash tools/collect_resolution_control_results.sh
bash tools/build_escs26_camera_ready.sh
```

The collector validates the protocol, merges CSV files, builds one
combined manifest, regenerates LaTeX tables/macros, and copies the
success/failure figures into the camera-ready source. The build script
rejects placeholder values, missing figures, papers over 15 pages, and
unverified embedded-device claims.

The collector removes only the completed Job pods, not the Job records.
It temporarily mounts each PVC through a read-only artifact-reader pod
because `kubectl cp` cannot copy from a container after it has exited.
