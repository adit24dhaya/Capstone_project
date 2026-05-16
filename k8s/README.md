# Nautilus Publication Jobs

These manifests are for publication-oriented PCB experiments on NRP Nautilus.
Use them when a browser/Jupyter session is too fragile for long training.

## Why Jobs

- Kubernetes Jobs keep running after your browser disconnects.
- The `/dev/shm` memory mount avoids PyTorch DataLoader bus errors.
- GPU, CPU, and memory requests are explicit and easier to justify.
- Outputs should be written to a mounted persistent volume, not temporary pod storage.

## Recommended Workflow

1. Ask CSUF/NRP support for access to a stronger GPU if needed:
   - A100 40GB first choice
   - A40 or RTX A6000 second choice
   - RTX 4090/3090 third choice
2. Create or reuse a persistent volume claim.
3. Copy datasets into the PVC, for example under `/workspace/data`.
4. Apply one Job YAML.
5. Watch logs and copy `/workspace/outputs` after completion.

## Useful Commands

```bash
kubectl config set-context --current --namespace=csuf-titans
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/pcb-publication-job.yaml
kubectl get pods
kubectl logs -f job/pcb-publication-job
kubectl describe job pcb-publication-job
kubectl delete job pcb-publication-job
```

## Notes

- Do not place Kaggle tokens or GitHub tokens in these YAML files.
- If a dataset needs Kaggle authentication, download it in Jupyter or use a Kubernetes Secret.
- Keep GPU utilization high; Nautilus administrators monitor underused GPU requests.
- The default job uses `yolo11m.pt` as a stronger YOLO baseline. Increase to `yolo11l.pt` only on larger GPUs.
