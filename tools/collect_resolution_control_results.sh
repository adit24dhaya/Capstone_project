#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="${NAMESPACE:-csuf-titans}"
LOCAL_ROOT="${LOCAL_ROOT:-reports/publication/resolution_control_results}"
GENERATED_DIR="${GENERATED_DIR:-reports/publication/generated}"
FIGURE_DIR="${FIGURE_DIR:-reports/publication/camera_ready_figures}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-Dataset/archive/primary_yolo_pcb/resolution_control}"

YOLO_JOB="pcb-yolo640-audit"
RT640_JOB="pcb-rtdetr640-audit"
READER="pcb-resolution-control-audit-reader"

require_complete_job() {
  local job="$1"
  local complete
  complete="$(kubectl get job -n "$NAMESPACE" "$job" -o jsonpath='{.status.succeeded}')"
  if [[ "$complete" != "1" ]]; then
    echo "Job has not completed successfully: $job" >&2
    exit 1
  fi
}

copy_file() {
  local pod="$1"
  local remote="$2"
  local local_path="$3"
  mkdir -p "$(dirname "$local_path")"
  kubectl cp \
    --request-timeout=0 \
    "$NAMESPACE/$pod:$remote" \
    "$local_path"
}

copy_optional_file() {
  local pod="$1"
  local remote="$2"
  local local_path="$3"
  if kubectl exec -n "$NAMESPACE" "$pod" -- test -f "$remote"; then
    copy_file "$pod" "$remote" "$local_path"
  fi
}

for job in "$YOLO_JOB" "$RT640_JOB"; do
  require_complete_job "$job"
done

kubectl delete pod -n "$NAMESPACE" \
  -l "job-name in ($YOLO_JOB,$RT640_JOB)" \
  --ignore-not-found \
  --wait=true
kubectl delete pod -n "$NAMESPACE" \
  "$READER" \
  --ignore-not-found \
  --wait=true

kubectl apply -f k8s/pcb-resolution-control-audit-reader.yaml
kubectl wait -n "$NAMESPACE" \
  --for=condition=Ready "pod/$READER" \
  --timeout=5m

pod="$READER"
mkdir -p "$LOCAL_ROOT"/{baseline,diagnostic,yolo,rt640,rthigh,combined}
mkdir -p "$CHECKPOINT_DIR"

copy_file \
  "$pod" \
  /workspace/resolution-control/legacy-eval/baseline_gate.json \
  "$LOCAL_ROOT/baseline/baseline_gate.json"
copy_file \
  "$pod" \
  /workspace/resolution-control/legacy-eval/runs/paper_unified_eval/paper_unified_eval_metrics.csv \
  "$LOCAL_ROOT/baseline/paper_unified_eval_metrics.csv"
copy_file \
  "$pod" \
  /workspace/resolution-control/diagnostic-eval/runs/paper_unified_eval/paper_unified_eval_metrics.csv \
  "$LOCAL_ROOT/diagnostic/paper_unified_eval_metrics.csv"
copy_file \
  "$pod" \
  /workspace/resolution-control/data/YOLO_PCB_legacy/dataset_manifest.json \
  "$LOCAL_ROOT/baseline/dataset_manifest.json"
copy_file \
  "$pod" \
  /workspace/resolution-control/data/YOLO_PCB_corrected/dataset_manifest.json \
  "$LOCAL_ROOT/yolo/dataset_manifest.json"
copy_file \
  "$pod" \
  /workspace/resolution-control/yolo-controlled-eval/runs/paper_unified_eval/paper_unified_eval_metrics.csv \
  "$LOCAL_ROOT/yolo/paper_unified_eval_metrics.csv"
copy_file \
  "$pod" \
  /workspace/resolution-control/yolo-controlled-eval/runs/paper_unified_eval/paper_unified_eval_per_class.csv \
  "$LOCAL_ROOT/yolo/paper_unified_eval_per_class.csv"
copy_file \
  "$pod" \
  /workspace/resolution-control/yolo-controlled-eval/runs/paper_unified_eval/paper_unified_eval_summary.json \
  "$LOCAL_ROOT/yolo/paper_unified_eval_summary.json"
copy_file \
  "$pod" \
  /workspace/resolution-control/yolo-paper-figures/prediction_example_manifest.csv \
  "$LOCAL_ROOT/yolo/prediction_example_manifest.csv"
copy_file \
  "$pod" \
  /workspace/resolution-control/yolo-paper-figures/class_balanced_success_examples.png \
  "$LOCAL_ROOT/yolo/class_balanced_success_examples.png"
copy_file \
  "$pod" \
  /workspace/resolution-control/yolo-paper-figures/representative_failure_cases.png \
  "$LOCAL_ROOT/yolo/representative_failure_cases.png"
copy_optional_file \
  "$pod" \
  /workspace/resolution-control/yolo_experiment_manifest.json \
  "$LOCAL_ROOT/yolo/experiment_manifest.json"
copy_file \
  "$pod" \
  /workspace/resolution-control/outputs/runs/detector_train/yolo11s_640_matched/args.yaml \
  "$LOCAL_ROOT/yolo/training_args.yaml"
copy_file \
  "$pod" \
  /workspace/resolution-control/outputs/runs/detector_train/yolo11s_640_matched/results.csv \
  "$LOCAL_ROOT/yolo/training_results.csv"
copy_file \
  "$pod" \
  /workspace/resolution-control/outputs/runs/detector_train/yolo11s_640_matched/detector_train_summary.json \
  "$LOCAL_ROOT/yolo/training_summary.json"
copy_file \
  "$pod" \
  /workspace/resolution-control/outputs/runs/detector_train/yolo11s_640_matched/weights/best.pt \
  "$CHECKPOINT_DIR/yolo11s_640_matched_best.pt"

copy_file \
  "$pod" \
  /workspace/resolution-control/eval/runs/paper_unified_eval/paper_unified_eval_metrics.csv \
  "$LOCAL_ROOT/rt640/paper_unified_eval_metrics.csv"
copy_file \
  "$pod" \
  /workspace/resolution-control/eval/runs/paper_unified_eval/paper_unified_eval_per_class.csv \
  "$LOCAL_ROOT/rt640/paper_unified_eval_per_class.csv"
copy_file \
  "$pod" \
  /workspace/resolution-control/eval/runs/paper_unified_eval/paper_unified_eval_summary.json \
  "$LOCAL_ROOT/rt640/paper_unified_eval_summary.json"
copy_file \
  "$pod" \
  /workspace/resolution-control/worker_manifest.json \
  "$LOCAL_ROOT/rt640/worker_manifest.json"
copy_file \
  "$pod" \
  /workspace/resolution-control/outputs/runs/rtdetr/rtdetr_l_640_matched/args.yaml \
  "$LOCAL_ROOT/rt640/training_args.yaml"
copy_file \
  "$pod" \
  /workspace/resolution-control/outputs/runs/rtdetr/rtdetr_l_640_matched/results.csv \
  "$LOCAL_ROOT/rt640/training_results.csv"
copy_file \
  "$pod" \
  /workspace/resolution-control/outputs/runs/rtdetr/rtdetr_l_640_matched/rtdetr_train_summary.json \
  "$LOCAL_ROOT/rt640/training_summary.json"
copy_file \
  "$pod" \
  /workspace/resolution-control/outputs/runs/rtdetr/rtdetr_l_640_matched/weights/best.pt \
  "$CHECKPOINT_DIR/rtdetr_l_640_matched_best.pt"

for required in \
  "$LOCAL_ROOT/rthigh/paper_unified_eval_metrics.csv" \
  "$LOCAL_ROOT/rthigh/paper_unified_eval_per_class.csv" \
  "$LOCAL_ROOT/rthigh/worker_manifest.json"
do
  if [[ ! -f "$required" ]]; then
    echo "Retained RT-DETR-L-1280 artifact is missing: $required" >&2
    exit 1
  fi
done

python3 tools/merge_resolution_control_results.py \
  --aggregate "$LOCAL_ROOT/yolo/paper_unified_eval_metrics.csv" \
  --aggregate "$LOCAL_ROOT/rt640/paper_unified_eval_metrics.csv" \
  --aggregate "$LOCAL_ROOT/rthigh/paper_unified_eval_metrics.csv" \
  --per-class "$LOCAL_ROOT/yolo/paper_unified_eval_per_class.csv" \
  --per-class "$LOCAL_ROOT/rt640/paper_unified_eval_per_class.csv" \
  --per-class "$LOCAL_ROOT/rthigh/paper_unified_eval_per_class.csv" \
  --output-dir "$LOCAL_ROOT/combined"

python3 tools/combine_resolution_control_manifests.py \
  --yolo-manifest "$LOCAL_ROOT/yolo/experiment_manifest.json" \
  --rt640-manifest "$LOCAL_ROOT/rt640/worker_manifest.json" \
  --rthigh-manifest "$LOCAL_ROOT/rthigh/worker_manifest.json" \
  --aggregate-csv "$LOCAL_ROOT/combined/paper_unified_eval_metrics.csv" \
  --per-class-csv "$LOCAL_ROOT/combined/paper_unified_eval_per_class.csv" \
  --output "$LOCAL_ROOT/combined/experiment_manifest.json"

python3 tools/render_resolution_control_tables.py \
  --legacy-metrics "$LOCAL_ROOT/baseline/paper_unified_eval_metrics.csv" \
  --diagnostic-metrics "$LOCAL_ROOT/diagnostic/paper_unified_eval_metrics.csv" \
  --controlled-metrics "$LOCAL_ROOT/combined/paper_unified_eval_metrics.csv" \
  --controlled-per-class "$LOCAL_ROOT/combined/paper_unified_eval_per_class.csv" \
  --output-dir "$GENERATED_DIR"

mkdir -p "$FIGURE_DIR"
cp \
  "$LOCAL_ROOT/yolo/class_balanced_success_examples.png" \
  "$FIGURE_DIR/class_balanced_success_examples.png"
cp \
  "$LOCAL_ROOT/yolo/representative_failure_cases.png" \
  "$FIGURE_DIR/representative_failure_cases.png"

kubectl delete pod -n "$NAMESPACE" \
  "$READER" \
  --wait=true

echo "Collected verified resolution-control artifacts in $LOCAL_ROOT"
