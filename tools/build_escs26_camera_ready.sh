#!/usr/bin/env bash
set -euo pipefail

PUBLICATION_DIR="${PUBLICATION_DIR:-reports/publication}"
SOURCE="escs26_camera_ready.tex"
PDF="escs26_camera_ready.pdf"

required_files=(
  "$PUBLICATION_DIR/$SOURCE"
  "$PUBLICATION_DIR/escs26_references.bib"
  "$PUBLICATION_DIR/generated/result_macros.tex"
  "$PUBLICATION_DIR/generated/practical_test_table.tex"
  "$PUBLICATION_DIR/generated/controlled_test_table.tex"
  "$PUBLICATION_DIR/generated/controlled_per_class_test_table.tex"
  "$PUBLICATION_DIR/camera_ready_figures/class_balanced_success_examples.png"
  "$PUBLICATION_DIR/camera_ready_figures/representative_failure_cases.png"
)

for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "Required camera-ready artifact is missing: $file" >&2
    exit 1
  fi
done

if grep -Eq '\\(newcommand|providecommand)\{\\[^}]+\}\{--\}' \
  "$PUBLICATION_DIR/generated/result_macros.tex"; then
  echo "Generated result macros still contain placeholders." >&2
  exit 1
fi

if grep -Eqi 'pending CSV generation|results pending|generated after final evaluation' \
  "$PUBLICATION_DIR"/generated/*.tex; then
  echo "Generated manuscript tables still contain placeholders." >&2
  exit 1
fi

python3 - "$PUBLICATION_DIR/$SOURCE" <<'PY'
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text().lower()
for pattern in (
    r"fully deployed (?:and benchmarked )?on jetson",
    r"tensorrt[- ]benchmarked",
    r"jetson[- ]benchmarked",
    r"measured embedded-device (?:latency|throughput|performance)",
):
    if re.search(pattern, text):
        raise SystemExit(f"Prohibited unverified deployment claim: {pattern}")
PY

(
  cd "$PUBLICATION_DIR"
  pdflatex -interaction=nonstopmode -halt-on-error "$SOURCE"
  bibtex "${SOURCE%.tex}"
  pdflatex -interaction=nonstopmode -halt-on-error "$SOURCE"
  pdflatex -interaction=nonstopmode -halt-on-error "$SOURCE"
)

pages="$(pdfinfo "$PUBLICATION_DIR/$PDF" | awk '/^Pages:/ {print $2}')"
if [[ -z "$pages" || "$pages" -gt 15 ]]; then
  echo "Camera-ready PDF has ${pages:-unknown} pages; limit is 15." >&2
  exit 1
fi

echo "Built $PUBLICATION_DIR/$PDF ($pages pages)."
