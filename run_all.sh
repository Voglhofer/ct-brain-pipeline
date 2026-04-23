#!/bin/bash
# Kør pipeline på alle .nii / .nii.gz filer i en mappe og saml resultater i CSV.
#
# Brug:
#   ./run_all.sh                          # default: ~/CT_test -> ~/ct-brain-pipeline/output_all
#   ./run_all.sh /sti/til/data            # custom input
#   ./run_all.sh /sti/til/data /sti/ud    # custom input + output

set -e

INPUT_DIR="${1:-$HOME/CT_test}"
OUT_ROOT="${2:-$HOME/ct-brain-pipeline/output_all}"
PIPELINE_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.csv"
echo "file,n_slices,hemorrhage_pct,hemorrhage_positive,ischemic_pct,ischemic_positive" > "$SUMMARY"

cd "$PIPELINE_DIR"

shopt -s nullglob
files=( "$INPUT_DIR"/*.nii "$INPUT_DIR"/*.nii.gz )
total=${#files[@]}
if [ "$total" -eq 0 ]; then
    echo "ERROR: No .nii / .nii.gz files found in $INPUT_DIR"
    exit 1
fi

echo "Found $total NIfTI file(s) in $INPUT_DIR"
echo "Output root: $OUT_ROOT"
echo

i=0
for f in "${files[@]}"; do
    i=$((i + 1))
    base=$(basename "$f")
    base="${base%.nii.gz}"
    base="${base%.nii}"

    if [[ "$base" == *_ROI* ]]; then
        echo "[$i/$total] SKIP (ROI): $base"
        continue
    fi

    echo "===================================================================="
    echo "[$i/$total] Processing: $base"
    echo "===================================================================="

    out_dir="$OUT_ROOT/$base"
    mkdir -p "$out_dir"

    python pipeline.py "$f" \
        --device cuda \
        --visualize \
        --keep-tom \
        --output-dir "$out_dir" \
        2>&1 | tee "$out_dir/log.txt"

    if [ -f "$out_dir/results.json" ]; then
        python - "$base" "$out_dir/results.json" "$SUMMARY" <<'PY'
import json, sys
base, results_path, summary_path = sys.argv[1], sys.argv[2], sys.argv[3]
with open(results_path) as fh:
    d = json.load(fh)
diag = d.get('patient_diagnosis', {}) or {}
hem = diag.get('hemorrhage', {}) or {}
isch = diag.get('ischemic', {}) or {}
hem_p = hem.get('subtypes', {}).get('any', {}).get('max_probability', 0) or 0
hem_pos = hem.get('patient_positive', False)
isch_p = isch.get('max_probability', 0) or 0
isch_pos = isch.get('patient_positive', False)
n = d.get('n_slices', 0)
with open(summary_path, 'a') as out:
    out.write(f"{base},{n},{hem_p*100:.2f},{hem_pos},{isch_p*100:.2f},{isch_pos}\n")
PY
    fi
done

echo ""
echo "===================================================================="
echo "Færdig. Samlet oversigt:"
echo "===================================================================="
if command -v column >/dev/null 2>&1; then
    column -s, -t "$SUMMARY"
else
    cat "$SUMMARY"
fi
echo ""
echo "Detaljer per patient: $OUT_ROOT/<patient>/"
echo "Visualiseringer:      $OUT_ROOT/<patient>/slice_*.png"
echo "CSV-oversigt:         $SUMMARY"
