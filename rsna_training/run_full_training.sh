#!/usr/bin/env bash
# Re-train all CNN backbones AND the SequenceModel, with full metric
# instrumentation enabled. Use this on the hospital workstation when the
# goal is to capture training metrics (advisor request) — not the regular
# inference pipeline (which is in run_full_pipeline.sh).
#
# IMPORTANT: this script does NOT change the training procedure itself.
# It runs the instrumented train.py + main.py that were modified to add
# logging only. Hyperparameters, data, augmentation, seeds — all unchanged.
#
# Estimated wall-clock on RTX 5090:
#   DenseNet121 (5 folds × 80 epochs): ~2 days
#   DenseNet169 (5 folds × 80 epochs): ~2-3 days
#   SE-ResNeXt101 (5 folds × 80 epochs): ~3-4 days
#   Feature extraction (predict.py × 3): ~1-2 days
#   SequenceModel training (5 folds × 40 epochs): ~5-8 hours
#   TOTAL: ~10-15 days sequential
#
# Usage:
#   bash run_full_training.sh           # full run
#   bash run_full_training.sh --smoke   # 1 fold × 2 epochs smoke test

set -euo pipefail

SMOKE_FLAG=""
SMOKE_SUFFIX=""
if [ "${1:-}" = "--smoke" ]; then
  SMOKE_FLAG="--smoke"
  SMOKE_SUFFIX="_smoke"
  echo ">>> SMOKE TEST MODE: 1 fold × 2 epochs, ~10-15 min total"
fi

if [ ! -f config.env ]; then
  echo "Opretter config.env fra config.env.example ..."
  cp config.env.example config.env
fi

export PROJECT_ROOT="$(pwd)"
source config.env

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== Step 1: Patch settings ==="
python scripts/patch_settings.py

echo "=== Step 2: Train DenseNet121 (5 folds × 80 epochs) ==="
cd "${SEUTAO_REPO_DIR}/2DNet/src"
python3 train.py \
  -backbone DenseNet121_change_avg \
  -img_size 256 -tbs 32 -vbs 32 \
  -save_path "DenseNet121_change_avg_256${SMOKE_SUFFIX}" \
  ${SMOKE_FLAG} 2>&1 | tee "${PROJECT_ROOT}/train_dn121${SMOKE_SUFFIX}.log"

echo "=== Step 3: Train DenseNet169 ==="
python3 train.py \
  -backbone DenseNet169_change_avg \
  -img_size 256 -tbs 32 -vbs 32 \
  -save_path "DenseNet169_change_avg_256${SMOKE_SUFFIX}" \
  ${SMOKE_FLAG} 2>&1 | tee "${PROJECT_ROOT}/train_dn169${SMOKE_SUFFIX}.log"

echo "=== Step 4: Train SE-ResNeXt101 ==="
python3 train.py \
  -backbone se_resnext101_32x4d \
  -img_size 256 -tbs 32 -vbs 32 \
  -save_path "se_resnext101_32x4d_256${SMOKE_SUFFIX}" \
  ${SMOKE_FLAG} 2>&1 | tee "${PROJECT_ROOT}/train_seresnext${SMOKE_SUFFIX}.log"

cd "${PROJECT_ROOT}"

if [ -n "${SMOKE_FLAG}" ]; then
  echo "=== Smoke test complete. Inspect data_test/*_smoke/fold0/ before running full pipeline. ==="
  exit 0
fi

echo "=== Step 5: Generate predictions + features (3 backbones) ==="
cd "${SEUTAO_REPO_DIR}/2DNet/src"
python3 predict.py -backbone DenseNet121_change_avg -img_size 256 -tbs 4 -vbs 4 -spth DenseNet121_change_avg_256
python3 predict.py -backbone DenseNet169_change_avg -img_size 256 -tbs 4 -vbs 4 -spth DenseNet169_change_avg_256
python3 predict.py -backbone se_resnext101_32x4d -img_size 256 -tbs 4 -vbs 4 -spth se_resnext101_32x4d_256
cd "${PROJECT_ROOT}"

echo "=== Step 6: Build sequence-model inputs ==="
python scripts/build_sequence_inputs.py --repo-root "${SEUTAO_REPO_DIR}"

echo "=== Step 7: Train SequenceModel (5 folds × 40 epochs) ==="
cd "${SEUTAO_REPO_DIR}/SequenceModel"
python3 main.py --skip-valid 2>&1 | tee "${PROJECT_ROOT}/train_sm.log"
cd "${PROJECT_ROOT}"

echo "=== Step 8: Generate analysis plots + LaTeX tables ==="
cd "${SEUTAO_REPO_DIR}"
python3 analyze_training.py \
  --root 2DNet/src/data_test \
  --out "${PROJECT_ROOT}/training_analysis"
cd "${PROJECT_ROOT}"

echo "=== Full training pipeline complete ==="
echo "Results:"
echo "  CNN logs:       ${SEUTAO_REPO_DIR}/2DNet/src/data_test/<backbone>/fold<N>/"
echo "  SM logs:        ${SEQUENCE_FINAL_OUTPUT_ROOT}/version3_debug/fold<N>/"
echo "  Plots+tables:   ${PROJECT_ROOT}/training_analysis/"
