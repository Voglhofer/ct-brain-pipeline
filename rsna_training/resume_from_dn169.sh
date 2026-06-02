#!/usr/bin/env bash
# Resume training from DN169 onwards. DN121 is already complete (5 folds × 80
# epochs with full instrumentation), so this skips it.
#
# Runs the rest of the pipeline autonomously:
#   1. DenseNet169 (5 folds × 80 epochs)        ~2-3 days
#   2. SE-ResNeXt101 (5 folds × 80 epochs)      ~3-4 days
#   3. predict.py × 3 backbones (DN121 + DN169 + SE-ResNeXt)  ~1-2 days
#   4. build_sequence_inputs.py                 minutes
#   5. SequenceModel (5 folds × 40 epochs)      ~5-8 hours
#   6. analyze_training.py                      minutes
#
# Each step is robust to the previous step failing — if DN169 crashes, the
# script logs the error and continues to SE-ResNeXt. If both fail, it still
# attempts predict.py on whatever checkpoints exist. This way you don't lose
# the SM run just because one CNN had a hiccup.
#
# Usage:
#   bash resume_from_dn169.sh

set -uo pipefail   # NOT -e: we want to continue past failures

if [ ! -f config.env ]; then
  echo "Opretter config.env fra config.env.example ..."
  cp config.env.example config.env
fi

export PROJECT_ROOT="$(pwd)"
source config.env

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

# Track step results so we can decide what to do at the end.
DN169_OK=0
SERES_OK=0
PREDICT_DN121_OK=0
PREDICT_DN169_OK=0
PREDICT_SERES_OK=0
BUILD_SEQ_OK=0
SM_OK=0

echo "================================================================"
echo "=== Resume training from DN169 (DN121 already complete)       ==="
echo "=== Started: $(date)                                          ==="
echo "================================================================"

echo ""
echo "=== Step 1: Train DenseNet169 (5 folds × 80 epochs) ==="
echo "Estimated wall-clock: ~2-3 days"
cd "${SEUTAO_REPO_DIR}/2DNet/src"
python3 train.py \
  -backbone DenseNet169_change_avg \
  -img_size 256 -tbs 32 -vbs 32 \
  -save_path "DenseNet169_change_avg_256" \
  2>&1 | tee "${PROJECT_ROOT}/train_dn169.log"
if [ ${PIPESTATUS[0]} -eq 0 ]; then
  DN169_OK=1
  echo ">>> DN169 finished OK at $(date)"
else
  echo ">>> DN169 FAILED at $(date) — continuing to SE-ResNeXt anyway"
fi

echo ""
echo "=== Step 2: Train SE-ResNeXt101 (5 folds × 80 epochs) ==="
echo "Estimated wall-clock: ~3-4 days"
python3 train.py \
  -backbone se_resnext101_32x4d \
  -img_size 256 -tbs 32 -vbs 32 \
  -save_path "se_resnext101_32x4d_256" \
  2>&1 | tee "${PROJECT_ROOT}/train_seresnext.log"
if [ ${PIPESTATUS[0]} -eq 0 ]; then
  SERES_OK=1
  echo ">>> SE-ResNeXt finished OK at $(date)"
else
  echo ">>> SE-ResNeXt FAILED at $(date) — continuing to predict step"
fi

echo ""
echo "=== Step 3: Generate predictions + features (all backbones with checkpoints) ==="
echo "Estimated wall-clock: ~1-2 days total"

# DN121 is already trained (from previous run).
if [ -d "${SEUTAO_REPO_DIR}/2DNet/src/data_test/DenseNet121_change_avg_256/fold0" ]; then
  echo ">>> Running predict.py for DenseNet121 ..."
  python3 predict.py -backbone DenseNet121_change_avg -img_size 256 -tbs 4 -vbs 4 \
    -spth DenseNet121_change_avg_256 \
    2>&1 | tee "${PROJECT_ROOT}/predict_dn121.log"
  [ ${PIPESTATUS[0]} -eq 0 ] && PREDICT_DN121_OK=1
fi

if [ -d "${SEUTAO_REPO_DIR}/2DNet/src/data_test/DenseNet169_change_avg_256/fold0" ]; then
  echo ">>> Running predict.py for DenseNet169 ..."
  python3 predict.py -backbone DenseNet169_change_avg -img_size 256 -tbs 4 -vbs 4 \
    -spth DenseNet169_change_avg_256 \
    2>&1 | tee "${PROJECT_ROOT}/predict_dn169.log"
  [ ${PIPESTATUS[0]} -eq 0 ] && PREDICT_DN169_OK=1
fi

if [ -d "${SEUTAO_REPO_DIR}/2DNet/src/data_test/se_resnext101_32x4d_256/fold0" ]; then
  echo ">>> Running predict.py for SE-ResNeXt101 ..."
  python3 predict.py -backbone se_resnext101_32x4d -img_size 256 -tbs 4 -vbs 4 \
    -spth se_resnext101_32x4d_256 \
    2>&1 | tee "${PROJECT_ROOT}/predict_seresnext.log"
  [ ${PIPESTATUS[0]} -eq 0 ] && PREDICT_SERES_OK=1
fi

cd "${PROJECT_ROOT}"

echo ""
echo "=== Step 4: Build sequence-model inputs ==="
# build_sequence_inputs.py needs predictions from all available backbones.
# It will fail if no predictions exist — that's expected and we handle it.
python3 scripts/build_sequence_inputs.py --repo-root "${SEUTAO_REPO_DIR}" \
  2>&1 | tee "${PROJECT_ROOT}/build_seq.log"
if [ ${PIPESTATUS[0]} -eq 0 ]; then
  BUILD_SEQ_OK=1
  echo ">>> build_sequence_inputs OK at $(date)"
else
  echo ">>> build_sequence_inputs FAILED — likely missing predictions for one or more backbones"
fi

echo ""
echo "=== Step 5: Train SequenceModel (5 folds × 40 epochs) ==="
echo "Estimated wall-clock: ~5-8 hours"
if [ "${BUILD_SEQ_OK}" -eq 1 ]; then
  cd "${SEUTAO_REPO_DIR}/SequenceModel"
  python3 main.py --skip-valid 2>&1 | tee "${PROJECT_ROOT}/train_sm.log"
  [ ${PIPESTATUS[0]} -eq 0 ] && SM_OK=1
  cd "${PROJECT_ROOT}"
else
  echo ">>> Skipping SM training because build_sequence_inputs failed"
fi

echo ""
echo "=== Step 6: Generate analysis plots + LaTeX tables ==="
cd "${SEUTAO_REPO_DIR}"
python3 analyze_training.py \
  --root 2DNet/src/data_test \
  --out "${PROJECT_ROOT}/training_analysis" \
  2>&1 | tee "${PROJECT_ROOT}/analyze.log"
cd "${PROJECT_ROOT}"

echo ""
echo "================================================================"
echo "=== Resume pipeline complete                                  ==="
echo "=== Finished: $(date)                                         ==="
echo "================================================================"
echo "Status summary:"
echo "  DN169 training:        $([ ${DN169_OK} -eq 1 ] && echo OK || echo FAILED)"
echo "  SE-ResNeXt training:   $([ ${SERES_OK} -eq 1 ] && echo OK || echo FAILED)"
echo "  predict.py DN121:      $([ ${PREDICT_DN121_OK} -eq 1 ] && echo OK || echo FAILED)"
echo "  predict.py DN169:      $([ ${PREDICT_DN169_OK} -eq 1 ] && echo OK || echo FAILED)"
echo "  predict.py SE-ResNeXt: $([ ${PREDICT_SERES_OK} -eq 1 ] && echo OK || echo FAILED)"
echo "  build_sequence_inputs: $([ ${BUILD_SEQ_OK} -eq 1 ] && echo OK || echo FAILED)"
echo "  SequenceModel:         $([ ${SM_OK} -eq 1 ] && echo OK || echo FAILED)"
echo ""
echo "Logs:"
echo "  ${PROJECT_ROOT}/train_dn169.log"
echo "  ${PROJECT_ROOT}/train_seresnext.log"
echo "  ${PROJECT_ROOT}/predict_*.log"
echo "  ${PROJECT_ROOT}/build_seq.log"
echo "  ${PROJECT_ROOT}/train_sm.log"
echo "  ${PROJECT_ROOT}/analyze.log"
