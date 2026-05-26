#!/usr/bin/env bash
set -euo pipefail

if [ ! -f config.env ]; then
  echo "Opretter config.env fra config.env.example ..."
  cp config.env.example config.env
elif ! grep -q "RSNA_TRAIN_DIR" config.env; then
  echo "config.env er forældet. Genskaber fra config.env.example ..."
  cp config.env.example config.env
fi

export PROJECT_ROOT="$(pwd)"
source config.env

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

bash smoke_test.sh

echo "=== Step 1/6: Patch originalt SeuTao-repo ==="
python scripts/patch_settings.py

echo "=== Step 2/6: DICOM -> PNG ==="
cd "${SEUTAO_REPO_DIR}/2DNet/src"
python3 prepare_data.py -dcm_path "${RSNA_TRAIN_DIR}" -png_path "${RSNA_TRAIN_PNG_DIR}"
python3 prepare_data.py -dcm_path "${RSNA_TEST_DIR}"  -png_path "${RSNA_TEST_PNG_DIR}"
cd "${PROJECT_ROOT}"

echo "=== Step 3/6: Byg 3-slice concat-billeder ==="
python scripts/create_concat_images.py \
  --repo-root "${SEUTAO_REPO_DIR}"

echo "=== Step 4/6: Kopiér pretrained weights ind i 2DNet ==="
python scripts/install_pretrained_weights.py

echo "=== Step 5/6: Kør 2D prediction ==="
cd "${SEUTAO_REPO_DIR}/2DNet/src"
python3 predict.py -backbone DenseNet121_change_avg -img_size 256 -tbs 4 -vbs 4 -spth "${SEUTAO_REPO_DIR}/2DNet/DenseNet121_change_avg_256"
python3 predict.py -backbone DenseNet169_change_avg -img_size 256 -tbs 4 -vbs 4 -spth "${SEUTAO_REPO_DIR}/2DNet/DenseNet169_change_avg_256"
python3 predict.py -backbone se_resnext101_32x4d -img_size 256 -tbs 4 -vbs 4 -spth "${SEUTAO_REPO_DIR}/2DNet/se_resnext101_32x4d_256"
cd "${PROJECT_ROOT}"

echo "=== Step 6/6: Byg sequence-inputs og træn sequence model ==="
python scripts/build_sequence_inputs.py --repo-root "${SEUTAO_REPO_DIR}"

cd "${SEUTAO_REPO_DIR}/SequenceModel"
python main.py

echo "=== Pipeline færdig ==="
echo "Forventet output: ${SEQUENCE_FINAL_OUTPUT_ROOT}/"