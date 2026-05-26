#!/usr/bin/env bash
set -euo pipefail

if [ ! -f config.env ]; then
  echo "Opretter config.env fra config.env.example ..."
  cp config.env.example config.env
elif ! grep -q "RSNA_TRAIN_DIR" config.env; then
  echo "config.env er forældet (mangler RSNA_TRAIN_DIR). Genskaber fra config.env.example ..."
  cp config.env.example config.env
fi

export PROJECT_ROOT="$(pwd)"
source config.env

mkdir -p "${DOWNLOAD_DIR}" "${PRETRAIN_DIR}" "${AUX_DATA_DIR}" external

# --- 1. Klon SeuTao-repo ---
if [ ! -d "${SEUTAO_REPO_DIR}" ]; then
  echo "=== Kloner originalrepo ==="
  git clone https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection.git "${SEUTAO_REPO_DIR}"
else
  echo "Originalrepo findes allerede: ${SEUTAO_REPO_DIR}"
fi

# --- 2. Download RSNA data fra Kaggle ---
echo "=== Downloader RSNA dataset fra Kaggle ==="
python scripts/download_kaggle_data.py

# --- 3. Download Google Drive-filer (data.zip, csv.zip, pretrained weights) ---
echo "=== Downloader Google Drive-filer ==="
python scripts/download_gdrive.py \
  --download-root "${AUX_DATA_DIR}" \
  --pretrained-root "${PRETRAIN_DIR}"

# --- 4. Udpak downloads ---
echo "=== Udpakker downloads ==="
python scripts/unpack_downloads.py

echo "=== Downloads færdige ==="
echo "Næste trin: bash smoke_test.sh"