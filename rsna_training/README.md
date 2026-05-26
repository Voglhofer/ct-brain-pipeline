# RSNA SeuTao reproduction wrapper

Reproduktion af SeuTaos 1. plads-løsning til RSNA 2019 Intracranial Hemorrhage Detection.
Alt data, pretrained modeller og hjælpefiler hentes automatisk – du behøver ikke have noget i forvejen.

## Pipeline-overblik

1. **Download**: RSNA DICOM-data (via Kaggle), pretrained 2D-modeller + metadata CSVs (via Google Drive)
2. **Preprocessing**: DICOM → PNG → 3-slice konkatenerede billeder (256×256)
3. **2D inference**: DenseNet121, DenseNet169, SE-ResNeXt101 (5-fold, TTA)
4. **Feature extraction**: Samler predictions + feature-vektorer til Sequence-modellen
5. **Sequence model**: Bidirektional GRU til endelig klassifikation

## Forudsætninger

- **Conda** (Miniconda / Anaconda)
- **Kaggle credentials**: `~/.kaggle/kaggle.json` eller env-variables `KAGGLE_USERNAME` + `KAGGLE_KEY`.
  Tilmeld også konkurrencen på [kaggle.com/c/rsna-intracranial-hemorrhage-detection](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection) og acceptér reglerne.
- **GPU** med CUDA (anbefalet; CPU virker men er ekstremt langsomt)
- ~100 GB ledig disk

## Hurtig start

```bash
# 1. Opret conda-miljø
conda env create -f environment.yml
conda activate rsna-seutao

# 2. Download alt (Kaggle data, Google Drive filer, udpak)
bash download_all.sh

# 3. Verificér setup
bash smoke_test.sh

# 4. Kør hele pipelinen
bash run_full_pipeline.sh
```

Ingen manuel redigering af `config.env` er nødvendig – den oprettes automatisk fra `config.env.example`.

## Filstruktur

```
config.env.example          # Standard-konfiguration (kopieres til config.env)
download_all.sh             # Henter alt data automatisk
run_full_pipeline.sh        # Kører hele pipelinen ende-til-ende
smoke_test.sh               # Hurtig verificering af setup
environment.yml             # Conda-miljødefinition
scripts/
  download_kaggle_data.py   # Henter RSNA DICOM via kagglehub
  download_gdrive.py        # Henter pretrained weights + CSVs fra Google Drive
  unpack_downloads.py       # Udpakker zip-filer
  patch_settings.py         # Patcher SeuTao-koden med korrekte stier
  create_concat_images.py   # Bygger 3-slice concat-billeder
  install_pretrained_weights.py  # Kopierer weights ind i 2DNet
  build_sequence_inputs.py  # Forbereder features til Sequence-modellen
  verify_setup.py           # Tjekker at alt er på plads
external/SeuTao_repo/       # Kopi af originalrepoet
```
