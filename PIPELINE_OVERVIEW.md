# Pipeline overblik — ende til ende reproduktionsguide

This document is the **self-contained technical reference** for the bachelor
thesis *AI-assisted Stroke Flagging in Non-contrast Brain CT Imaging*
(Voglhofer & Bjerg, DTU, 2026). A reader with no prior knowledge of the
project should be able to reproduce the full pipeline from this file alone —
data acquisition, preprocessing, training, inference, evaluation, and
deployment.

The system has **two parallel branches**: a hemorrhage detection branch and
an ischemic stroke classification branch. They run on the same input scan,
produce independent scan-level scores, and are combined at the end into a
single prioritization flag.

> **Note on scope.** The current deployed pipeline (`pipeline.py`) implements
> only the CNN ensemble for hemorrhage. SeuTao's original solution included
> a Sequence Model stage that is documented here but not yet integrated; the
> re-training instrumentation (`rsna_training/`) is in place to fix this.

---

## Table of contents

1. [System diagram](#1-system-diagram)
2. [Reproduction prerequisites](#2-reproduction-prerequisites)
3. [Data sources and acquisition](#3-data-sources-and-acquisition)
4. [Shared preprocessing](#4-shared-preprocessing)
5. [Hemorrhage detection branch](#5-hemorrhage-detection-branch)
6. [Ischemic stroke classification branch](#6-ischemic-stroke-classification-branch)
7. [Sequence Model (planned addition)](#7-sequence-model-planned-addition)
8. [Combined prioritization decision](#8-combined-prioritization-decision)
9. [External evaluation](#9-external-evaluation)
10. [Threshold derivation](#10-threshold-derivation)
11. [Known discrepancies with the thesis report](#11-known-discrepancies-with-the-thesis-report)
12. [Step-by-step reproduction guide](#12-step-by-step-reproduction-guide)
13. [Repository layout](#13-repository-layout)
14. [End-to-end command reference](#14-end-to-end-command-reference)

---

## 1. System diagram

```
                    ┌──────────────────────────────────────┐
                    │  Input: DICOM folder / NIfTI volume  │
                    └──────────────────┬───────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                │       Shared preprocessing                  │
                │  • DICOM → HU (slope/intercept)             │
                │  • NIfTI → HU (already calibrated)          │
                │  • Series filtering, slice sorting          │
                │  • Optional reslice to 5 mm                 │
                └──────────────┬──────────────┬───────────────┘
                               │              │
              ┌────────────────▼───┐    ┌─────▼──────────────┐
              │  Hemorrhage branch │    │  Ischemic branch    │
              │  (15-model CNN     │    │  (DenseNet121       │
              │   ensemble)        │    │   transfer model)   │
              └─────────┬──────────┘    └─────────┬───────────┘
                        │ 6 probabilities         │ 1 probability
                        │ per slice               │ per slice
                        ▼                         ▼
              ┌─────────────────────┐    ┌─────────────────────┐
              │ Max-pool over scan  │    │ Max-pool over scan  │
              │ Youden thresholds   │    │ Threshold 0.5       │
              └─────────┬───────────┘    └─────────┬───────────┘
                        │                          │
                        └──────────┬───────────────┘
                                   ▼
                    ┌──────────────────────────┐
                    │  Final flag: ANY branch  │
                    │  exceeds threshold       │
                    └──────────────────────────┘
```

---

## 2. Reproduction prerequisites

### 2.1 Hardware

| Component | Minimum (inference) | Recommended (training) |
|---|---|---|
| GPU | 8 GB VRAM (RTX 2070 / V100) | 24+ GB VRAM (RTX 5090 / A100 / H100) |
| RAM | 16 GB | 64 GB |
| Disk | 50 GB (inference + small data) | 250 GB (RSNA raw DICOM ~140 GB) |
| CPU | 4 cores | 16+ cores (DICOM I/O bound) |

**Training wall-clock estimate** on RTX 5090, sequential:
- DenseNet121: 5 folds × 80 epochs ≈ 2 days
- DenseNet169: 5 folds × 80 epochs ≈ 2-3 days
- SE-ResNeXt101: 5 folds × 80 epochs ≈ 3-4 days
- 3× `predict.py` (TTA inference for feature extraction) ≈ 1-2 days
- SequenceModel: 5 folds × 40 epochs ≈ 5-8 hours
- **Total: ~10-15 days**

### 2.2 Software environment

- **OS**: Ubuntu 22.04+ or macOS 13+ (verified). Windows not tested.
- **CUDA**: 11.8+ (PyTorch 2.0 requirement). For RTX 5090: CUDA 12.4+.
- **Python**: 3.10
- **Conda** (Miniconda or Anaconda)

### 2.3 Python dependencies

Exact versions (from [rsna_training/environment.yml](rsna_training/environment.yml) and [requirements.txt](requirements.txt)):

```yaml
# Training environment (rsna_training/environment.yml)
python=3.10
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
scikit-image>=0.21.0
scikit-learn>=1.3.0
scipy>=1.11.0
pandas>=2.0.0
numpy>=1.24.0,<2.0.0
pydicom>=2.4.0
pillow>=10.0.0
seaborn>=0.13.0
matplotlib>=3.7.0
tqdm>=4.66.0
joblib>=1.3.0
albumentations>=1.3.0
pretrainedmodels>=0.7.4
efficientnet-pytorch>=0.7.1
gdown>=5.2.0
kagglehub>=0.3.0
kaggle>=1.6.17
```

```text
# Inference environment (requirements.txt)
torch>=2.0
torchvision>=0.15
pydicom>=2.4
nibabel>=5.0
opencv-python>=4.8
numpy>=1.24
albumentations>=1.3
matplotlib>=3.7
kagglehub>=0.2
pretrainedmodels>=0.7.4
```

### 2.4 External credentials

- **Kaggle account** with API token (`~/.kaggle/kaggle.json`) — required to download RSNA dataset.
  Accept competition rules at: https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection
- **Google Drive access** — used by `gdown` to fetch SeuTao's pretrained weights and auxiliary CSVs. No login required (public files).
- **No PhysioNet account needed** for CT-ICH (publicly available).

---

## 3. Data sources and acquisition

### 3.1 Summary table

| Dataset | Role | Size | Source | Where to put it |
|---|---|---|---|---|
| RSNA 2019 IHD | Train hemorrhage CNN | ~140 GB DICOM, 21k studies, 750k slices, 6-label | Kaggle: `rsna-intracranial-hemorrhage-detection` | `rsna_training/data/rsna_raw/` |
| RSNA labels CSV | Hemorrhage train labels | 164 MB total | Google Drive (SeuTao's `csv.zip`) | `rsna_training/data/` |
| Fold splits | Train/val splits | 57 MB tekst | Google Drive (SeuTao's `data.zip`) | `rsna_training/data/fold_5_by_study*/` |
| Pretrained ImageNet weights | CNN init | ~150 MB | Auto-downloaded by torchvision/pretrainedmodels | `~/.cache/torch/` |
| CPAISD | Ischemic train | ~30 GB, 112 patients | https://zenodo.org/records/10892316 | `Bachelor_ischemic_test/data/cpaisd_multiplane/` |
| AISD | Ischemic train | ~30 GB, 397 patients | https://github.com/griffinliang/aisd | `Bachelor_ischemic_test/data/aisd_multiplane_hu/` |
| RSNA preprocessed slices | Ischemic neg controls | varies | Generated locally from RSNA DICOM (via `prepare_data.py`) | `Bachelor_ischemic_test/data/RSNA/stage_2_train_npy/` |
| CT-ICH | External hemorrhage eval | ~2 GB, 82 patients, JPGs | https://physionet.org/content/ct-ich/ | `Bachelor_RSNA_test/ct_ich_download/` |
| Brain Stroke CT (Kaggle) | External combined eval | ~3 GB, 6,636 DICOM slices | Kaggle: `ozguraslank/brain-stroke-ct-dataset` | `Bachelor_RSNA_test/brain_stroke_dataset/` |
| Hospital data | Clinical evaluation | ~100 patients | Zealand University Hospital, Department of Radiology | On-site workstation only — cannot be transferred |

### 3.2 Download commands

**RSNA dataset (via Kaggle CLI, requires accepted competition rules):**
```bash
cd rsna_training/
bash download_all.sh  # downloads RSNA + SeuTao auxiliary files
```

This wrapper invokes:
- `scripts/download_kaggle_data.py` — pulls RSNA stage_2_train and stage_2_test
- `scripts/download_gdrive.py` — pulls Google Drive bundles (SeuTao's CSVs + pretrained weights)
- `scripts/unpack_downloads.py` — extracts zip archives

**Google Drive bundles fetched** ([scripts/download_gdrive.py](rsna_training/scripts/download_gdrive.py)):

| File | Google Drive ID | Contents |
|---|---|---|
| `data.zip` | `1buISR_b3HQDU4KeNc_DmvKTYJ1gvj5-3` | Fold splits |
| `csv.zip` | `1qYi4k-DuOLJmyZ7uYYrnomU2U7MrYRBV` | Stage 1/2 cls CSVs |
| `feature_samples.zip` | `1lJgzZoHFu6HI4JBktkGY3qMk--28IUkC` | Sample features (for SM) |
| `seresnext101_256.zip` | `18Py5eW1E4hSbTT6658IAjQjJGS28grdx` | SeuTao's SE-ResNeXt101 pretrained weights (5 folds) |
| `densenet169_256.zip` | `1vCsX12pMZxBmuGGNVnjFFiZ-5u5vD-h6` | SeuTao's DenseNet169 pretrained weights (5 folds) |
| `densenet121_512.zip` | `1o0ok-6I2hY1ygSWdZOKmSD84FsEpgDaa` | SeuTao's DenseNet121 pretrained weights (5 folds) |

**CT-ICH** (PhysioNet, public):
```bash
wget -r --no-parent \
  https://physionet.org/files/ct-ich/1.0.0/ \
  -P Bachelor_RSNA_test/ct_ich_download/
```

**Kaggle Brain Stroke CT:**
```bash
kaggle datasets download -d ozguraslank/brain-stroke-ct-dataset \
  -p Bachelor_RSNA_test/brain_stroke_dataset/
unzip Bachelor_RSNA_test/brain_stroke_dataset/brain-stroke-ct-dataset.zip
```

**CPAISD:**
```bash
wget https://zenodo.org/records/10892316/files/cpaisd.zip \
  -P Bachelor_ischemic_test/data/
unzip Bachelor_ischemic_test/data/cpaisd.zip
# Then run the multiplane preprocessing script (project-specific):
python Bachelor_ischemic_test/prepare_cpaisd_multiplane.py
```

**AISD:**
```bash
git clone https://github.com/griffinliang/aisd Bachelor_ischemic_test/data/aisd
# Then run the multiplane preprocessing script:
python Bachelor_ischemic_test/prepare_aisd_multiplane.py
```

### 3.3 RSNA fold splits

The 5-fold cross-validation splits are **patient-level** (by `StudyInstanceUID`), pre-computed by SeuTao and included in `data.zip`. Two parallel directory structures:

- `data/fold_5_by_study/fold[0-4]/train.txt` — train studies (one per line)
- `data/fold_5_by_study_image/fold[0-4]/val.txt` — val slices (one filename per line)

**Important**: the random seed for the splits is hardcoded in SeuTao's code; the splits are deterministic and cannot be regenerated independently. Always use the files from `data.zip`.

---

## 4. Shared preprocessing

Source: [pipeline.py](pipeline.py).

### 4.1 DICOM → HU conversion

```python
def dicom_to_hu(dcm_path):
    ds = pydicom.dcmread(dcm_path, force=True)
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    slope = float(getattr(ds, "RescaleSlope", 1))
    img = ds.pixel_array.astype(np.float32) * slope + intercept
    return img  # 2D float32 array in HU
```

Defaults assumed if metadata missing: intercept=0, slope=1.

### 4.2 NIfTI → HU conversion

```python
img = nibabel.load(path)
img = nibabel.as_closest_canonical(img)  # reorient to RAS
data = img.get_fdata(dtype=np.float32)
```

Robustness:
- If 4D (perfusion / time-series), take first volume only.
- Sanity-check the slice axis: a real head CT spans 80-350 mm superior-inferior with ≤1.5 mm in-plane voxels. If the canonical Z-axis violates these, the code falls back to the axis with the largest voxel spacing as the slice axis (handles nnUNet-preprocessed NIfTIs with mislabeled affines).

### 4.3 Series filtering

DICOM folders may contain scouts/localizers/non-axial reformats. The filter ([pipeline.py:filter_dicom_series](pipeline.py)) keeps only:
- Series with ≥10 slices (eliminates scout images)
- SeriesDescription matching axial CT patterns
- ImageOrientationPatient consistent with axial acquisition

Disable with `--no-filter` if you've already pre-selected files.

### 4.4 Slice sorting

By `ImagePositionPatient[2]` (z-coordinate) ascending → inferior-to-superior.

### 4.5 Optional reslicing to 5 mm

For scans with <5 mm slice thickness, [reslice_to_thickness(images_hu, dz, target=5.0)](pipeline.py#L210) averages consecutive slices to reduce ensemble compute. **Disabled by default**; enable in code if needed.

---

## 5. Hemorrhage detection branch

### 5.1 Training data and split

- **Source**: RSNA 2019 Intracranial Hemorrhage Detection (Kaggle)
- **Size**: ~21,000 head CT studies, ~750,000 axial slices
- **Labels**: 6 binary classes per slice — `any, epidural, intraparenchymal, intraventricular, subarachnoid, subdural` (multi-label; "any" is positive if any subtype is positive)
- **Split**: 5-fold cross-validation by `StudyInstanceUID` (patient-level), from `data.zip`
- **Sizes per fold** (approximate):
  - Train: ~15,624 studies → ~600,000 slices
  - Val: ~134,788 slices

### 5.2 PNG conversion — `2DNet/src/prepare_data.py`

Each DICOM is converted to a single grayscale PNG. **The window used is the DICOM file's own `WindowCenter`/`WindowWidth` from metadata** (typically a brain window for RSNA — center=40, width=80).

```python
def window_image(img, window_center, window_width, intercept, slope):
    img = img.astype(np.float64) * slope + intercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img

def prepare_image(img_path):
    img_dicom = pydicom.dcmread(img_path)
    metadata = {
        "window_center": img_dicom.WindowCenter,
        "window_width": img_dicom.WindowWidth,
        "intercept": img_dicom.RescaleIntercept,
        "slope": img_dicom.RescaleSlope,
    }
    img = window_image(img_dicom.pixel_array, **metadata)
    img = normalize_minmax(img) * 255  # → [0, 255]
    img = PIL.Image.fromarray(img.astype(np.uint8))
    return img_id, img
```

**This is NOT three different HU windows per slice.** Each PNG is one slice, one window.

Command:
```bash
cd rsna_training/external/SeuTao_repo/2DNet/src
python3 prepare_data.py -dcm_path /path/to/stage_2_train -png_path /path/to/output_train_png
python3 prepare_data.py -dcm_path /path/to/stage_2_test -png_path /path/to/output_test_png
```

### 5.3 3-slice concatenation (precomputed) — `scripts/create_concat_images.py`

Builds the prev/curr/next stacked images that are used during inference's `predict.py` (not used during training — training assembles them on the fly):

```python
prev_f = files[i - 1] if i > 0 else fname
next_f = files[i + 1] if i < len(files) - 1 else fname

img_prev = cv2.imread(str(png_dir / prev_f), 0)
img_cur = cv2.imread(str(png_dir / fname), 0)
img_next = cv2.imread(str(png_dir / next_f), 0)

# Resize each to 512×512, then merge into 3-channel BGR-equivalent
img_merged = cv2.merge([img_prev, img_cur, img_next])
img_merged = cv2.resize(img_merged, (256, 256))
cv2.imwrite(str(out_path), img_merged)
```

Output: 256×256 3-channel PNGs in `data/train_concat_3images_256/` and `data/stage2_test_concat_3images/`.

### 5.4 Training input — `2DNet/src/dataset/dataset.py`

For each target slice during training, the dataset reads three adjacent PNGs (prev/curr/next) on the fly and stacks them as channels:

```python
class RSNA_Dataset_train_by_study_context(data.Dataset):
    def __getitem__(self, idx):
        # ... pick a random slice from the study ...
        image = _safe_imread(train_png_dir + filename)        # current
        image_up = _safe_imread(train_png_dir + filename_up)  # next slice (s+1)
        image_down = _safe_imread(train_png_dir + filename_down)  # prev slice (s-1)

        image_cat = np.concatenate(
            [image_up[:,:,np.newaxis], image[:,:,np.newaxis], image_down[:,:,np.newaxis]],
            axis=2
        )
        # → (512, 512, 3) where channels = (next, current, prev)

        # 50% chance to permute channels via BGR↔RGB
        if random.random() < 0.5:
            image_cat = cv2.cvtColor(image_cat, cv2.COLOR_BGR2RGB)

        image_cat = aug_image(image_cat, is_infer=False)  # apply augmentations
        image_cat = transform(image_cat)['image']         # resize + normalize
        return image_cat.transpose(2, 0, 1), label
```

Channel ordering: **(next=s+1, current=s, previous=s-1)**. Boundary slices clamp to the current slice. Each merged image is resized to 256×256 and normalised with ImageNet statistics (mean=0.456, std=0.224 — single-channel statistics applied to all 3 channels because the channels are grayscale slices).

> SeuTao's documentation diagram (`docs/overview.png`) shows three different HU windows (40/80, 80/200, 600/2800) of the same slice. The released codebase implements three adjacent slices with the DICOM default window. The pretrained weights were produced by the released code. **Follow the code, not the diagram.**

### 5.5 Training augmentations — `dataset.py:aug_image`

Applied per-batch during training (`is_infer=False`):

| Augmentation | Probability | Parameters |
|---|---|---|
| Horizontal flip | 0.5 | — |
| Affine shift+scale+rotate | always | shift_limit=0.1, scale_limit=0.1, rotate_limit=30° |
| Random erasing | 0.5 | area 2-40% of image, aspect ratio 0.3-3.3 |
| Random crop | 1.0 | ratio 0.6-0.99, then resize back |

At inference (`is_infer=True`): only 80% center crop.

Final transform applied after `aug_image`:
```python
albumentations.Compose([
    albumentations.Resize(image_size, image_size),  # 256×256
    albumentations.Normalize(
        mean=(0.456, 0.456, 0.456),
        std=(0.224, 0.224, 0.224),
        max_pixel_value=255.0,
        p=1.0
    )
])
```

### 5.6 Model architectures — `2DNet/src/net/models.py`

All three architectures follow the pattern: pretrained backbone → adaptive avg-pool → linear(6).

**DenseNet121_change_avg** ([models.py:66-82](rsna_training/external/SeuTao_repo/2DNet/src/net/models.py)):
```python
class DenseNet121_change_avg(nn.Module):
    def __init__(self):
        super().__init__()
        self.densenet121 = torchvision.models.densenet121(weights='IMAGENET1K_V1').features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1024, 6)
    def forward(self, x):
        x = self.densenet121(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        return self.mlp(x)
```

**DenseNet169_change_avg** — same but `densenet169`, `mlp = nn.Linear(1664, 6)`.

**se_resnext101_32x4d** — uses `pretrainedmodels` package:
```python
class se_resnext101_32x4d(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_ft = pretrainedmodels.__dict__['se_resnext101_32x4d'](
            num_classes=1000, pretrained=None
        )
        num_ftrs = self.model_ft.last_linear.in_features  # 2048
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 6, bias=True))
    def forward(self, x):
        return self.model_ft(x)
```

**Activation**: training applies `BCEWithLogitsLoss` to raw logits. Inference applies `torch.sigmoid` to get probabilities.

### 5.7 Training hyperparameters — `2DNet/src/train.py`

```python
# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.00002
)

# Scheduler — Cosine Annealing with Warm Restarts
scheduler = WarmRestart(optimizer, T_max=5, T_mult=1, eta_min=1e-5)
# At epoch 10+: scheduler.step() + warm_restart(scheduler, T_mult=2) per epoch
# At epoch 80+: optimizer.param_groups[0]['lr'] = 1e-5 (fixed)

# Loss
loss_cls = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.FloatTensor([1.0]*6).cuda()
)

# Training schedule
trMaxEpoch = 80
train_batch_size = 32  # SeuTao README suggests 256 for DenseNet121/169, 80 for SE-ResNeXt101 on multi-GPU
val_batch_size = 32
workers = 24

# Determinism
torch.manual_seed(1992)
torch.cuda.manual_seed(1992)
np.random.seed(1992)
random.seed(1992)
torch.backends.cudnn.benchmark = True  # not deterministic; small CUDA drift expected
```

### 5.8 Validation policy

Original schedule (unchanged for best-model selection):
- **Full validation** at epoch 0, every 5 epochs, and epoch 79+ on entire ~134k-slice val set
- Used for best-model checkpoint selection
- Checkpoint saved every 5 epochs + epoch 79 + at every new best val AUC

Added per-epoch light validation (responds to advisor's request for a continuously-tracked validation metric):
- **Light validation** every epoch on a fixed 5000-slice subset (~3% of val.txt)
- CSV columns prefixed `Light_*` and `LightVal*`
- Predictions saved as `epoch_<E>_lightval_predictions.npz` for the learning curve
- Adds ~2-3 min per epoch on RTX 5090 (~3 extra hours across the full run)
- Does **NOT** influence best-model selection — that remains driven by full val

**No early stopping** — the Warm Restart schedule requires the full 80 epochs to complete its restart cycles. Best-model checkpoint selection is done via our new instrumentation (`model_best_<fold>.pth`).

### 5.9 Training instrumentation (our addition) — `shared/metric_logger.py`

The training procedure itself is **unchanged**; only metric logging was added. Per fold per epoch, the instrumentation produces:

- **`log.csv`** (~110 columns): epoch, LR, time, GPU memory, throughput, gradient norms, train loss, val loss, and per-class (× 6 hemorrhage labels):
  - AUC, PR-AUC, precision/recall/F1/sensitivity/specificity at threshold 0.5
  - Youden-optimal threshold + sensitivity/specificity at it
  - F1-optimal threshold + F1 at it
  - TP/FP/TN/FN at threshold 0.5
  - Macro-averaged AUC, PR-AUC, F1
- **`epoch_<E>_val_predictions.npz`**: raw `outGT` + `outPRED` arrays (recompute any metric post-hoc)
- **`epoch_<E>_metrics.json`**: structured nested metrics (including calibration data)
- **`train_metadata.json`**: hyperparameters + dataset info (written once at fold start)
- **`best_epoch.json`**, **`model_best_<N>.pth`**: best val AUC checkpoint
- **`log_legacy.csv`**: SeuTao's original CSV format preserved as fallback

### 5.10 Training command (full schedule)

```bash
cd rsna_training/external/SeuTao_repo/2DNet/src

# DenseNet121
python3 train.py \
  -backbone DenseNet121_change_avg \
  -img_size 256 -tbs 32 -vbs 32 \
  -save_path DenseNet121_change_avg_256

# DenseNet169
python3 train.py \
  -backbone DenseNet169_change_avg \
  -img_size 256 -tbs 32 -vbs 32 \
  -save_path DenseNet169_change_avg_256

# SE-ResNeXt101
python3 train.py \
  -backbone se_resnext101_32x4d \
  -img_size 256 -tbs 32 -vbs 32 \
  -save_path se_resnext101_32x4d_256
```

Each invocation runs all 5 folds internally. Output: `data_test/<save_path>/fold<N>/{log.csv, model_best_<N>.pth, ...}`.

Or use the wrapper:
```bash
cd rsna_training/
bash run_full_training.sh         # full
bash run_full_training.sh --smoke # 1 fold × 2 epochs smoke test (10-15 min)
```

### 5.11 Inference preprocessing (deployed) — `pipeline.py:prepare_hemorrhage_input_series`

```python
def prepare_hemorrhage_input_series(images_hu, idx):
    prev_idx = max(0, idx - 1)
    next_idx = min(len(images_hu) - 1, idx + 1)

    ch_prev = apply_window(images_hu[prev_idx], center=40, width=80)
    ch_curr = apply_window(images_hu[idx],     center=40, width=80)
    ch_next = apply_window(images_hu[next_idx], center=40, width=80)

    ch_prev = cv2.resize(ch_prev, (512, 512))
    ch_curr = cv2.resize(ch_curr, (512, 512))
    ch_next = cv2.resize(ch_next, (512, 512))

    return np.stack([ch_prev, ch_curr, ch_next], axis=-1)  # (512, 512, 3) uint8
```

For a single isolated slice (e.g. Kaggle dataset where DICOM patient series are not grouped), [prepare_hemorrhage_input](pipeline.py#L265) replicates the current slice into all 3 channels.

**Inference image size: 512×512** (not 256×256 as during training). This mirrors SeuTao's `predict.py` convention — the CNN is fully convolutional and accepts any input size; the 2× larger inference resolution often gives a small boost.

**Normalization** (applied after preprocessing, before forward pass):
```python
albumentations.Normalize(
    mean=(0.456, 0.456, 0.456),
    std=(0.224, 0.224, 0.224),
    max_pixel_value=255.0,
)
```

### 5.12 Inference flow — `pipeline.py:predict_hemorrhage_batch`

1. **Load 15 models** from `models/hemorrhage/<backbone>/model_epoch_79_<fold>.pth`. Layout:
   ```
   models/hemorrhage/
     DenseNet121/
       model_epoch_79_0.pth
       model_epoch_79_1.pth
       ...
       model_epoch_79_4.pth
     DenseNet169/   (5 files)
     SE-ResNeXt101/ (5 files)
   ```
   Falls back to a flat DenseNet121-only layout if subdirectories are missing.

2. **Per-batch inference**: each of the 15 models runs forward, then `sigmoid()`. The 15 probability vectors are averaged → 6-dim probability per slice.

3. **Per-class threshold** (Youden-optimal, see Section 10):
   ```
   any: 0.3715, epidural: 0.0247, intraparenchymal: 0.1738,
   intraventricular: 0.1018, subarachnoid: 0.1967, subdural: 0.2191
   ```

4. **Scan-level aggregation**: max probability across all slices per label. A scan is flagged hemorrhage-positive for label X if at least one slice exceeds threshold X.

5. The "any" label drives the overall hemorrhage flag; subtype probabilities are retained for interpretability.

Code reference: [pipeline.py:predict_hemorrhage_batch](pipeline.py#L513).

---

## 6. Ischemic stroke classification branch

### 6.1 Training data — combined set

| Source | Train | Val | Test | Positive rate |
|---|---|---|---|---|
| CPAISD | 8,376 | 980 | 809 | ~30% positive |
| AISD | 8,099 | 1,357 | 1,506 | ~37% positive |
| RSNA controls | 5,000 | 1,000 | 2,000 | 0% (all neg) |

Positive labels derived from segmentation masks: a slice is positive if its mask contains any nonzero voxel. RSNA controls are random slices from `stage_2_train_npy/` treated as ischemia-negative (includes both healthy and hemorrhage cases — they are *non-ischemic*, not *normal*).

### 6.2 Training preprocessing — `Bachelor_ischemic_test/train_ischemic_robust.py:stack_prev_curr_next`

The **deployed** ischemic model (`models/ischemic/best_model.pth`, md5: `bc1fb9f6f50978c66ff4119cd55e3241`) was trained with prev/curr/next adjacent slices, all brain-windowed:

```python
def stack_prev_curr_next(slices_hu, idx, center=40, width=80):
    n = len(slices_hu)
    prev_hu = slices_hu[idx - 1] if idx > 0 else slices_hu[idx]
    curr_hu = slices_hu[idx]
    next_hu = slices_hu[idx + 1] if idx < n - 1 else slices_hu[idx]
    ch_prev = apply_window(prev_hu, center, width)
    ch_curr = apply_window(curr_hu, center, width)
    ch_next = apply_window(next_hu, center, width)
    return np.stack([ch_prev, ch_curr, ch_next], axis=-1)
```

- All three channels use brain window (center=40, width=80)
- Resized to **256×256**
- Normalised with ImageNet statistics (same as hemorrhage)

For RSNA controls (no volumetric context available — slices stored as individual `.npy` files), the current slice is replicated into all 3 channels.

> **Report discrepancy.** Section 5.3.2 describes ischemic input as *three different HU windows of the same slice* (40/80, 32/8, 40/120). That setup exists in `train_ischemic_transfer.py` but its checkpoint (`results_ischemic_transfer/best_model.pth`, md5: `df4a0f2f...`) is NOT the deployed one. The deployed checkpoint uses prev/curr/next + single brain window 40/80, identical to the hemorrhage input format.

### 6.3 Architecture and transfer learning

```python
class IschemicClassifier(nn.Module):
    def __init__(self, pretrained_densenet_features):
        super().__init__()
        self.features = pretrained_densenet_features  # from hemorrhage CNN
        self.relu = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.dropout(x)
        return self.fc(x).squeeze(1)
```

**Initialization**: from hemorrhage-pretrained DenseNet121 fold with lowest val loss (specifically `Bachelor_RSNA_test/models/DenseNet121/model_epoch_79_4.pth`).

**Freezing strategy** ([train_ischemic_robust.py:138-141](Bachelor_ischemic_test/train_ischemic_robust.py)):
```python
for name, param in classifier.features.named_parameters():
    if "denseblock4" in name or "norm5" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False
```
Only the deepest DenseBlock4 + final norm + new classification head are trained. Approximately 2.5M trainable / 7M total parameters.

### 6.4 Training hyperparameters

```python
SEED = 42
IMG_SIZE = 256
BATCH_SIZE = 32
NUM_EPOCHS = 25
LR = 1e-4
WD = 1e-5
```

| Parameter | Value |
|---|---|
| Optimizer | Adam, lr=1e-4, weight_decay=1e-5 |
| Scheduler | Cosine annealing over 25 epochs |
| Loss | BCEWithLogitsLoss with **dynamic positive class weight** (computed from actual train class ratio per epoch) |
| Class balancing | 3:1 negative-to-positive downsampling per epoch |
| Augmentation | Horizontal flip (p=0.5), in-plane rotation ±10° |
| Best checkpoint | Tracked by val AUC |
| Bootstrap CI | 2000 resamples on test predictions |

### 6.5 Inference preprocessing (deployed) — `pipeline.py:prepare_ischemic_input`

```python
def prepare_ischemic_input(image_hu):
    brain = apply_window(image_hu, center=40, width=80)
    brain = cv2.resize(brain, (256, 256))
    return np.stack([brain, brain, brain], axis=-1)  # SAME slice × 3
```

> **Inference-vs-training discrepancy.** Even when `pipeline.py:run_pipeline_batched` has multiple slices available, it calls `prepare_ischemic_input(images_hu[i])` per slice ([pipeline.py:643](pipeline.py#L643)), which replicates the current slice instead of passing (prev, curr, next). The model was trained with real prev/curr/next context but is deployed without it. This is likely a contributing factor to the weak external Kaggle AUC (0.591) reported in the thesis.
>
> **Fix**: replace the per-slice call with one that builds the 3-slice stack from `images_hu` (mirror `prepare_hemorrhage_input_series`).

### 6.6 Inference flow — `pipeline.py:predict_ischemic_batch`

1. **Load 1 model** from `models/ischemic/best_model.pth`.
2. **Per-batch inference**: model forward → `sigmoid` → 1 probability per slice.
3. **Threshold**: 0.5 (fixed, not Youden-optimised).
4. **Scan-level aggregation**: max probability across all slices. Flag positive if max ≥ 0.5.

---

## 7. Sequence Model (planned addition)

**NOT in the current `pipeline.py`** — the deployed system uses only the CNN ensemble. Re-training the Sequence Model is part of the upcoming hospital re-training.

### 7.1 Architecture overview

SeuTao's original solution included two cascaded sequence models, implemented as a single combined model in [`rsna_training/external/SeuTao_repo/SequenceModel/seq_model.py`](rsna_training/external/SeuTao_repo/SequenceModel/seq_model.py):

**Sequence Model 1** (auxiliary, "logit_help"):
- Input: per-slice CNN feature vectors stacked into a sequence per study
- Architecture: FC → FC → FC → bidirectional GRU (2 layers) → Linear + skip-connection → sequence logits
- Output: refined 6-dim probability per slice (with cross-slice context)

**Sequence Model 2** (final, "logit"):
- Input: concatenation of (CNN classifier output 6-dim + SM1 output 6-dim + slice position diff "position 2") per slice
- Architecture: 1D CNN (4 × Conv1D) → bidirectional GRU (2 layers, 96 hidden) → Linear + skip-connection → final sequence logits
- Output: final 6-dim probability per slice (with full study context)

Both are trained jointly with `loss = loss0 + loss1` where `loss0` is SM2's loss and `loss1` is SM1's.

### 7.2 Training data flow

1. After CNN training completes, run `predict.py` for each of the 3 backbones to generate:
   - Per-fold predictions: `val_aug_10.csv`, `test_aug_10.csv` (TTA-averaged, 10 augmentations)
   - Per-slice intermediate features: `npy_train/*.npy`, `npy_test/*.npy` (128-dim or larger feature vectors)

2. Run `scripts/build_sequence_inputs.py` to assemble:
   - `SequenceModel/features/stage2_finetune/<backbone>/<backbone>_val_oof_feature_TTA.npy` — out-of-fold features
   - `SequenceModel/features/stage2_finetune/<backbone>/<backbone>_test_feature_TTA_stage2.npy` — test features
   - `<backbone>_val_prob_TTA.csv` and `<backbone>_test_prob_TTA_stage2.csv` — TTA-averaged probabilities

3. The Sequence Model loads these for all 3 backbones and trains on the combined feature set.

### 7.3 Sequence Model training hyperparameters

```python
fold_num = 5
Add_position = True       # use slice position-2 feature
lstm_layers = 2
seq_len = 24               # training sequence length (slices)
hidden = 96
drop_out = 0.5
train_epoch = 40
class_num = 6
batch_size = 128           # training; val batch_size = 1 (one study at a time)

optimizer = optim.Adam(model.parameters(), lr=3e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 30, 40], gamma=0.1)

# Loss: weighted BCE, weight [2,1,1,1,1,1] for "any" hemorrhage emphasis
def criterion(logit, labels):
    w = [2.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    loss = [F.binary_cross_entropy_with_logits(
                logit[:, 0, :, i:i+1], labels[:,:, i:i+1]) * w[i]
            for i in range(6)]
    return sum(loss) / sum(w)
```

### 7.4 Training command

```bash
cd rsna_training/external/SeuTao_repo/SequenceModel
python3 main.py
# Produces: FinalSubmission/version3_debug/fold_<N>.pt for N in 0..4
```

Or skip `valid()` and `inference()` phases (useful during smoke test):
```bash
python3 main.py --skip-valid
```

### 7.5 Integration into ct-brain-pipeline

After SM training is done, deploying it requires:
1. Add CNN feature extraction in `pipeline.py` (hook into DenseNet's intermediate layer matching what `build_sequence_inputs.py` used)
2. Load 5 SM checkpoints from `FinalSubmission/version3_debug/fold_*.pt`
3. After getting CNN slice-level logits, feed them + features + slice positions through SM2
4. Use SM2's output (averaged across 5 folds) instead of CNN max-pool for scan-level aggregation

This integration step is **not yet implemented**.

---

## 8. Combined prioritization decision

In [pipeline.py:summarise_patient](pipeline.py):

```
scan flagged for prioritized review
    ⇔
(hemorrhage_any_max ≥ 0.3715) OR (ischemia_max ≥ 0.5)
```

The two branches are **independent thresholded outputs**, not a single fused probability. This preserves interpretability: a flagged scan can be traced back to hemorrhage-driven, ischemia-driven, or both-driven. Trade-off: the combined false-positive rate is roughly the union of each branch's false-positive rate.

---

## 9. External evaluation

### 9.1 CT-ICH (PhysioNet) — hemorrhage external validation

- **Size**: 82 patients, brain-windowed JPGs (NOT native DICOM with full HU)
- **Annotation**: hemorrhage subtype labels + segmentation masks
- **Source**: Al Hilla Teaching Hospital, Iraq — independent of RSNA
- **Evaluation level**: patient-level (aggregate slice probabilities via max)
- **Script**: [Bachelor_RSNA_test/run_ctich_full_ensemble.py](Bachelor_RSNA_test/run_ctich_full_ensemble.py)

**Workflow**:
1. Load all 15 hemorrhage models
2. For each patient, load all JPG slices
3. Convert JPG → 3-channel by replicating (no HU data → no windowing applied)
4. Run all 15 models, average sigmoids → 6-dim per slice
5. Aggregate per patient: max probability per label
6. Apply Youden thresholds → patient-level positive/negative per label
7. Compute AUC, sensitivity, specificity per label

**Expected results** (from thesis Table 3):
- Any-hemorrhage AUC: 0.952, sensitivity 1.000, specificity 0.435
- Best subtype: intraventricular (AUC 0.997)
- Worst subtype: epidural (AUC 0.818)

### 9.2 Kaggle Brain Stroke CT — combined external validation

- **Size**: 6,636 individual DICOM slices: 1,093 bleeding / 1,116 ischemia / 4,427 normal
- **Limitation**: slices are NOT grouped by patient — only slice-level evaluation possible
- **Source**: Kaggle dataset `ozguraslank/brain-stroke-ct-dataset`
- **Script**: [Bachelor_RSNA_test/run_brain_stroke_full.py](Bachelor_RSNA_test/run_brain_stroke_full.py)

**Workflow**:
1. Load all 15 hemorrhage models + 1 ischemic model
2. Per slice: DICOM → HU → preprocessing → both branches → probabilities
3. Slice-level metrics (no patient grouping)

**Expected results** (from thesis Table 4):
- Hemorrhage (Bleeding vs Normal): AUC 0.934
- Ischemia (Ischemia vs Normal): AUC 0.591 — weak (see Section 6.5 inference discrepancy)
- Any-stroke (Bleeding+Ischemia vs Normal): AUC 0.647

### 9.3 Hospital data (Zealand University Hospital)

- **Size**: ~100 patients, native DICOM
- **Pending**: full-pipeline evaluation runs on-site only (data cannot leave the hospital)
- **Metrics planned**: patient-level AUC, sensitivity, specificity, F1 per label + overall stroke prioritization

---

## 10. Threshold derivation

**Internal validation** (during training) is used to choose the operating thresholds applied at inference.

### 10.1 Youden-optimal threshold computation

From [run_brain_stroke_full.py:181](Bachelor_RSNA_test/run_brain_stroke_full.py):

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_true, y_score)
youden_index = tpr - fpr             # Youden's J statistic
best_idx = int(np.argmax(youden_index))
youden_threshold = thresholds[best_idx]
youden_sensitivity = tpr[best_idx]
youden_specificity = 1 - fpr[best_idx]
```

### 10.2 Hemorrhage thresholds (as deployed)

These are baked into `pipeline.py:HEMORRHAGE_THRESHOLDS`:

| Label | Youden threshold |
|---|---|
| any | 0.3715 |
| epidural | 0.0247 |
| intraparenchymal | 0.1738 |
| intraventricular | 0.1018 |
| subarachnoid | 0.1967 |
| subdural | 0.2191 |

**Caveat**: these were derived from a specific internal validation set. They may not transfer optimally to external datasets (CT-ICH, Kaggle, hospital). The thesis discussion explicitly addresses this — practical deployment may require **local threshold calibration** before clinical use.

### 10.3 Ischemic threshold

Fixed at **0.5** (default sigmoid cutoff). Not Youden-optimised — a tunable parameter for future work.

---

## 11. Known discrepancies with the thesis report

These should be addressed in the report after the hospital re-training:

| # | Report claim | Code reality | Section to update |
|---|---|---|---|
| 1 | "prev/curr/next instead of stacked HU" framed as a deviation from SeuTao | Both jeres reproduction AND SeuTao's released code use prev/curr/next. SeuTao's *diagram* shows stacked HU, but his code does not. | 5.2.1, 5.2.2 |
| 2 | Hemorrhage pipeline excludes Sequence Model | True now; will change after the upcoming re-training | 5.2.1, 5.2.4 — add SM section once trained |
| 3 | Ischemic input: three HU windows (40/80, 32/8, 40/120) of same slice | Deployed model was trained with prev/curr/next + single brain window 40/80. The multi-window setup exists in `train_ischemic_transfer.py` but its checkpoint is not deployed. | 5.3.2 |
| 4 | Iskæmi-input at inference matches training | At inference, `prepare_ischemic_input` replicates one slice 3× instead of using prev/curr/next from the scan. | 5.3.2 + Discussion |
| 5 | Iskæmi class weighting "is not adopted from the original SeuTao pipeline, which uses a uniform loss weight" | Misleading — SeuTao never trained an ischemic classifier. Reformulate as a design choice for ischemic, not a deviation. | 3.4, 5.3.4 |
| 6 | Hemorrhage uses 5-fold ensemble at inference | True for *trained* models, but [predict.py:363](rsna_training/external/SeuTao_repo/2DNet/src/predict.py#L363) iterates `[1,2,3,4]` — fold 0 is excluded from prediction averaging. The deployed pipeline.py loads all 5 folds, so this only affects re-running predict.py during SM feature extraction. | Mention in 5.2.5 (or fix the code) |

---

## 12. Step-by-step reproduction guide

### 12.1 Phase A — One-time setup

```bash
# 1. Clone the repository
git clone https://github.com/Voglhofer/ct-brain-pipeline.git
cd ct-brain-pipeline/

# 2. Create the training environment (Python 3.10)
cd rsna_training/
conda env create -f environment.yml
conda activate rsna-seutao

# 3. Configure Kaggle credentials
mkdir -p ~/.kaggle
cp /your/secure/path/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
# Accept RSNA competition rules at
# https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection

# 4. Download RSNA + SeuTao auxiliary files (~140 GB DICOM + ~3 GB aux)
bash download_all.sh

# 5. Verify the setup
bash smoke_test.sh
```

After this you should have:
```
rsna_training/
  data/
    rsna_raw/stage_2_train/         # ~140 GB DICOM
    rsna_raw/stage_2_test/
    train_png/                       # ~30 GB PNG (after Phase B step 2)
    test_png/
    train_concat_3images_256/        # ~10 GB (after Phase B step 3)
    stage2_test_concat_3images/
    fold_5_by_study/fold0..4/        # text files
    fold_5_by_study_image/fold0..4/  # text files
    stage1_train_cls.csv             # 131 MB
    stage1_test_cls.csv              # 13 MB
    stage2_test_cls.csv              # 20 MB
  downloads/
    pretrained/                      # SeuTao's pretrained .zip's
    auxiliary/
```

### 12.2 Phase B — Hemorrhage CNN re-training (10-15 days)

```bash
cd rsna_training/

# 1. Smoke test (10-15 min)
bash run_full_training.sh --smoke
ls external/SeuTao_repo/2DNet/src/data_test/DenseNet121_change_avg_256_smoke/fold0/
# Expect: log.csv with ~110 columns, epoch_000_*.npz, epoch_000_*.json,
#         train_metadata.json, best_epoch.json, model_best_0.pth

# 2. Full training in background (logs to file)
nohup bash run_full_training.sh > full_run.log 2>&1 &
disown

# 3. Monitor
tail -f full_run.log
tail -f external/SeuTao_repo/2DNet/src/data_test/DenseNet121_change_avg_256/fold0/log.csv
```

The wrapper script runs sequentially:
1. DenseNet121 training (5 folds × 80 epochs)
2. DenseNet169 training
3. SE-ResNeXt101 training
4. `predict.py` × 3 backbones (TTA inference → predictions + features)
5. `build_sequence_inputs.py`
6. `SequenceModel/main.py` training
7. `analyze_training.py` → plots + LaTeX tables

### 12.3 Phase C — Ischemic re-training (separate from CNN, ~1-2 days)

This is **not** part of `run_full_training.sh`. To re-train:

```bash
cd Bachelor_ischemic_test/

# Ensure CPAISD, AISD, and RSNA NPY data exist (see Section 3.2)
python3 train_ischemic_robust.py
# Output: results_ischemic_robust_3slice/best_model.pth + metrics
```

Then copy the new checkpoint to the inference location:
```bash
cp results_ischemic_robust_3slice/best_model.pth ../ct-brain-pipeline/models/ischemic/best_model.pth
```

### 12.4 Phase D — Deploy trained models into ct-brain-pipeline

```bash
cd ct-brain-pipeline/

# Hemorrhage: collect best checkpoints from each (backbone, fold) combination
mkdir -p models/hemorrhage/DenseNet121 models/hemorrhage/DenseNet169 models/hemorrhage/SE-ResNeXt101

for fold in 0 1 2 3 4; do
  cp rsna_training/external/SeuTao_repo/2DNet/src/data_test/DenseNet121_change_avg_256/fold${fold}/model_best_${fold}.pth \
     models/hemorrhage/DenseNet121/model_epoch_79_${fold}.pth
  cp rsna_training/external/SeuTao_repo/2DNet/src/data_test/DenseNet169_change_avg_256/fold${fold}/model_best_${fold}.pth \
     models/hemorrhage/DenseNet169/model_epoch_79_${fold}.pth
  cp rsna_training/external/SeuTao_repo/2DNet/src/data_test/se_resnext101_32x4d_256/fold${fold}/model_best_${fold}.pth \
     models/hemorrhage/SE-ResNeXt101/model_epoch_79_${fold}.pth
done
```

(The filename `model_epoch_79_<N>.pth` is hardcoded in `pipeline.py`. The legacy SeuTao convention is to rename the best-AUC checkpoint to `model_epoch_best_<N>.pth`, but our pipeline uses the simpler `model_epoch_79_<N>.pth` pattern.)

### 12.5 Phase E — Run inference

```bash
cd ct-brain-pipeline/

# Single DICOM file
python pipeline.py /path/to/scan.dcm

# DICOM folder (auto-filter axial brain CT series)
python pipeline.py /path/to/patient_dicom_folder/

# NIfTI volume
python pipeline.py /path/to/scan.nii.gz

# Disable series filtering (use all DICOMs)
python pipeline.py /path/to/folder/ --no-filter

# Save per-slice visualization
python pipeline.py /path/to/folder/ --visualize
```

Output (in `output/`):
- `<scan_id>_slice_<N>.json` — per-slice probabilities for both branches
- `<scan_id>_patient_summary.json` — aggregated patient-level decision
- `<scan_id>_visualization.png` — (if `--visualize`)
- Console summary with flag + probabilities

### 12.6 Phase F — External evaluation

```bash
cd Bachelor_RSNA_test/

# CT-ICH (hemorrhage external)
python run_ctich_full_ensemble.py
# → ct_ich_results_full_ensemble/{all_predictions.csv, metrics.json, roc_curves.png}

# Kaggle Brain Stroke CT (combined external)
python run_brain_stroke_full.py
# → brain_stroke_results_full/{predictions.csv, metrics.json, roc_curves.png}
```

---

## 13. Repository layout

```
ct-brain-pipeline/                     ← deployed pipeline (this repo)
├── PIPELINE_OVERVIEW.md               ← this document
├── README.md
├── pipeline.py                        ← main inference entry point
├── evaluate_cq500.py                  ← (legacy, not in current results)
├── evaluate_ctich.py                  ← (laptop wrapper; full version is in Bachelor_RSNA_test/)
├── evaluate_kaggle.py                 ← (laptop wrapper)
├── run_all.sh
├── requirements.txt                   ← inference deps
├── models/
│   ├── hemorrhage/
│   │   ├── DenseNet121/   model_epoch_79_{0..4}.pth
│   │   ├── DenseNet169/   model_epoch_79_{0..4}.pth
│   │   └── SE-ResNeXt101/ model_epoch_79_{0..4}.pth
│   └── ischemic/
│       └── best_model.pth
├── output/                            ← inference outputs (.gitignored)
└── rsna_training/                     ← training subtree, deployed to hospital
    ├── HOSPITAL_README.md
    ├── README.md
    ├── config.env.example
    ├── environment.yml
    ├── download_all.sh
    ├── run_full_training.sh
    ├── run_full_pipeline.sh
    ├── smoke_test.sh
    ├── data/                          ← .gitignored, populated by download_all.sh
    ├── downloads/                     ← .gitignored
    ├── scripts/
    │   ├── download_kaggle_data.py
    │   ├── download_gdrive.py
    │   ├── unpack_downloads.py
    │   ├── patch_settings.py
    │   ├── create_concat_images.py
    │   ├── install_pretrained_weights.py
    │   ├── build_sequence_inputs.py
    │   └── verify_setup.py
    └── external/SeuTao_repo/          ← modified SeuTao codebase
        ├── analyze_training.py
        ├── shared/metric_logger.py    ← our metric instrumentation
        ├── 2DNet/
        │   └── src/
        │       ├── train.py           ← modified
        │       ├── predict.py
        │       ├── prepare_data.py
        │       ├── settings.py
        │       ├── dataset/dataset.py
        │       ├── net/{models.py, common.py}
        │       └── tuils/{tools.py, loss_function.py, lrs_scheduler.py}
        ├── SequenceModel/
        │   ├── main.py                ← modified
        │   ├── seq_model.py
        │   ├── seq_dataset.py
        │   ├── settings.py
        │   ├── check_feature.py
        │   └── check_oof.py
        └── 3DNet/                     ← experimental, not used

Bachelor_RSNA_test/                    ← thesis-side scripts (separate repo)
├── models/                            ← original 15 CNN checkpoints
│   ├── DenseNet121/model_epoch_79_{0..4}.pth
│   ├── DenseNet169/model_epoch_79_{0..4}.pth
│   └── SE-ResNeXt101/model_epoch_79_{0..4}.pth
├── ct_ich_download/                   ← CT-ICH dataset
├── brain_stroke_dataset/              ← Kaggle Brain Stroke CT
├── run_ctich_full_ensemble.py         ← CT-ICH external eval
├── run_brain_stroke_full.py           ← Kaggle external eval
└── run_combined_pipeline.py           ← (legacy version of pipeline.py)

Bachelor_ischemic_test/                ← ischemic-side scripts (separate repo)
├── train_ischemic_robust.py           ← producer of deployed best_model.pth
├── train_ischemic_transfer.py         ← multi-window variant (not deployed)
├── train_ischemic_no_rsna.py
├── eval_kaggle_3slice.py
├── eval_kaggle_full.py
├── data/
│   ├── cpaisd_multiplane/
│   ├── aisd_multiplane_hu/
│   └── RSNA/stage_2_train_npy/
└── results_ischemic_robust_3slice/    ← contains the deployed best_model.pth
```

---

## 14. End-to-end command reference

### 14.1 Inference on a single scan

```bash
cd ct-brain-pipeline/
python pipeline.py /path/to/scan_folder_or_file
# Options:
#   --no-filter     skip DICOM series filtering
#   --visualize     produce per-slice PNG overlays in output/
#   --device cpu    force CPU (default: cuda if available, else cpu)
#   --batch-size 8  inference batch size (default 8)
```

### 14.2 Re-training (hospital, ~10-15 days)

```bash
cd ct-brain-pipeline/rsna_training/
conda activate rsna-seutao

# First time only:
bash download_all.sh

# Smoke test FIRST (10-15 min) — verify everything logs correctly:
bash run_full_training.sh --smoke

# Full run:
nohup bash run_full_training.sh > full_run.log 2>&1 &

# Or run individual stages manually:
cd external/SeuTao_repo/2DNet/src
python3 train.py -backbone DenseNet121_change_avg -save_path DenseNet121_change_avg_256
python3 train.py -backbone DenseNet169_change_avg -save_path DenseNet169_change_avg_256
python3 train.py -backbone se_resnext101_32x4d -save_path se_resnext101_32x4d_256
python3 predict.py -backbone DenseNet121_change_avg -img_size 256 -tbs 4 -vbs 4 -spth DenseNet121_change_avg_256
python3 predict.py -backbone DenseNet169_change_avg -img_size 256 -tbs 4 -vbs 4 -spth DenseNet169_change_avg_256
python3 predict.py -backbone se_resnext101_32x4d -img_size 256 -tbs 4 -vbs 4 -spth se_resnext101_32x4d_256
cd ../../..
python3 scripts/build_sequence_inputs.py --repo-root external/SeuTao_repo
cd external/SeuTao_repo/SequenceModel
python3 main.py
```

### 14.3 Ischemic re-training (separate)

```bash
cd Bachelor_ischemic_test/
python3 train_ischemic_robust.py
# → results_ischemic_robust_3slice/best_model.pth
```

### 14.4 External evaluation

```bash
cd Bachelor_RSNA_test/
python3 run_ctich_full_ensemble.py
python3 run_brain_stroke_full.py
```

### 14.5 Post-training analysis

```bash
cd ct-brain-pipeline/rsna_training/external/SeuTao_repo/
python3 analyze_training.py \
  --root 2DNet/src/data_test \
  --out ../../../training_analysis/
# Produces: learning_curves/*.png, roc_curves/*.png, latex/auc_summary.tex
```

### 14.6 Compress training artifacts for transfer off the hospital workstation

```bash
cd rsna_training/
tar -czf training_metrics_$(date +%Y%m%d).tar.gz \
  external/SeuTao_repo/2DNet/src/data_test/*/fold*/log.csv \
  external/SeuTao_repo/2DNet/src/data_test/*/fold*/*.json \
  external/SeuTao_repo/2DNet/src/data_test/*/fold*/*.npz \
  external/SeuTao_repo/FinalSubmission/version3_debug/fold*/log.csv \
  external/SeuTao_repo/FinalSubmission/version3_debug/fold*/*.json \
  ../../training_analysis/
```

---

**End of document.** For questions or issues, refer to the thesis report
([Bachelor_RSNA_test/Bachelor_Rapport-6.pdf](Bachelor_RSNA_test/Bachelor_Rapport-6.pdf))
or the original SeuTao repository at
https://github.com/SeuTao/RSNA2019_Intracranial-Hemorrhage-Detection.
