# CT Brain Pipeline — Hemorrhage & Ischemic Stroke Detection

Hospital-ready pipeline that analyses a full patient non-contrast CT head scan
for **intracranial hemorrhage** (6 subtypes) and **ischemic stroke** in parallel.

## Models

| Model | Architecture | Task | Training |
|-------|-------------|------|----------|
| Hemorrhage | DenseNet121 5-fold ensemble | 6-class (any, epidural, intraparenchymal, intraventricular, subarachnoid, subdural) | RSNA 2019 Intracranial Hemorrhage |
| Ischemic | DenseNet121 (transfer-learned) | Binary (ischemic stroke) | CPAISD + AISD, fine-tuned from hemorrhage model |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Place model weights
mkdir -p models/hemorrhage models/ischemic
# Copy the 5 hemorrhage folds into models/hemorrhage/
#   model_epoch_79_0.pth ... model_epoch_79_4.pth
# Copy the ischemic model into models/ischemic/
#   best_model.pth
```

## Usage

```bash
# Full patient scan (DICOM folder from hospital PACS)
python pipeline.py /path/to/patient_dicom_folder/

# Single DICOM file
python pipeline.py /path/to/scan.dcm

# With visualization + custom output dir
python pipeline.py /path/to/patient_dicom_folder/ --visualize --output-dir results/

# GPU inference
python pipeline.py /path/to/patient_dicom_folder/ --device cuda

# Skip DICOM series filtering (use all files as-is)
python pipeline.py /path/to/folder/ --no-filter

# Adjust batch size for memory-constrained systems
python pipeline.py /path/to/folder/ --batch-size 4
```

## Pipeline Overview

```
DICOM folder
  → DICOM series filtering (skip scouts, localizers, non-CT)
  → Sort by z-position
  → Extract patient metadata
  → Read all slices → Hounsfield Units
  → Parallel preprocessing:
      Hemorrhage: brain window (40/80), 512×512, 3ch neighbor-slice context
      Ischemic:   multi-window (brain/stroke/soft tissue), 256×256
  → Parallel batched inference (ThreadPoolExecutor)
  → Per-slice thresholding (Youden-optimal for hemorrhage, 0.5 for ischemic)
  → Patient-level aggregation
  → Output: JSON results + console report + per-slice visualizations
```

## Output

### Console Report
```
======================================================================
  COMBINED PIPELINE — PATIENT REPORT
======================================================================

  Patient ID:      12345
  Age:             072Y
  Sex:             M
  Study Date:      2024-03-15

  Slices Analyzed: 42

──────────────────────────────────────────────────────────────────────
  PATIENT-LEVEL DIAGNOSIS
──────────────────────────────────────────────────────────────────────

  ⚠ HEMORRHAGE DETECTED  (3/42 slices)
    → subarachnoid: max p=0.573 (2 slice(s))

  ✓ No ischemic stroke detected  (max p=0.296)
```

### JSON (`results.json`)
Contains patient metadata, patient-level diagnosis, and per-slice probabilities.

### Visualizations
Per-slice PNG with CT scan (brain window), hemorrhage bar chart with per-class thresholds, and ischemic gauge.

## Hemorrhage Thresholds (Youden-Optimal)

| Subtype | Threshold |
|---------|-----------|
| any | 0.3715 |
| epidural | 0.0247 |
| intraparenchymal | 0.1738 |
| intraventricular | 0.1018 |
| subarachnoid | 0.1967 |
| subdural | 0.2191 |

## Architecture Diagram

See [pipeline_diagram.md](pipeline_diagram.md) for a Mermaid flowchart (viewable in Obsidian/GitHub).
