#!/usr/bin/env python3
"""
Combined CT Brain Scan Pipeline: Hemorrhage + Ischemic Stroke Detection
========================================================================
Hospital-ready pipeline for analysing a full patient CT head scan.
Takes DICOM files/folders OR NIfTI (.nii / .nii.gz) volumes as input,
runs two models in parallel:
  1. Hemorrhage model  — DenseNet121 5-fold ensemble (RSNA-trained)
  2. Ischemic model    — DenseNet121 transfer-learned binary classifier

Usage:
  # Full patient folder (auto-filters to axial brain CT series)
  python pipeline.py /path/to/patient_dicom_folder/

  # Single DICOM file
  python pipeline.py /path/to/scan.dcm

  # NIfTI volume (3D CT in HU)
  python pipeline.py /path/to/scan.nii.gz

  # Skip series filtering (use all files as-is)
  python pipeline.py /path/to/folder/ --no-filter

Output:
  - Console summary with patient-level diagnosis
  - Per-slice JSON results saved to output/
  - Optional per-slice visualization with --visualize flag
  - Patient-level aggregated report
"""

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pydicom
import nibabel as nib
import torch
import torch.nn as nn
import torchvision
import albumentations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
HEMORRHAGE_MODEL_DIR = BASE_DIR / "models" / "hemorrhage"
ISCHEMIC_MODEL_PATH = BASE_DIR / "models" / "ischemic" / "best_model.pth"
OUTPUT_DIR = BASE_DIR / "output"

# Hemorrhage model labels & Youden-optimal thresholds
HEMORRHAGE_LABELS = ["any", "epidural", "intraparenchymal",
                     "intraventricular", "subarachnoid", "subdural"]
HEMORRHAGE_THRESHOLDS = {
    "any": 0.3715,
    "epidural": 0.0247,
    "intraparenchymal": 0.1738,
    "intraventricular": 0.1018,
    "subarachnoid": 0.1967,
    "subdural": 0.2191,
}

# ── DICOM → HU conversion ─────────────────────────────────────────────────

def dicom_to_hu(dcm_path: str) -> np.ndarray:
    """Read a DICOM file and return the image in Hounsfield Units (float32)."""
    ds = pydicom.dcmread(dcm_path, force=True)
    if not hasattr(ds, "pixel_array"):
        raise ValueError(f"No pixel_array in {dcm_path}")

    intercept = float(getattr(ds, "RescaleIntercept", 0))
    slope = float(getattr(ds, "RescaleSlope", 1))
    img = ds.pixel_array.astype(np.float32) * slope + intercept
    return img


# ── NIfTI → HU conversion ─────────────────────────────────────────────────

NIFTI_EXTS = (".nii", ".nii.gz")


def is_nifti_path(path: Path) -> bool:
    """Check whether a path points to a NIfTI file."""
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def nifti_to_hu_slices(nii_path: Path) -> tuple[list[np.ndarray], dict]:
    """
    Read a NIfTI volume and return:
      - a list of 2D axial slices in Hounsfield Units (ordered inferior→superior)
      - a metadata dict (voxel spacing, shape, affine-derived info)

    Assumes the volume is already calibrated in HU (standard for NIfTI CT).
    Reorients to canonical RAS so the last axis is axial (superior +Z).
    """
    img = nib.load(str(nii_path))
    # Reorient to canonical RAS — last axis becomes superior-inferior (axial slices).
    img = nib.as_closest_canonical(img)
    data = img.get_fdata(dtype=np.float32)
    zooms = img.header.get_zooms()

    if data.ndim == 4:
        # Take first volume if 4D (e.g. time series / perfusion)
        data = data[..., 0]
        zooms = zooms[:3]

    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI volume, got shape {data.shape} in {nii_path}")

    # data shape after canonical reorient: (X, Y, Z) with Z = axial
    n_slices = data.shape[2]
    slices = [data[:, :, i].T.astype(np.float32) for i in range(n_slices)]
    # .T so rows = Y (anterior→posterior visually flipped to standard radiologic view)

    z_thickness = float(zooms[2]) if len(zooms) >= 3 else None
    px_spacing = (float(zooms[0]), float(zooms[1])) if len(zooms) >= 2 else None

    meta = {
        "PatientID": nii_path.stem.replace(".nii", ""),
        "PatientName": "Unknown",
        "PatientAge": "Unknown",
        "PatientSex": "Unknown",
        "StudyDate": "Unknown",
        "StudyDescription": "NIfTI volume",
        "SeriesDescription": nii_path.name,
        "InstitutionName": "Unknown",
        "Modality": "CT",
        "SliceThickness": f"{z_thickness:.3f}" if z_thickness is not None else "Unknown",
        "SliceThicknessFloat": z_thickness,
        "Manufacturer": "Unknown",
        "PixelSpacing": f"{px_spacing[0]:.3f} x {px_spacing[1]:.3f}" if px_spacing is not None else "Unknown",
        "NumSlices": n_slices,
        "SourceFile": str(nii_path),
    }
    return slices, meta


# ── Reslicing (thin→thick averaging) ─────────────────────────────────────────────

def reslice_to_thickness(
    slices: list[np.ndarray],
    source_thickness: float,
    target_thickness: float,
) -> tuple[list[np.ndarray], float, int]:
    """
    Average groups of consecutive thin slices into thicker effective slices.

    This mimics how clinical scanners reconstruct thick slices from thin
    acquisitions, and brings the input closer to the slice thickness our
    models were trained on (RSNA + AISD: typically 3–10 mm).

    Args:
        slices:            list of 2D HU arrays (ordered inferior→superior)
        source_thickness:  acquisition slice thickness (mm)
        target_thickness:  desired effective thickness (mm)

    Returns:
        (resliced_slices, achieved_thickness_mm, group_size)
        If no reslicing is needed (group_size <= 1), returns the input unchanged.
    """
    if source_thickness is None or source_thickness <= 0 or target_thickness <= 0:
        return slices, source_thickness or 0.0, 1

    group = int(round(target_thickness / source_thickness))
    if group <= 1:
        return slices, source_thickness, 1

    n = len(slices)
    if n < group:
        return slices, source_thickness, 1

    resliced: list[np.ndarray] = []
    for start in range(0, n - group + 1, group):
        chunk = slices[start:start + group]
        # Element-wise mean across slices in the group
        avg = np.mean(np.stack(chunk, axis=0), axis=0).astype(np.float32)
        resliced.append(avg)

    achieved = source_thickness * group
    return resliced, achieved, group


# ── Windowing helpers ──────────────────────────────────────────────────────

def apply_window(image_hu: np.ndarray, center: float, width: float) -> np.ndarray:
    """Apply HU window and return uint8 [0-255]."""
    low = center - width / 2
    high = center + width / 2
    img = np.clip(image_hu, low, high)
    return ((img - low) / (high - low) * 255).astype(np.uint8)


# ── Hemorrhage model preprocessing ────────────────────────────────────────

def prepare_hemorrhage_input(image_hu: np.ndarray) -> np.ndarray:
    """
    Brain-window the HU image and create a 3-channel input for the
    hemorrhage model (prev/curr/next context — for single slice we
    replicate the current slice into all 3 channels).
    Returns (512, 512, 3) uint8.
    """
    brain = apply_window(image_hu, center=40, width=80)
    brain = cv2.resize(brain, (512, 512))
    return np.stack([brain, brain, brain], axis=-1)


def prepare_hemorrhage_input_series(images_hu: list[np.ndarray], idx: int) -> np.ndarray:
    """
    Create 3-channel hemorrhage input with neighbor-slice context.
    Returns (512, 512, 3) uint8.
    """
    prev_idx = max(0, idx - 1)
    next_idx = min(len(images_hu) - 1, idx + 1)

    ch_prev = apply_window(images_hu[prev_idx], center=40, width=80)
    ch_curr = apply_window(images_hu[idx], center=40, width=80)
    ch_next = apply_window(images_hu[next_idx], center=40, width=80)

    ch_prev = cv2.resize(ch_prev, (512, 512))
    ch_curr = cv2.resize(ch_curr, (512, 512))
    ch_next = cv2.resize(ch_next, (512, 512))

    return np.stack([ch_prev, ch_curr, ch_next], axis=-1)


# ── Ischemic model preprocessing ──────────────────────────────────────────

def prepare_ischemic_input(image_hu: np.ndarray) -> np.ndarray:
    """
    Multi-window HU preprocessing for the ischemic model.
    Returns (256, 256, 3) uint8 with brain/stroke/soft-tissue channels.
    """
    ch_brain = apply_window(image_hu, center=40, width=80)
    ch_stroke = apply_window(image_hu, center=32, width=8)
    ch_soft = apply_window(image_hu, center=40, width=120)
    img_3ch = np.stack([ch_brain, ch_stroke, ch_soft], axis=-1)
    return cv2.resize(img_3ch, (256, 256))


# ── Model definitions ─────────────────────────────────────────────────────

class DenseNet121_Hemorrhage(nn.Module):
    """RSNA hemorrhage detection model (6-class)."""
    def __init__(self):
        super().__init__()
        self.densenet121 = torchvision.models.densenet121(weights=None).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1024, 6)

    def forward(self, x):
        x = self.densenet121(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(-1, 1024)
        x = self.mlp(x)
        return x


class DenseNet121_Ischemic(nn.Module):
    """Ischemic stroke classifier (binary), transfer-learned from hemorrhage model."""
    def __init__(self, densenet_features):
        super().__init__()
        self.features = densenet_features
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
        x = self.fc(x)
        return x.squeeze(1)


# ── Normalization (shared ImageNet-style) ──────────────────────────────────

NORMALIZE = albumentations.Compose([
    albumentations.Normalize(
        mean=(0.456, 0.456, 0.456),
        std=(0.224, 0.224, 0.224),
        max_pixel_value=255.0,
    )
])


def to_tensor(img_uint8: np.ndarray) -> torch.Tensor:
    """Normalize and convert HWC uint8 image to NCHW float tensor."""
    augmented = NORMALIZE(image=img_uint8)["image"]
    return torch.from_numpy(augmented.transpose(2, 0, 1)).unsqueeze(0).float()


# ── Model loading ──────────────────────────────────────────────────────────

def load_hemorrhage_models(model_dir: Path, device: torch.device) -> list:
    """Load all 5 folds of the DenseNet121 hemorrhage ensemble."""
    models = []
    for fold in range(5):
        ckpt_path = model_dir / f"model_epoch_79_{fold}.pth"
        if not ckpt_path.exists():
            print(f"  WARNING: {ckpt_path} not found, skipping fold {fold}")
            continue
        model = DenseNet121_Hemorrhage()
        model = nn.DataParallel(model)
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        models.append(model)
    print(f"  Loaded {len(models)} hemorrhage model folds")
    return models


def load_ischemic_model(model_path: Path, device: torch.device) -> nn.Module:
    """Load the ischemic stroke classifier."""
    features = torchvision.models.densenet121(weights=None).features
    model = DenseNet121_Ischemic(features)
    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    # Checkpoint may be a raw state_dict or wrapped in {'state_dict': ...}
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"  Loaded ischemic model from {model_path.name}")
    return model


# ── Inference functions ────────────────────────────────────────────────────

@torch.no_grad()
def predict_hemorrhage(models: list, img_3ch: np.ndarray, device: torch.device) -> dict:
    """Run 5-fold ensemble hemorrhage prediction. Returns dict with probs and labels."""
    tensor = to_tensor(img_3ch).to(device)
    preds = []
    for model in models:
        logits = model(tensor)
        probs = logits.sigmoid().squeeze(0).cpu().numpy()
        preds.append(probs)
    avg_probs = np.mean(preds, axis=0)

    result = {}
    for i, label in enumerate(HEMORRHAGE_LABELS):
        p = float(avg_probs[i])
        thresh = HEMORRHAGE_THRESHOLDS[label]
        result[label] = {
            "probability": round(p, 4),
            "threshold": thresh,
            "positive": p >= thresh,
        }
    return result


@torch.no_grad()
def predict_hemorrhage_batch(
    models: list, imgs_3ch: list[np.ndarray], device: torch.device, batch_size: int = 8
) -> list[dict]:
    """Run batched 5-fold ensemble hemorrhage prediction on multiple slices."""
    all_results = []
    for start in range(0, len(imgs_3ch), batch_size):
        batch_imgs = imgs_3ch[start:start + batch_size]
        tensors = [to_tensor(img).squeeze(0) for img in batch_imgs]
        batch_tensor = torch.stack(tensors).to(device)

        preds = []
        for model in models:
            logits = model(batch_tensor)
            probs = logits.sigmoid().cpu().numpy()  # (B, 6)
            preds.append(probs)
        avg_probs = np.mean(preds, axis=0)  # (B, 6)

        for b in range(avg_probs.shape[0]):
            result = {}
            for i, label in enumerate(HEMORRHAGE_LABELS):
                p = float(avg_probs[b, i])
                thresh = HEMORRHAGE_THRESHOLDS[label]
                result[label] = {
                    "probability": round(p, 4),
                    "threshold": thresh,
                    "positive": p >= thresh,
                }
            all_results.append(result)
    return all_results


@torch.no_grad()
def predict_ischemic(model: nn.Module, img_3ch: np.ndarray, device: torch.device) -> dict:
    """Run ischemic stroke prediction. Returns dict with prob and label."""
    tensor = to_tensor(img_3ch).to(device)
    logit = model(tensor)
    prob = torch.sigmoid(logit).cpu().item()
    return {
        "ischemic_stroke": {
            "probability": round(prob, 4),
            "threshold": 0.5,
            "positive": prob >= 0.5,
        }
    }


@torch.no_grad()
def predict_ischemic_batch(
    model: nn.Module, imgs_3ch: list[np.ndarray], device: torch.device, batch_size: int = 16
) -> list[dict]:
    """Run batched ischemic prediction on multiple slices."""
    all_results = []
    for start in range(0, len(imgs_3ch), batch_size):
        batch_imgs = imgs_3ch[start:start + batch_size]
        tensors = [to_tensor(img).squeeze(0) for img in batch_imgs]
        batch_tensor = torch.stack(tensors).to(device)

        logits = model(batch_tensor)
        probs = torch.sigmoid(logits).cpu().numpy()

        for b in range(len(batch_imgs)):
            prob = float(probs[b])
            all_results.append({
                "ischemic_stroke": {
                    "probability": round(prob, 4),
                    "threshold": 0.5,
                    "positive": prob >= 0.5,
                }
            })
    return all_results


# ── Parallel pipeline ─────────────────────────────────────────────────────

def run_pipeline_single_slice(
    image_hu: np.ndarray,
    hemorrhage_models: list,
    ischemic_model: nn.Module,
    device: torch.device,
    images_hu_series: list[np.ndarray] | None = None,
    slice_idx: int | None = None,
) -> dict:
    """
    Run both models on a single slice in parallel threads.
    Returns combined results dict.
    """
    if images_hu_series is not None and slice_idx is not None:
        hem_input = prepare_hemorrhage_input_series(images_hu_series, slice_idx)
    else:
        hem_input = prepare_hemorrhage_input(image_hu)
    isch_input = prepare_ischemic_input(image_hu)

    results = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_hem = executor.submit(
            predict_hemorrhage, hemorrhage_models, hem_input, device
        )
        future_isch = executor.submit(
            predict_ischemic, ischemic_model, isch_input, device
        )

        results["hemorrhage"] = future_hem.result()
        results["ischemic"] = future_isch.result()

    return results


def run_pipeline_batched(
    images_hu: list[np.ndarray],
    hemorrhage_models: list,
    ischemic_model: nn.Module,
    device: torch.device,
    batch_size: int = 8,
) -> list[dict]:
    """
    Run both models on all slices using batched inference + parallel threads.
    Much faster than per-slice for a full patient scan (30-200+ slices).
    """
    n = len(images_hu)
    use_series = n > 1

    # Prepare all inputs
    hem_inputs = []
    isch_inputs = []
    for i in range(n):
        if use_series:
            hem_inputs.append(prepare_hemorrhage_input_series(images_hu, i))
        else:
            hem_inputs.append(prepare_hemorrhage_input(images_hu[i]))
        isch_inputs.append(prepare_ischemic_input(images_hu[i]))

    # Run batched inference in parallel threads
    with ThreadPoolExecutor(max_workers=2) as executor:
        future_hem = executor.submit(
            predict_hemorrhage_batch, hemorrhage_models, hem_inputs, device, batch_size
        )
        future_isch = executor.submit(
            predict_ischemic_batch, ischemic_model, isch_inputs, device, batch_size
        )
        hem_results = future_hem.result()
        isch_results = future_isch.result()

    # Combine per-slice
    combined = []
    for i in range(n):
        combined.append({
            "hemorrhage": hem_results[i],
            "ischemic": isch_results[i],
        })
    return combined


# ── Visualization ──────────────────────────────────────────────────────────

def visualize_results(image_hu: np.ndarray, results: dict, title: str, out_path: Path):
    """Create a combined visualization showing CT scan + both model predictions."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    BAR_COLORS = {
        "any":                "#e74c3c",
        "epidural":           "#e67e22",
        "intraparenchymal":   "#2980b9",
        "intraventricular":   "#8e44ad",
        "subarachnoid":       "#27ae60",
        "subdural":           "#f39c12",
    }

    # Panel 1: CT scan (brain window)
    brain_img = apply_window(image_hu, center=40, width=80)
    axes[0].imshow(brain_img, cmap="gray")
    axes[0].set_title(f"Brain Window CT\n{title}", fontsize=11)
    axes[0].axis("off")

    # Panel 2: Hemorrhage predictions
    hem = results["hemorrhage"]
    labels = list(hem.keys())
    probs = [hem[l]["probability"] for l in labels]
    bar_colors = [BAR_COLORS[l] for l in labels]

    y_pos = np.arange(len(labels))
    axes[1].barh(y_pos, probs, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(labels, fontsize=10)
    axes[1].set_xlim(0, 1.15)
    axes[1].set_xlabel("Probability")
    axes[1].set_title("Hemorrhage Detection", fontsize=11, fontweight="bold")

    for i, (l, p) in enumerate(zip(labels, probs)):
        thresh = hem[l]["threshold"]
        color = BAR_COLORS[l]
        axes[1].plot([thresh, thresh], [i - 0.4, i + 0.4],
                     color=color, linestyle="--", linewidth=1.8, alpha=0.7)
        marker = "  \u2713" if hem[l]["positive"] else ""
        axes[1].text(p + 0.02, i, f"{p:.3f}{marker}", va="center", fontsize=9,
                     fontweight="bold" if hem[l]["positive"] else "normal",
                     color=color)

    any_hemorrhage = hem["any"]["positive"]
    status_color = "#e74c3c" if any_hemorrhage else "#27ae60"
    status_text = "HEMORRHAGE DETECTED" if any_hemorrhage else "No hemorrhage"
    axes[1].text(0.5, -0.12, status_text, transform=axes[1].transAxes,
                 ha="center", fontsize=12, fontweight="bold", color=status_color)

    # Panel 3: Ischemic prediction (gauge)
    isch = results["ischemic"]["ischemic_stroke"]
    isch_prob = isch["probability"]
    isch_pos = isch["positive"]

    theta = np.linspace(0, np.pi, 100)
    axes[2].plot(np.cos(theta), np.sin(theta), color="#bdc3c7", linewidth=8)

    n_fill = int(isch_prob * 100)
    fill_color = "#e74c3c" if isch_pos else "#27ae60"
    if n_fill > 0:
        theta_fill = np.linspace(0, np.pi * isch_prob, n_fill)
        axes[2].plot(np.cos(theta_fill), np.sin(theta_fill),
                     color=fill_color, linewidth=10)

    needle_angle = np.pi * (1 - isch_prob)
    axes[2].annotate("", xy=(0.7 * np.cos(needle_angle), 0.7 * np.sin(needle_angle)),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2))

    axes[2].text(0, -0.15, f"{isch_prob:.1%}", ha="center", fontsize=24,
                 fontweight="bold", color=fill_color)

    isch_status = "ISCHEMIC STROKE DETECTED" if isch_pos else "No ischemic stroke"
    axes[2].text(0, -0.35, isch_status, ha="center", fontsize=12,
                 fontweight="bold", color=fill_color)

    axes[2].set_xlim(-1.2, 1.2)
    axes[2].set_ylim(-0.5, 1.2)
    axes[2].set_aspect("equal")
    axes[2].set_title("Ischemic Stroke Detection", fontsize=11, fontweight="bold")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


# ── DICOM file/folder collection ──────────────────────────────────────────

# Substrings (case-insensitive) in filenames that should be skipped automatically.
#   - ROI / mask: segmentation overlays, not raw scans
#   - TOM:        "tom" (Danish for "empty") series — known to be low-quality
DEFAULT_SKIP_PATTERNS = ("_roi", "tom")


def _should_skip_filename(name: str, patterns: tuple[str, ...]) -> bool:
    lower = name.lower()
    return any(pat in lower for pat in patterns)


def collect_input_paths(
    input_paths: list[str],
    skip_patterns: tuple[str, ...] = DEFAULT_SKIP_PATTERNS,
) -> list[Path]:
    """Collect all DICOM and NIfTI files from the given paths.

    Files whose name contains any of `skip_patterns` (case-insensitive) are
    excluded — by default this drops ROI masks and "TOM" series.
    """
    files = []
    skipped = []
    for p in input_paths:
        path = Path(p)
        if path.is_file():
            if _should_skip_filename(path.name, skip_patterns):
                skipped.append(path)
                continue
            files.append(path)
        elif path.is_dir():
            # Recursively find DICOM and NIfTI files
            for f in sorted(path.rglob("*")):
                if not f.is_file():
                    continue
                if _should_skip_filename(f.name, skip_patterns):
                    skipped.append(f)
                    continue
                if is_nifti_path(f):
                    files.append(f)
                elif (
                    f.suffix.lower() in (".dcm", ".dicom")
                    or f.suffix == ""  # DICOM files often have no extension
                ):
                    files.append(f)
        else:
            print(f"  WARNING: {path} does not exist, skipping")
    if skipped:
        print(f"  Skipped {len(skipped)} file(s) matching skip patterns "
              f"({', '.join(skip_patterns)}):")
        for s in skipped[:5]:
            print(f"    - {s.name}")
        if len(skipped) > 5:
            print(f"    ... and {len(skipped) - 5} more")
    return files


def split_inputs_by_type(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Split a list of input paths into (nifti_files, dicom_files)."""
    nifti_files = [p for p in paths if is_nifti_path(p)]
    dicom_files = [p for p in paths if not is_nifti_path(p)]
    return nifti_files, dicom_files


def sort_dicom_by_position(dcm_paths: list[Path]) -> list[Path]:
    """Sort DICOM files by ImagePositionPatient[2] (z-position) if available."""
    positions = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            z = float(ds.ImagePositionPatient[2])
            positions.append((z, p))
        except Exception:
            positions.append((0, p))

    positions.sort(key=lambda x: x[0])
    return [p for _, p in positions]


# ── DICOM series filtering (hospital data) ─────────────────────────────────

def _is_localizer(ds) -> bool:
    """Check if DICOM is a scout/localizer image."""
    image_type = getattr(ds, "ImageType", [])
    if isinstance(image_type, (list, pydicom.multival.MultiValue)):
        image_type_str = "\\".join(str(x).upper() for x in image_type)
    else:
        image_type_str = str(image_type).upper()
    return "LOCALIZER" in image_type_str or "SCOUT" in image_type_str


def _is_ct_modality(ds) -> bool:
    """Check if DICOM is CT modality."""
    modality = str(getattr(ds, "Modality", "")).upper()
    return modality == "CT" or modality == ""


def filter_dicom_series(dcm_paths: list[Path], verbose: bool = True) -> list[Path]:
    """
    Filter DICOM files to keep only the relevant axial brain CT series.

    Filtering logic:
      1. Skip non-CT modalities (MR, US, etc.)
      2. Skip scout/localizer images
      3. If multiple SeriesInstanceUIDs exist, pick the series with the most
         slices (typically the primary axial series)
    """
    if not dcm_paths:
        return []

    # Read metadata for all files (header only, no pixels)
    meta = []
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            meta.append((p, ds))
        except Exception:
            continue

    if not meta:
        return dcm_paths

    # Step 1: Filter out non-CT
    ct_meta = [(p, ds) for p, ds in meta if _is_ct_modality(ds)]
    if verbose and len(ct_meta) < len(meta):
        dropped = len(meta) - len(ct_meta)
        print(f"  Filtered out {dropped} non-CT file(s)")
    if not ct_meta:
        if verbose:
            print("  WARNING: No CT modality files found, using all files")
        ct_meta = meta

    # Step 2: Filter out localizers/scouts
    axial_meta = [(p, ds) for p, ds in ct_meta if not _is_localizer(ds)]
    if verbose and len(axial_meta) < len(ct_meta):
        dropped = len(ct_meta) - len(axial_meta)
        print(f"  Filtered out {dropped} scout/localizer image(s)")
    if not axial_meta:
        if verbose:
            print("  WARNING: All files appear to be localizers, using all CT files")
        axial_meta = ct_meta

    # Step 3: Group by SeriesInstanceUID and pick the largest series
    series_groups: dict[str, list[Path]] = {}
    for p, ds in axial_meta:
        uid = str(getattr(ds, "SeriesInstanceUID", "unknown"))
        series_groups.setdefault(uid, []).append(p)

    if len(series_groups) > 1:
        largest_uid = max(series_groups, key=lambda uid: len(series_groups[uid]))
        if verbose:
            print(f"  Found {len(series_groups)} DICOM series:")
            for uid, paths in series_groups.items():
                desc = ""
                try:
                    ds = pydicom.dcmread(str(paths[0]), stop_before_pixels=True, force=True)
                    desc = getattr(ds, "SeriesDescription", "")
                except Exception:
                    pass
                marker = " <- selected" if uid == largest_uid else ""
                print(f"    Series '{desc}' ({len(paths)} slices){marker}")
        return series_groups[largest_uid]

    return [p for p, _ in axial_meta]


def extract_patient_metadata(dcm_paths: list[Path]) -> dict:
    """Extract patient-level metadata from the first readable DICOM file."""
    for p in dcm_paths:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            return {
                "PatientID": str(getattr(ds, "PatientID", "Unknown")),
                "PatientName": str(getattr(ds, "PatientName", "Unknown")),
                "PatientAge": str(getattr(ds, "PatientAge", "Unknown")),
                "PatientSex": str(getattr(ds, "PatientSex", "Unknown")),
                "StudyDate": str(getattr(ds, "StudyDate", "Unknown")),
                "StudyDescription": str(getattr(ds, "StudyDescription", "Unknown")),
                "SeriesDescription": str(getattr(ds, "SeriesDescription", "Unknown")),
                "InstitutionName": str(getattr(ds, "InstitutionName", "Unknown")),
                "Modality": str(getattr(ds, "Modality", "Unknown")),
                "SliceThickness": str(getattr(ds, "SliceThickness", "Unknown")),
                "Manufacturer": str(getattr(ds, "Manufacturer", "Unknown")),
            }
        except Exception:
            continue
    return {"PatientID": "Unknown"}


# ── Patient-level aggregation ──────────────────────────────────────────────

def aggregate_patient_results(all_results: list[dict]) -> dict:
    """
    Aggregate per-slice results into a patient-level diagnosis.

    Strategy:
      - Hemorrhage: positive if ANY slice is positive for "any" class.
        Reports max probability across slices for each subtype.
      - Ischemic: positive if ANY slice is positive.
        Reports max probability across slices.
    """
    n = len(all_results)
    if n == 0:
        return {}

    # Hemorrhage aggregation
    hem_max_probs = {label: 0.0 for label in HEMORRHAGE_LABELS}
    hem_pos_counts = {label: 0 for label in HEMORRHAGE_LABELS}
    hem_pos_slices = {label: [] for label in HEMORRHAGE_LABELS}

    for res in all_results:
        hem = res["results"]["hemorrhage"]
        for label in HEMORRHAGE_LABELS:
            p = hem[label]["probability"]
            if p > hem_max_probs[label]:
                hem_max_probs[label] = p
            if hem[label]["positive"]:
                hem_pos_counts[label] += 1
                hem_pos_slices[label].append(res["slice_index"])

    hemorrhage_diagnosis = {
        "patient_positive": hem_pos_counts["any"] > 0,
        "n_positive_slices": hem_pos_counts["any"],
        "n_total_slices": n,
        "subtypes": {},
    }
    for label in HEMORRHAGE_LABELS:
        hemorrhage_diagnosis["subtypes"][label] = {
            "max_probability": round(hem_max_probs[label], 4),
            "n_positive_slices": hem_pos_counts[label],
            "positive_slice_indices": hem_pos_slices[label],
            "patient_positive": hem_pos_counts[label] > 0,
        }

    # Ischemic aggregation
    isch_probs = []
    isch_pos_slices = []
    for res in all_results:
        isch = res["results"]["ischemic"]["ischemic_stroke"]
        isch_probs.append(isch["probability"])
        if isch["positive"]:
            isch_pos_slices.append(res["slice_index"])

    ischemic_diagnosis = {
        "patient_positive": len(isch_pos_slices) > 0,
        "max_probability": round(max(isch_probs), 4),
        "mean_probability": round(float(np.mean(isch_probs)), 4),
        "n_positive_slices": len(isch_pos_slices),
        "n_total_slices": n,
        "positive_slice_indices": isch_pos_slices,
    }

    return {
        "hemorrhage": hemorrhage_diagnosis,
        "ischemic": ischemic_diagnosis,
    }


# ── Print summary ──────────────────────────────────────────────────────────

def print_summary(all_results: list[dict], patient_meta: dict | None = None,
                  patient_agg: dict | None = None):
    """Print a formatted patient-level summary."""
    print("\n" + "=" * 70)
    print("  COMBINED PIPELINE — PATIENT REPORT")
    print("=" * 70)

    if patient_meta and patient_meta.get("PatientID", "Unknown") != "Unknown":
        print(f"\n  Patient ID:      {patient_meta.get('PatientID', 'Unknown')}")
        if patient_meta.get("PatientName", "Unknown") != "Unknown":
            print(f"  Patient Name:    {patient_meta['PatientName']}")
        if patient_meta.get("PatientAge", "Unknown") != "Unknown":
            print(f"  Age:             {patient_meta['PatientAge']}")
        if patient_meta.get("PatientSex", "Unknown") != "Unknown":
            print(f"  Sex:             {patient_meta['PatientSex']}")
        if patient_meta.get("StudyDate", "Unknown") != "Unknown":
            sd = patient_meta["StudyDate"]
            if len(sd) == 8:
                sd = f"{sd[:4]}-{sd[4:6]}-{sd[6:8]}"
            print(f"  Study Date:      {sd}")
        if patient_meta.get("InstitutionName", "Unknown") != "Unknown":
            print(f"  Institution:     {patient_meta['InstitutionName']}")
        if patient_meta.get("SeriesDescription", "Unknown") != "Unknown":
            print(f"  Series:          {patient_meta['SeriesDescription']}")
        if patient_meta.get("SliceThickness", "Unknown") != "Unknown":
            print(f"  Slice Thickness: {patient_meta['SliceThickness']} mm")

    n_total = len(all_results)
    print(f"\n  Slices Analyzed: {n_total}")

    if patient_agg:
        hem = patient_agg["hemorrhage"]
        isch = patient_agg["ischemic"]

        print(f"\n{'─' * 70}")
        print("  PATIENT-LEVEL DIAGNOSIS")
        print(f"{'─' * 70}")

        if hem["patient_positive"]:
            print(f"\n  ⚠ HEMORRHAGE DETECTED  ({hem['n_positive_slices']}/{n_total} slices)")
            for label in HEMORRHAGE_LABELS[1:]:
                sub = hem["subtypes"][label]
                if sub["patient_positive"]:
                    print(f"    → {label}: max p={sub['max_probability']:.3f} "
                          f"({sub['n_positive_slices']} slice(s))")
        else:
            any_max = hem["subtypes"]["any"]["max_probability"]
            print(f"\n  ✓ No hemorrhage detected  (max p={any_max:.3f})")

        if isch["patient_positive"]:
            print(f"\n  ⚠ ISCHEMIC STROKE DETECTED  "
                  f"({isch['n_positive_slices']}/{n_total} slices, "
                  f"max p={isch['max_probability']:.3f})")
        else:
            print(f"\n  ✓ No ischemic stroke detected  (max p={isch['max_probability']:.3f})")

    if n_total <= 10:
        print(f"\n{'─' * 70}")
        print("  PER-SLICE DETAILS")
        print(f"{'─' * 70}")
        for res in all_results:
            hem = res["results"]["hemorrhage"]
            isch = res["results"]["ischemic"]["ischemic_stroke"]
            any_hem = hem["any"]
            h_status = "⚠ HEM" if any_hem["positive"] else "  neg"
            i_status = "⚠ ISC" if isch["positive"] else "  neg"
            print(f"  Slice {res['slice_index']:3d}: "
                  f"{h_status} (p={any_hem['probability']:.3f})  "
                  f"{i_status} (p={isch['probability']:.3f})  "
                  f"[{res['file']}]")
    else:
        pos_slices = [r for r in all_results
                      if r["results"]["hemorrhage"]["any"]["positive"]
                      or r["results"]["ischemic"]["ischemic_stroke"]["positive"]]
        if pos_slices:
            print(f"\n{'─' * 70}")
            print(f"  POSITIVE SLICES ({len(pos_slices)}/{n_total})")
            print(f"{'─' * 70}")
            for res in pos_slices:
                hem = res["results"]["hemorrhage"]
                isch = res["results"]["ischemic"]["ischemic_stroke"]
                any_hem = hem["any"]
                flags = []
                if any_hem["positive"]:
                    subtypes = [l for l in HEMORRHAGE_LABELS[1:] if hem[l]["positive"]]
                    flags.append(f"HEM p={any_hem['probability']:.3f} [{','.join(subtypes)}]")
                if isch["positive"]:
                    flags.append(f"ISC p={isch['probability']:.3f}")
                print(f"  Slice {res['slice_index']:3d}: {' | '.join(flags)}")

    print(f"\n{'=' * 70}\n")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Combined CT Brain Pipeline: Hemorrhage + Ischemic Stroke Detection"
    )
    parser.add_argument("input", nargs="+",
                        help="DICOM file(s)/folder(s) or NIfTI (.nii/.nii.gz) volume to analyze")
    parser.add_argument("--visualize", action="store_true",
                        help="Save visualization images")
    parser.add_argument("--device", default="auto",
                        help="Device: 'cpu', 'cuda', or 'auto' (default)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: output/)")
    parser.add_argument("--hemorrhage-model-dir", type=str, default=None,
                        help="Override hemorrhage model directory")
    parser.add_argument("--ischemic-model-path", type=str, default=None,
                        help="Override ischemic model path")
    parser.add_argument("--no-filter", action="store_true",
                        help="Skip DICOM series filtering (use all files as-is)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for inference (default: 8)")
    parser.add_argument("--target-thickness", type=float, default=5.0,
                        help="Target effective slice thickness in mm. Thin source "
                             "slices are averaged in groups to approximate this. "
                             "Set to 0 to disable. Default: 5.0 mm.")
    parser.add_argument("--source-thickness", type=float, default=None,
                        help="Override source slice thickness (mm) for reslicing. "
                             "By default this is read from the NIfTI/DICOM header.")
    parser.add_argument("--no-reslice", action="store_true",
                        help="Disable thin-to-thick slice averaging.")
    parser.add_argument("--keep-tom", action="store_true",
                        help="Do not skip files with 'TOM' in the name.")
    parser.add_argument("--keep-roi", action="store_true",
                        help="Do not skip ROI/mask files.")
    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Resolve paths
    out_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    hem_dir = Path(args.hemorrhage_model_dir) if args.hemorrhage_model_dir else HEMORRHAGE_MODEL_DIR
    isch_path = Path(args.ischemic_model_path) if args.ischemic_model_path else ISCHEMIC_MODEL_PATH

    # Collect input files (DICOM and/or NIfTI)
    skip_patterns = tuple(
        pat for pat, keep in (
            ("_roi", args.keep_roi),
            ("tom", args.keep_tom),
        ) if not keep
    )
    input_files = collect_input_paths(args.input, skip_patterns=skip_patterns)
    if not input_files:
        print("ERROR: No DICOM or NIfTI files found in the provided input(s).")
        sys.exit(1)

    nifti_files, dicom_files = split_inputs_by_type(input_files)
    print(f"\nFound {len(dicom_files)} DICOM file(s) and {len(nifti_files)} NIfTI file(s)")

    if nifti_files and dicom_files:
        print("  WARNING: Mixed DICOM + NIfTI input. Processing NIfTI volume(s) only.")
        dicom_files = []

    use_nifti = len(nifti_files) > 0

    if use_nifti and len(nifti_files) > 1:
        print(f"  WARNING: Multiple NIfTI files supplied; using only the first: {nifti_files[0].name}")
        nifti_files = nifti_files[:1]

    if not use_nifti:
        # ── DICOM series filtering ─────────────────────────────────────
        if not args.no_filter and len(dicom_files) > 1:
            print("\nFiltering DICOM series...")
            dicom_files = filter_dicom_series(dicom_files)
            print(f"  {len(dicom_files)} file(s) after filtering")

        # Sort by z-position if multiple
        if len(dicom_files) > 1:
            dicom_files = sort_dicom_by_position(dicom_files)
            print("  Sorted DICOM files by slice position")

        # ── Extract patient metadata ───────────────────────────────────
        patient_meta = extract_patient_metadata(dicom_files)
    else:
        # NIfTI: metadata comes from the volume header
        patient_meta = {}

    if patient_meta.get("PatientID", "Unknown") != "Unknown":
        print(f"\n  Patient: {patient_meta['PatientID']}")

    # ── Load models in parallel ────────────────────────────────────────
    print("\nLoading models...")
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_hem = executor.submit(load_hemorrhage_models, hem_dir, device)
        future_isch = executor.submit(load_ischemic_model, isch_path, device)
        hemorrhage_models = future_hem.result()
        ischemic_model = future_isch.result()

    print(f"  Models loaded in {time.time() - t0:.1f}s")

    if not hemorrhage_models:
        print("ERROR: No hemorrhage model folds found in " + str(hem_dir))
        print("  Expected files: model_epoch_79_0.pth ... model_epoch_79_4.pth")
        sys.exit(1)

    # ── Read input volume(s) → list of HU slices ──────────────────────
    images_hu: list[np.ndarray] = []
    valid_paths: list[Path] = []

    if use_nifti:
        nii_path = nifti_files[0]
        print(f"\nReading NIfTI volume: {nii_path.name}")
        try:
            slices, nii_meta = nifti_to_hu_slices(nii_path)
        except Exception as e:
            print(f"ERROR: Could not read NIfTI file {nii_path}: {e}")
            sys.exit(1)
        images_hu = slices
        # Synthesize one virtual "path" per slice for downstream reporting
        valid_paths = [nii_path.with_name(f"{nii_path.name}#slice_{i:04d}") for i in range(len(slices))]
        patient_meta = nii_meta
        print(f"  Loaded {len(slices)} axial slice(s) "
              f"(thickness={nii_meta.get('SliceThickness', 'Unknown')} mm, "
              f"pixel_spacing={nii_meta.get('PixelSpacing', 'Unknown')} mm)")
    else:
        print("\nReading DICOM files...")
        for dcm_path in dicom_files:
            try:
                hu = dicom_to_hu(str(dcm_path))
                images_hu.append(hu)
                valid_paths.append(dcm_path)
            except Exception as e:
                print(f"  WARNING: Could not read {dcm_path.name}: {e}")

    if not images_hu:
        print("ERROR: No valid input slices could be read.")
        sys.exit(1)

    print(f"  Read {len(images_hu)} valid slice(s)")

    # Reslice thin slices into thicker effective slices to match training data
    if not args.no_reslice and args.target_thickness > 0:
        source_thk = args.source_thickness
        if source_thk is None:
            if use_nifti:
                source_thk = patient_meta.get("SliceThicknessFloat")
            else:
                try:
                    ds = pydicom.dcmread(str(valid_paths[0]),
                                         stop_before_pixels=True, force=True)
                    st = getattr(ds, "SliceThickness", None)
                    if st is not None:
                        source_thk = float(st)
                except Exception:
                    source_thk = None

        if source_thk is None or source_thk <= 0:
            print("  Reslicing skipped: source slice thickness unknown "
                  "(use --source-thickness to override)")
        else:
            new_slices, achieved, group = reslice_to_thickness(
                images_hu, source_thk, args.target_thickness,
            )
            if group > 1:
                print(f"  Reslicing: averaged groups of {group} slice(s) "
                      f"({source_thk:.2f}mm -> ~{achieved:.2f}mm effective, "
                      f"target {args.target_thickness:.2f}mm)")
                print(f"  Slice count: {len(images_hu)} -> {len(new_slices)}")
                images_hu = new_slices
                if use_nifti:
                    src = Path(patient_meta.get("SourceFile", str(valid_paths[0])))
                    valid_paths = [
                        src.with_name(f"{src.name}#resliced_{i:04d}")
                        for i in range(len(new_slices))
                    ]
                else:
                    base = valid_paths[0]
                    valid_paths = [
                        base.with_name(f"resliced_{i:04d}{base.suffix}")
                        for i in range(len(new_slices))
                    ]
                patient_meta = dict(patient_meta)
                patient_meta["SliceThickness"] = (
                    f"{achieved:.3f} (resliced from {source_thk:.3f})"
                )
                patient_meta["NumSlices"] = len(new_slices)
            else:
                print(f"  Reslicing not needed (source {source_thk:.2f}mm "
                      f">= target {args.target_thickness:.2f}mm)")

    # ── Run pipeline ───────────────────────────────────────────────────
    print("\nRunning inference...")
    t0 = time.time()
    all_results = []

    if len(images_hu) > 1:
        print(f"  Using batched inference (batch_size={args.batch_size})")
        combined_results = run_pipeline_batched(
            images_hu=images_hu,
            hemorrhage_models=hemorrhage_models,
            ischemic_model=ischemic_model,
            device=device,
            batch_size=args.batch_size,
        )
        for i, (hu, path, results) in enumerate(zip(images_hu, valid_paths, combined_results)):
            entry = {
                "file": str(path.name),
                "path": str(path),
                "slice_index": i,
                "results": results,
            }
            all_results.append(entry)
            if args.visualize:
                vis_path = out_dir / f"slice_{i:04d}_{path.stem}.png"
                visualize_results(hu, results, path.name, vis_path)
            if (i + 1) % 10 == 0 or (i + 1) == len(images_hu):
                print(f"  {i + 1}/{len(images_hu)} slices processed")
    else:
        results = run_pipeline_single_slice(
            image_hu=images_hu[0],
            hemorrhage_models=hemorrhage_models,
            ischemic_model=ischemic_model,
            device=device,
        )
        entry = {
            "file": str(valid_paths[0].name),
            "path": str(valid_paths[0]),
            "slice_index": 0,
            "results": results,
        }
        all_results.append(entry)
        if args.visualize:
            vis_path = out_dir / f"slice_0000_{valid_paths[0].stem}.png"
            visualize_results(images_hu[0], results, valid_paths[0].name, vis_path)
        print("  1/1 slices processed")

    elapsed = time.time() - t0
    print(f"\n  Inference completed in {elapsed:.1f}s "
          f"({elapsed / len(images_hu):.2f}s per slice)")

    # ── Aggregate patient-level results ────────────────────────────────
    patient_agg = aggregate_patient_results(all_results)

    # ── Print and save results ─────────────────────────────────────────
    print_summary(all_results, patient_meta=patient_meta, patient_agg=patient_agg)

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "pipeline": "Combined Hemorrhage + Ischemic Stroke Detection",
            "device": str(device),
            "n_slices": len(all_results),
            "patient_metadata": patient_meta,
            "patient_diagnosis": patient_agg,
            "hemorrhage_model": f"DenseNet121 5-fold ensemble",
            "ischemic_model": f"DenseNet121 transfer-learned",
            "hemorrhage_thresholds": HEMORRHAGE_THRESHOLDS,
            "ischemic_threshold": 0.5,
            "slices": all_results,
        }, f, indent=2)
    print(f"Results saved to {results_path}")

    if args.visualize:
        print(f"Visualizations saved to {out_dir}/")


if __name__ == "__main__":
    main()
