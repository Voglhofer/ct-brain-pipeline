#!/usr/bin/env python3
"""
Combined CT Brain Scan Pipeline: Hemorrhage + Ischemic Stroke Detection
========================================================================
Hospital-ready pipeline for analysing a full patient CT head scan.
Takes DICOM files/folders OR NIfTI (.nii / .nii.gz) volumes as input,
runs two models in parallel:
  1. Hemorrhage model  — 3-backbone × 5-fold ensemble (15 models total:
                          DenseNet121 + DenseNet169 + SE-ResNeXt101),
                          all RSNA-trained. Falls back to a flat
                          DenseNet121-only layout if the per-backbone
                          subdirectories are missing.
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

    Robustness: some publicly available NIfTI files (notably some nnUNet
    preprocessed datasets) ship with a broken affine where the in-plane vs.
    slice axes are mislabeled. We detect this by checking the Z-extent
    (head height should be roughly 80-350 mm) and, if implausible, fall
    back to using the axis with the largest voxel spacing as the slice
    axis (which is what clinical CT looks like in practice).
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
    zooms = tuple(float(z) for z in zooms[:3])
    extents = tuple(s * z for s, z in zip(data.shape, zooms))

    slice_axis = 2  # canonical RAS → Z is last
    z_thickness = zooms[slice_axis]
    in_plane_axes = [a for a in (0, 1, 2) if a != slice_axis]

    # Sanity check: a real head CT spans ~80-350 mm superior-inferior, has
    # ~10-400 slices, and in-plane voxels are typically ≤ 1.5 mm.  If the
    # canonical Z-axis violates these, the affine is almost certainly
    # mis-assembled (we have seen this on some nnUNet-preprocessed datasets
    # where shape=(96,512,512) but zooms=(0.5,0.5,3.0) — the slice axis is
    # axis 0 with thickness 3.0 mm, but the affine puts thickness on axis 2).
    z_extent = extents[slice_axis]
    in_plane_max = max(zooms[a] for a in in_plane_axes)
    affine_looks_wrong = (
        not (80.0 <= z_extent <= 350.0)
        or in_plane_max > 1.5
        or data.shape[slice_axis] > 600
    )

    if affine_looks_wrong:
        # Heuristic: slice axis has the fewest voxels; slice thickness is
        # the largest zoom value. Treat them independently because the
        # broken affine may have them on different axes.
        guessed_axis = int(np.argmin(data.shape))
        guessed_thk = float(max(zooms))
        guessed_extent = data.shape[guessed_axis] * guessed_thk
        if 60.0 <= guessed_extent <= 400.0:
            print(f"  WARNING: NIfTI affine looks broken "
                  f"(canonical: axis {slice_axis}, extent {z_extent:.0f} mm, "
                  f"in-plane max {in_plane_max:.2f} mm). "
                  f"Falling back to axis {guessed_axis} as slice axis with "
                  f"thickness {guessed_thk:.2f} mm "
                  f"(extent {guessed_extent:.0f} mm).")
            slice_axis = guessed_axis
            z_thickness = guessed_thk
            in_plane_axes = [a for a in (0, 1, 2) if a != slice_axis]

    n_slices = data.shape[slice_axis]

    # Extract axial slices. Each slice should be 2D in_plane × in_plane.
    # Using .T so rows = first in-plane axis (matches the rest of the
    # pipeline's convention for HU images).
    slices: list[np.ndarray] = [
        np.take(data, i, axis=slice_axis).T.astype(np.float32)
        for i in range(n_slices)
    ]

    # In-plane spacing: when affine was wrong, fall back to the smaller of
    # the remaining zooms (typical head CT has equal x/y, ~0.4-1 mm).
    if affine_looks_wrong:
        remaining = sorted(zooms[a] for a in in_plane_axes)
        px_spacing = (remaining[0], remaining[0])  # assume isotropic in-plane
    else:
        px_spacing = (zooms[in_plane_axes[0]], zooms[in_plane_axes[1]])

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
    """RSNA hemorrhage detection model (6-class) — DenseNet121 backbone."""
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


class DenseNet169_Hemorrhage(nn.Module):
    """RSNA hemorrhage detection model (6-class) — DenseNet169 backbone.

    Mirrors SeuTao 2DNet `DenseNet169_change_avg`.
    """
    def __init__(self):
        super().__init__()
        self.densenet169 = torchvision.models.densenet169(weights=None).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1664, 6)

    def forward(self, x):
        x = self.densenet169(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)
        return x


class SeResNext101_Hemorrhage(nn.Module):
    """RSNA hemorrhage detection model (6-class) — SE-ResNeXt101 backbone.

    Mirrors SeuTao 2DNet `SeResNext101_change_avg`. Requires the
    `pretrainedmodels` package (declared in requirements.txt). The
    ImageNet weights are *not* downloaded; the checkpoint overwrites
    everything anyway.
    """
    def __init__(self):
        super().__init__()
        import pretrainedmodels  # local import — optional dep at runtime
        self.model_ft = pretrainedmodels.__dict__["se_resnext101_32x4d"](
            num_classes=1000, pretrained=None
        )
        num_ftrs = self.model_ft.last_linear.in_features
        self.model_ft.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.model_ft.last_linear = nn.Sequential(nn.Linear(num_ftrs, 6, bias=True))

    def forward(self, x):
        return self.model_ft(x)


# Backbone registry: directory name (under models/hemorrhage/) -> class
HEMORRHAGE_BACKBONES = {
    "DenseNet121":   DenseNet121_Hemorrhage,
    "DenseNet169":   DenseNet169_Hemorrhage,
    "SE-ResNeXt101": SeResNext101_Hemorrhage,
}


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
    """Load the hemorrhage ensemble.

    If per-backbone subdirectories exist (`DenseNet121/`, `DenseNet169/`,
    `SE-ResNeXt101/`) we load all 5 folds from each one, giving a 15-model
    3-backbone ensemble that matches Bachelor_RSNA_test/run_ctich_full_ensemble.py.
    Otherwise we fall back to the legacy flat layout (5 DenseNet121 folds
    directly under model_dir) for backwards compatibility.
    """
    models: list = []
    subdir_layout = any((model_dir / name).is_dir() for name in HEMORRHAGE_BACKBONES)

    if subdir_layout:
        for name, cls in HEMORRHAGE_BACKBONES.items():
            arch_dir = model_dir / name
            if not arch_dir.is_dir():
                print(f"  WARNING: {arch_dir} not found, skipping backbone {name}")
                continue
            n_loaded = 0
            for fold in range(5):
                ckpt_path = arch_dir / f"model_epoch_79_{fold}.pth"
                if not ckpt_path.exists():
                    print(f"  WARNING: {ckpt_path} not found, skipping {name} fold {fold}")
                    continue
                model = cls()
                model = nn.DataParallel(model)
                ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
                model.load_state_dict(ckpt["state_dict"])
                model.to(device)
                model.eval()
                models.append(model)
                n_loaded += 1
            print(f"  Loaded {n_loaded} {name} folds")
        print(f"  Hemorrhage ensemble: {len(models)} models across {sum(1 for n in HEMORRHAGE_BACKBONES if (model_dir / n).is_dir())} backbones")
        return models

    # Legacy flat DenseNet121-only layout
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
    print(f"  Loaded {len(models)} hemorrhage model folds (legacy DenseNet121-only layout)")
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

def visualize_patient_summary(all_results: list[dict], patient_agg: dict,
                              patient_meta: dict | None, out_path: Path):
    """
    Patient-level overview: probability vs. slice index for hemorrhage (any)
    and ischemic, with thresholds and positive slices clearly marked.
    """
    n = len(all_results)
    if n == 0:
        return

    slice_idx = [r["slice_index"] for r in all_results]
    hem_probs = [r["results"]["hemorrhage"]["any"]["probability"] for r in all_results]
    isch_probs = [r["results"]["ischemic"]["ischemic_stroke"]["probability"]
                  for r in all_results]

    hem_thresh = HEMORRHAGE_THRESHOLDS["any"]
    isch_thresh = all_results[0]["results"]["ischemic"]["ischemic_stroke"].get(
        "threshold", 0.5)

    hem_pos_idx = patient_agg["hemorrhage"]["subtypes"]["any"]["positive_slice_indices"]
    isch_pos_idx = patient_agg["ischemic"]["positive_slice_indices"]

    hem_patient_prob = patient_agg["hemorrhage"]["subtypes"]["any"]["max_probability"]
    isch_patient_prob = patient_agg["ischemic"]["max_probability"]
    hem_pos = patient_agg["hemorrhage"]["patient_positive"]
    isch_pos = patient_agg["ischemic"]["patient_positive"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    pid = (patient_meta or {}).get("PatientID", "")
    title = f"Patient summary — {pid}" if pid and pid != "Unknown" else "Patient summary"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # --- Hemorrhage (any) ---
    ax = axes[0]
    ax.plot(slice_idx, hem_probs, color="#34495e", linewidth=1.4, alpha=0.8)
    ax.scatter(slice_idx, hem_probs, s=18, color="#34495e", alpha=0.6)
    pos_x = [s for s, p in zip(slice_idx, hem_probs) if s in set(hem_pos_idx)]
    pos_y = [p for s, p in zip(slice_idx, hem_probs) if s in set(hem_pos_idx)]
    if pos_x:
        ax.scatter(pos_x, pos_y, s=70, color="#e74c3c", edgecolor="white",
                   linewidth=1.2, zorder=5, label=f"positive ({len(pos_x)})")
    ax.axhline(hem_thresh, color="#e74c3c", linestyle="--", linewidth=1,
               alpha=0.7, label=f"threshold {hem_thresh:.2f}")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("p(hemorrhage)")
    flag = "POSITIVE" if hem_pos else "negative"
    ax.set_title(f"Hemorrhage (any)   max={hem_patient_prob:.3f}   →  {flag}",
                 fontsize=11, color="#e74c3c" if hem_pos else "#27ae60")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)

    # --- Ischemic ---
    ax = axes[1]
    ax.plot(slice_idx, isch_probs, color="#34495e", linewidth=1.4, alpha=0.8)
    ax.scatter(slice_idx, isch_probs, s=18, color="#34495e", alpha=0.6)
    pos_x = [s for s, p in zip(slice_idx, isch_probs) if s in set(isch_pos_idx)]
    pos_y = [p for s, p in zip(slice_idx, isch_probs) if s in set(isch_pos_idx)]
    if pos_x:
        ax.scatter(pos_x, pos_y, s=70, color="#e74c3c", edgecolor="white",
                   linewidth=1.2, zorder=5, label=f"positive ({len(pos_x)})")
    ax.axhline(isch_thresh, color="#e74c3c", linestyle="--", linewidth=1,
               alpha=0.7, label=f"threshold {isch_thresh:.2f}")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("p(ischemic)")
    ax.set_xlabel("slice index")
    flag = "POSITIVE" if isch_pos else "negative"
    ax.set_title(f"Ischemic stroke   max={isch_patient_prob:.3f}   →  {flag}",
                 fontsize=11, color="#e74c3c" if isch_pos else "#27ae60")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close()


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

    # Panel 3: Ischemic prediction (clean number readout)
    isch = results["ischemic"]["ischemic_stroke"]
    isch_prob = isch["probability"]
    isch_pos = isch["positive"]
    isch_thresh = isch.get("threshold", 0.5)
    fill_color = "#e74c3c" if isch_pos else "#27ae60"

    axes[2].text(0.5, 0.78, "Ischemic stroke probability",
                 ha="center", va="center", transform=axes[2].transAxes,
                 fontsize=12, color="#2c3e50")
    axes[2].text(0.5, 0.50, f"{isch_prob:.3f}",
                 ha="center", va="center", transform=axes[2].transAxes,
                 fontsize=48, fontweight="bold", color=fill_color)
    axes[2].text(0.5, 0.28, f"({isch_prob*100:.2f}%)",
                 ha="center", va="center", transform=axes[2].transAxes,
                 fontsize=14, color=fill_color)
    axes[2].text(0.5, 0.14, f"threshold {isch_thresh:.2f}",
                 ha="center", va="center", transform=axes[2].transAxes,
                 fontsize=10, color="#7f8c8d")
    isch_status = "ISCHEMIC STROKE DETECTED" if isch_pos else "No ischemic stroke"
    axes[2].text(0.5, 0.02, isch_status,
                 ha="center", va="center", transform=axes[2].transAxes,
                 fontsize=12, fontweight="bold", color=fill_color)

    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(0, 1)
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
                  patient_agg: dict | None = None,
                  show_slices: bool = False):
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
        hem_prob = hem["subtypes"]["any"]["max_probability"]
        hem_thresh = HEMORRHAGE_THRESHOLDS["any"]
        isch_prob = isch["max_probability"]
        isch_thresh = 0.5

        print(f"\n{'=' * 70}")
        print("  FINAL PATIENT PROBABILITY")
        print(f"{'=' * 70}")

        hem_flag = "POSITIVE" if hem["patient_positive"] else "negative"
        isch_flag = "POSITIVE" if isch["patient_positive"] else "negative"

        print(f"\n  HEMORRHAGE       : {hem_prob*100:6.2f}%   "
              f"(threshold {hem_thresh*100:.2f}%)   {hem_flag}")
        print(f"  ISCHEMIC STROKE  : {isch_prob*100:6.2f}%   "
              f"(threshold {isch_thresh*100:.2f}%)   {isch_flag}")

        if hem["patient_positive"]:
            print("\n  Hemorrhage subtype breakdown (max probability across slices):")
            for label in HEMORRHAGE_LABELS[1:]:
                sub = hem["subtypes"][label]
                marker = "*" if sub["patient_positive"] else " "
                print(f"    {marker} {label:18s} {sub['max_probability']*100:6.2f}%   "
                      f"(threshold {HEMORRHAGE_THRESHOLDS[label]*100:.2f}%)")

        # Always surface which slices fired (helps localise the finding)
        def _fmt_slices(idx_list, total_n):
            if not idx_list:
                return "  (none)"
            shown = idx_list if len(idx_list) <= 20 else idx_list[:20] + ["..."]
            return "  " + ", ".join(str(s) for s in shown) + f"   [{len(idx_list)}/{total_n}]"

        hem_any_idx = hem["subtypes"]["any"]["positive_slice_indices"]
        isch_idx    = isch["positive_slice_indices"]

        print("\n  Slices flagged HEMORRHAGE (any):")
        print(_fmt_slices(hem_any_idx, n_total))
        if hem["patient_positive"]:
            for label in HEMORRHAGE_LABELS[1:]:
                sub = hem["subtypes"][label]
                if sub["patient_positive"]:
                    print(f"    - {label}:")
                    print("    " + _fmt_slices(sub["positive_slice_indices"], n_total))

        print("\n  Slices flagged ISCHEMIC:")
        print(_fmt_slices(isch_idx, n_total))

        print(f"\n{'=' * 70}")

    if show_slices and n_total > 0:
        print(f"\n{'-' * 70}")
        print(f"  PER-SLICE DETAILS ({n_total} slices)")
        print(f"{'-' * 70}")
        for res in all_results:
            hem = res["results"]["hemorrhage"]
            isch = res["results"]["ischemic"]["ischemic_stroke"]
            any_hem = hem["any"]
            h_status = "HEM+" if any_hem["positive"] else "neg "
            i_status = "ISC+" if isch["positive"] else "neg "
            print(f"  Slice {res['slice_index']:3d}: "
                  f"{h_status} (p={any_hem['probability']:.3f})  "
                  f"{i_status} (p={isch['probability']:.3f})")

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
    parser.add_argument("--show-slices", action="store_true",
                        help="Show per-slice details in console output.")
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
    print_summary(all_results, patient_meta=patient_meta,
                  patient_agg=patient_agg, show_slices=args.show_slices)

    # Always write the patient-level overview plot when there are
    # multiple slices — this is the easiest way to see which slices fired.
    if len(all_results) > 1:
        summary_plot = out_dir / "patient_summary.png"
        try:
            visualize_patient_summary(all_results, patient_agg,
                                      patient_meta, summary_plot)
            print(f"Patient summary plot: {summary_plot}")
        except Exception as e:
            print(f"  WARNING: could not save patient summary plot: {e}")

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
