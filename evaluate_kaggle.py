#!/usr/bin/env python3
"""
Evaluate the CT brain pipeline on the Kaggle Brain Stroke CT Dataset.
====================================================================
Dataset: ozguraslank/brain-stroke-ct-dataset

This dataset contains 2D PNG/JPG CT slices already brain-windowed and
organised by class (Bleeding / Ischemia / Normal / External_Test) — i.e.
NOT raw DICOM/NIfTI volumes with Hounsfield Units.

Because the original HU values are lost, we feed the pre-windowed grayscale
image directly to both models (replicated across the 3 input channels).
Results are therefore a *practical baseline* on this dataset, not a
fully apples-to-apples evaluation.

What the script does:
  1. Downloads the dataset via kagglehub (cached locally).
  2. Iterates over every image in Bleeding / Ischemia / Normal.
  3. Runs the hemorrhage 5-fold ensemble + ischemic classifier per slice.
  4. Compares predictions vs. ground-truth folder label.
  5. Writes per-image CSV + per-class accuracy / confusion matrix.

Usage:
  python evaluate_kaggle.py                    # default: all 3 classes, GPU if available
  python evaluate_kaggle.py --device cuda
  python evaluate_kaggle.py --limit 100        # quick smoke test
  python evaluate_kaggle.py --output-dir output_kaggle/
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

# Reuse model loaders + inference from the main pipeline
from pipeline import (
    HEMORRHAGE_LABELS,
    HEMORRHAGE_MODEL_DIR,
    ISCHEMIC_MODEL_PATH,
    load_hemorrhage_models,
    load_ischemic_model,
    predict_hemorrhage_batch,
    predict_ischemic_batch,
    prepare_hemorrhage_input,
    prepare_ischemic_input,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Kaggle dataset class folders → ground-truth labels
GT_CLASSES = {
    "Bleeding":  {"hemorrhage": True,  "ischemic": False},
    "Ischemia":  {"hemorrhage": False, "ischemic": True},
    "Normal":    {"hemorrhage": False, "ischemic": False},
}

# The Kaggle PNGs are pre-windowed with the standard brain window.
# Inverting that window recovers pseudo-HU values in the [0, 80] HU range,
# which lets us run the pipeline's real preprocessing (multi-window for the
# ischemic model in particular). HU outside [0, 80] is clipped, but the
# stroke window (center=32, width=8 → HU [28, 36]) and brain window
# (center=40, width=80 → HU [0, 80]) both live inside that range, so most
# diagnostically useful contrast is preserved.
BRAIN_WINDOW_CENTER = 40.0
BRAIN_WINDOW_WIDTH = 80.0


# ── Image loading & preprocessing ──────────────────────────────────────────

def load_image_as_pseudo_hu(path: Path) -> np.ndarray:
    """
    Load a Kaggle PNG and invert the brain window to pseudo-HU.

    NOTE: the dataset's Bleeding / Ischemia images are stored as RGBA with
    very small differences between R/G/B (presumably multi-window encoded),
    while Normal images are pure grayscale. Using channel differences would
    be a label leak, so we collapse to luminance via cv2.IMREAD_GRAYSCALE.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if img.shape != (512, 512):
        img = cv2.resize(img, (512, 512))

    low = BRAIN_WINDOW_CENTER - BRAIN_WINDOW_WIDTH / 2  # 0 HU
    high = BRAIN_WINDOW_CENTER + BRAIN_WINDOW_WIDTH / 2  # 80 HU
    return img.astype(np.float32) / 255.0 * (high - low) + low


def collect_images(root: Path, classes: list[str], limit: int | None) -> list[tuple[Path, str]]:
    """Return list of (image_path, class_label) for every image under root/<class>/**."""
    items: list[tuple[Path, str]] = []
    for cls in classes:
        cls_dir = root / cls
        if not cls_dir.exists():
            print(f"  WARNING: class folder not found: {cls_dir}")
            continue
        files = [p for p in cls_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
        files.sort()
        if limit is not None:
            files = files[:limit]
        items.extend((p, cls) for p in files)
        print(f"  {cls}: {len(files)} image(s)")
    return items


def find_dataset_root(download_path: Path) -> Path:
    """
    kagglehub returns the dataset root, but this dataset nests everything
    inside Brain_Stroke_CT_Dataset/. Walk down to the first directory that
    contains at least one of the expected class folders.
    """
    candidates = [download_path] + [p for p in download_path.rglob("*") if p.is_dir()]
    for c in candidates:
        if any((c / cls).is_dir() for cls in GT_CLASSES):
            return c
    raise FileNotFoundError(
        f"Could not locate Bleeding/Ischemia/Normal folders under {download_path}"
    )


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(
    items: list[tuple[Path, str]],
    hem_models: list,
    isch_model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    csv_path: Path,
) -> dict:
    """Run both models on all images, write per-image CSV, return aggregated stats."""
    n = len(items)
    print(f"\nRunning inference on {n} image(s) (batch size {batch_size})…")

    # Per-class counters
    stats: dict = {
        cls: {
            "n": 0,
            "hem_tp": 0, "hem_fp": 0, "hem_tn": 0, "hem_fn": 0,
            "isch_tp": 0, "isch_fp": 0, "isch_tn": 0, "isch_fn": 0,
        }
        for cls in GT_CLASSES
    }

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file", "class",
        "gt_hemorrhage", "pred_hemorrhage", "p_any",
        *[f"p_{l}" for l in HEMORRHAGE_LABELS],
        "gt_ischemic", "pred_ischemic", "p_ischemic",
    ]

    t0 = time.time()
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        # Process in batches to keep memory bounded
        for start in range(0, n, batch_size):
            batch = items[start:start + batch_size]
            hem_inputs, isch_inputs, valid = [], [], []
            for path, cls in batch:
                try:
                    hu = load_image_as_pseudo_hu(path)
                    # Use the pipeline's real preprocessing — brain window for
                    # hemorrhage, multi-window (brain/stroke/soft) for ischemic.
                    hem_inputs.append(prepare_hemorrhage_input(hu))
                    isch_inputs.append(prepare_ischemic_input(hu))
                    valid.append((path, cls))
                except Exception as e:
                    print(f"  SKIP {path.name}: {e}")
            if not valid:
                continue

            hem_results = predict_hemorrhage_batch(hem_models, hem_inputs, device, batch_size)
            isch_results = predict_ischemic_batch(isch_model, isch_inputs, device, batch_size)

            for (path, cls), hem, isch in zip(valid, hem_results, isch_results):
                gt = GT_CLASSES[cls]
                pred_hem = hem["any"]["positive"]
                pred_isch = isch["ischemic_stroke"]["positive"]

                row = {
                    "file": str(path),
                    "class": cls,
                    "gt_hemorrhage": int(gt["hemorrhage"]),
                    "pred_hemorrhage": int(pred_hem),
                    "p_any": hem["any"]["probability"],
                    "gt_ischemic": int(gt["ischemic"]),
                    "pred_ischemic": int(pred_isch),
                    "p_ischemic": isch["ischemic_stroke"]["probability"],
                }
                for l in HEMORRHAGE_LABELS:
                    row[f"p_{l}"] = hem[l]["probability"]
                writer.writerow(row)

                s = stats[cls]
                s["n"] += 1
                # hemorrhage confusion
                if gt["hemorrhage"] and pred_hem:        s["hem_tp"] += 1
                elif gt["hemorrhage"] and not pred_hem:  s["hem_fn"] += 1
                elif not gt["hemorrhage"] and pred_hem:  s["hem_fp"] += 1
                else:                                    s["hem_tn"] += 1
                # ischemic confusion
                if gt["ischemic"] and pred_isch:         s["isch_tp"] += 1
                elif gt["ischemic"] and not pred_isch:   s["isch_fn"] += 1
                elif not gt["ischemic"] and pred_isch:   s["isch_fp"] += 1
                else:                                    s["isch_tn"] += 1

            done = start + len(batch)
            if done % (batch_size * 10) == 0 or done >= n:
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  [{done}/{n}]  {rate:.1f} img/s")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({n / max(elapsed, 1e-6):.1f} img/s)")
    return stats


def print_report(stats: dict) -> None:
    print("\n" + "=" * 72)
    print("  PER-CLASS RESULTS")
    print("=" * 72)
    for cls, s in stats.items():
        if s["n"] == 0:
            continue
        gt = GT_CLASSES[cls]
        hem_correct = s["hem_tp"] + s["hem_tn"]
        isch_correct = s["isch_tp"] + s["isch_tn"]
        print(f"\n  {cls}  (n={s['n']}, gt_hem={int(gt['hemorrhage'])}, gt_isch={int(gt['ischemic'])})")
        print(f"    Hemorrhage  → acc {hem_correct/s['n']*100:6.2f}%   "
              f"TP={s['hem_tp']} FP={s['hem_fp']} TN={s['hem_tn']} FN={s['hem_fn']}")
        print(f"    Ischemic    → acc {isch_correct/s['n']*100:6.2f}%   "
              f"TP={s['isch_tp']} FP={s['isch_fp']} TN={s['isch_tn']} FN={s['isch_fn']}")

    # Overall metrics across all classes
    total = sum(s["n"] for s in stats.values())
    if total == 0:
        return
    hem_tp = sum(s["hem_tp"] for s in stats.values())
    hem_fp = sum(s["hem_fp"] for s in stats.values())
    hem_tn = sum(s["hem_tn"] for s in stats.values())
    hem_fn = sum(s["hem_fn"] for s in stats.values())
    isch_tp = sum(s["isch_tp"] for s in stats.values())
    isch_fp = sum(s["isch_fp"] for s in stats.values())
    isch_tn = sum(s["isch_tn"] for s in stats.values())
    isch_fn = sum(s["isch_fn"] for s in stats.values())

    def metrics(tp, fp, tn, fn):
        acc = (tp + tn) / max(tp + fp + tn + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        return acc, prec, rec, spec, f1

    print("\n" + "=" * 72)
    print(f"  OVERALL  (n={total})")
    print("=" * 72)
    a, p, r, sp, f1 = metrics(hem_tp, hem_fp, hem_tn, hem_fn)
    print(f"  Hemorrhage  acc={a*100:.2f}%  prec={p*100:.2f}%  "
          f"recall={r*100:.2f}%  spec={sp*100:.2f}%  F1={f1*100:.2f}%")
    a, p, r, sp, f1 = metrics(isch_tp, isch_fp, isch_tn, isch_fn)
    print(f"  Ischemic    acc={a*100:.2f}%  prec={p*100:.2f}%  "
          f"recall={r*100:.2f}%  spec={sp*100:.2f}%  F1={f1*100:.2f}%")
    print()


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default_device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    parser.add_argument("--device", default=default_device,
                        choices=["cpu", "cuda", "mps"], help="Inference device")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit images per class (smoke test)")
    parser.add_argument("--classes", nargs="+", default=list(GT_CLASSES.keys()),
                        choices=list(GT_CLASSES.keys()),
                        help="Which class folders to evaluate")
    parser.add_argument("--dataset-path", default=None,
                        help="Skip kagglehub download and use this local path")
    parser.add_argument("--output-dir", default="output_kaggle",
                        help="Where to write CSV / report")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download dataset (or use provided path)
    if args.dataset_path:
        dl_path = Path(args.dataset_path)
        print(f"Using local dataset path: {dl_path}")
    else:
        try:
            import kagglehub
        except ImportError:
            print("ERROR: kagglehub not installed. Run: pip install kagglehub", file=sys.stderr)
            return 1
        print("Downloading dataset via kagglehub: ozguraslank/brain-stroke-ct-dataset …")
        dl_path = Path(kagglehub.dataset_download("ozguraslank/brain-stroke-ct-dataset"))
        print(f"  Dataset cached at: {dl_path}")

    root = find_dataset_root(dl_path)
    print(f"Dataset root: {root}")

    # 2. Collect images
    print("\nCollecting images:")
    items = collect_images(root, args.classes, args.limit)
    if not items:
        print("ERROR: no images found", file=sys.stderr)
        return 1

    # 3. Load models
    device = torch.device(args.device)
    print(f"\nLoading models on {device} …")
    hem_models = load_hemorrhage_models(HEMORRHAGE_MODEL_DIR, device)
    isch_model = load_ischemic_model(ISCHEMIC_MODEL_PATH, device)
    if not hem_models:
        print("ERROR: no hemorrhage model folds loaded", file=sys.stderr)
        return 1

    # 4. Evaluate
    csv_path = out_dir / "predictions.csv"
    stats = evaluate(items, hem_models, isch_model, device, args.batch_size, csv_path)

    # 5. Report
    print_report(stats)
    print(f"Per-image predictions: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
