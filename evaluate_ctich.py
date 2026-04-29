#!/usr/bin/env python3
"""
Evaluate the CT brain pipeline on the CT-ICH dataset (Hssayeni et al., 2020).
============================================================================
Dataset: PhysioNet "Computed Tomography Images for Intracranial Hemorrhage
Detection and Segmentation" v1.0.0 — 82 patients, brain-windowed JPGs,
slice-level hemorrhage labels (5 subtypes + "no hemorrhage").

Layout expected:
  <root>/Patients_CT/<PatientID>/brain/<slice>.jpg
  <root>/hemorrhage_diagnosis.csv

The dataset stores already brain-windowed JPG slices (NO HU values).  We
invert the brain window (center=40, width=80) to recover pseudo-HU just
like evaluate_kaggle.py, then feed slices into the same pipeline.

Patient-level aggregation: max-pool across all slices (same strategy as
evaluate_cq500.py), so the patient is positive if ANY slice fires.

Usage:
  python evaluate_ctich.py --dataset-path /path/to/ct_ich_download/computed-tomography-images-...-1.0.0
  python evaluate_ctich.py --dataset-path ... --limit 5            # smoke test
  python evaluate_ctich.py --dataset-path ... --output-dir output_ctich
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

from pipeline import (
    HEMORRHAGE_LABELS,
    HEMORRHAGE_MODEL_DIR,
    ISCHEMIC_MODEL_PATH,
    aggregate_patient_results,
    load_hemorrhage_models,
    load_ischemic_model,
    predict_hemorrhage_batch,
    predict_ischemic_batch,
    prepare_hemorrhage_input,
    prepare_ischemic_input,
)

# CT-ICH labels: subtype columns in hemorrhage_diagnosis.csv
SUBTYPE_COLS = {
    "intraventricular": "Intraventricular",
    "intraparenchymal": "Intraparenchymal",
    "subarachnoid":     "Subarachnoid",
    "epidural":         "Epidural",
    "subdural":         "Subdural",
}

# Brain-window parameters used by the dataset (standard 40/80 brain window)
BRAIN_WINDOW_CENTER = 40.0
BRAIN_WINDOW_WIDTH = 80.0


# ── Image loading ──────────────────────────────────────────────────────────

def load_image_as_pseudo_hu(path: Path) -> np.ndarray:
    """Load a brain-windowed JPG and invert the window to pseudo-HU [0, 80]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if img.shape != (512, 512):
        img = cv2.resize(img, (512, 512))
    low = BRAIN_WINDOW_CENTER - BRAIN_WINDOW_WIDTH / 2   # 0 HU
    high = BRAIN_WINDOW_CENTER + BRAIN_WINDOW_WIDTH / 2  # 80 HU
    return img.astype(np.float32) / 255.0 * (high - low) + low


# ── GT loading ─────────────────────────────────────────────────────────────

def load_ctich_labels(csv_path: Path) -> dict[str, dict]:
    """
    Parse hemorrhage_diagnosis.csv and aggregate to patient-level.

    Returns: {patient_id: {"any": 0/1, "<subtype>": 0/1, "n_pos_slices": int, "n_slices": int}}
    """
    per_patient: dict[str, dict] = defaultdict(
        lambda: {"any": 0, "n_pos_slices": 0, "n_slices": 0,
                 **{k: 0 for k in SUBTYPE_COLS}}
    )
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pid = row["PatientNumber"].zfill(3)
            entry = per_patient[pid]
            entry["n_slices"] += 1
            slice_pos = False
            for sub_name, col in SUBTYPE_COLS.items():
                if int(row[col]) == 1:
                    entry[sub_name] = 1
                    slice_pos = True
            if slice_pos:
                entry["n_pos_slices"] += 1
                entry["any"] = 1
    return dict(per_patient)


# ── Per-patient inference ──────────────────────────────────────────────────

def list_patient_slices(patient_dir: Path) -> list[tuple[int, Path]]:
    """Return [(slice_index, path)] sorted by slice number, skipping seg masks."""
    brain_dir = patient_dir / "brain"
    if not brain_dir.is_dir():
        return []
    slices: list[tuple[int, Path]] = []
    for p in brain_dir.iterdir():
        if p.suffix.lower() != ".jpg":
            continue
        if "_HGE_Seg" in p.stem:  # ground-truth segmentation overlays
            continue
        try:
            idx = int(p.stem)
        except ValueError:
            continue
        slices.append((idx, p))
    slices.sort(key=lambda t: t[0])
    return slices


def run_one_patient(
    patient_dir: Path,
    hem_models: list,
    isch_model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> dict | None:
    """Run both models on every slice and return aggregate_patient_results dict."""
    slices = list_patient_slices(patient_dir)
    if not slices:
        return None

    hem_inputs, isch_inputs, slice_indices = [], [], []
    for idx, path in slices:
        try:
            hu = load_image_as_pseudo_hu(path)
            hem_inputs.append(prepare_hemorrhage_input(hu))
            isch_inputs.append(prepare_ischemic_input(hu))
            slice_indices.append(idx)
        except Exception as e:
            print(f"    SKIP slice {path.name}: {e}")

    if not hem_inputs:
        return None

    hem_results = predict_hemorrhage_batch(hem_models, hem_inputs, device, batch_size)
    isch_results = predict_ischemic_batch(isch_model, isch_inputs, device, batch_size)

    # Build the per-slice dicts that aggregate_patient_results expects
    all_results = []
    for idx, hem, isch in zip(slice_indices, hem_results, isch_results):
        all_results.append({
            "slice_index": idx,
            "results": {"hemorrhage": hem, "ischemic": isch},
        })

    agg = aggregate_patient_results(all_results)
    agg["n_slices"] = len(all_results)
    return agg


# ── Evaluation loop ────────────────────────────────────────────────────────

def evaluate(
    patient_dirs: list[Path],
    labels: dict[str, dict],
    hem_models: list,
    isch_model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    out_dir: Path,
) -> tuple[Path, list[dict]]:
    csv_path = out_dir / "patient_predictions.csv"

    fieldnames = [
        "patient", "n_slices",
        "gt_any", "pred_any", "p_any",
        *[f"gt_{l}" for l in HEMORRHAGE_LABELS if l != "any"],
        *[f"pred_{l}" for l in HEMORRHAGE_LABELS if l != "any"],
        *[f"p_{l}" for l in HEMORRHAGE_LABELS if l != "any"],
        "pred_ischemic", "p_ischemic",
        "in_gt_csv",
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    n_total = len(patient_dirs)
    t0 = time.time()

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for i, pdir in enumerate(patient_dirs, 1):
            pid = pdir.name.zfill(3)
            t_p = time.time()
            try:
                agg = run_one_patient(pdir, hem_models, isch_model, device, batch_size)
            except Exception as e:
                print(f"  [{i}/{n_total}] {pid}  ERROR: {e}")
                continue
            if agg is None:
                print(f"  [{i}/{n_total}] {pid}  no usable slices, skipping")
                continue

            hem = agg["hemorrhage"]
            isch = agg["ischemic"]
            gt = labels.get(pid)

            row = {
                "patient": pid,
                "n_slices": agg["n_slices"],
                "gt_any": (gt or {}).get("any", ""),
                "pred_any": int(hem["patient_positive"]),
                "p_any": hem["subtypes"]["any"]["max_probability"],
                "pred_ischemic": int(isch["patient_positive"]),
                "p_ischemic": isch["max_probability"],
                "in_gt_csv": int(gt is not None),
            }
            for l in HEMORRHAGE_LABELS:
                if l == "any":
                    continue
                row[f"gt_{l}"]   = (gt or {}).get(l, "")
                row[f"pred_{l}"] = int(hem["subtypes"][l]["patient_positive"])
                row[f"p_{l}"]    = hem["subtypes"][l]["max_probability"]

            writer.writerow(row)
            fh.flush()
            rows.append(row)

            elapsed = time.time() - t_p
            rate = i / max(time.time() - t0, 1e-6)
            tag = "" if gt is not None else "  [no GT]"
            print(
                f"  [{i}/{n_total}] {pid}  "
                f"slices={agg['n_slices']:3d}  "
                f"p_any={hem['subtypes']['any']['max_probability']:.3f}  "
                f"pred={'POS' if hem['patient_positive'] else 'neg'}  "
                f"gt={'POS' if (gt and gt['any']) else 'neg'}  "
                f"({elapsed:.1f}s, avg {rate*60:.1f} pat/min){tag}"
            )

    return csv_path, rows


# ── Reporting ──────────────────────────────────────────────────────────────

def metrics(tp: int, fp: int, tn: int, fn: int) -> tuple[float, float, float, float, float]:
    acc = (tp + tn) / max(tp + fp + tn + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return acc, prec, rec, spec, f1


def print_report(rows: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("  PATIENT-LEVEL RESULTS — CT-ICH")
    print("=" * 72)

    eval_rows = [r for r in rows if r["in_gt_csv"]]
    n = len(eval_rows)
    if n == 0:
        print("  No patients with ground truth — nothing to score.")
        return

    # "any" hemorrhage
    tp = fp = tn = fn = 0
    for r in eval_rows:
        gt, pr = int(r["gt_any"]), int(r["pred_any"])
        if gt and pr:        tp += 1
        elif gt and not pr:  fn += 1
        elif not gt and pr:  fp += 1
        else:                tn += 1
    acc, prec, rec, spec, f1 = metrics(tp, fp, tn, fn)
    print(f"\n  HEMORRHAGE (any)   n={n}")
    print(f"    TP={tp}  FP={fp}  TN={tn}  FN={fn}")
    print(f"    acc={acc*100:.2f}%  prec={prec*100:.2f}%  "
          f"recall={rec*100:.2f}%  spec={spec*100:.2f}%  F1={f1*100:.2f}%")

    # Per-subtype
    print(f"\n  SUBTYPES (patient-level)")
    for l in HEMORRHAGE_LABELS:
        if l == "any":
            continue
        tp = fp = tn = fn = 0
        for r in eval_rows:
            gt = int(r.get(f"gt_{l}") or 0)
            pr = int(r.get(f"pred_{l}") or 0)
            if gt and pr:        tp += 1
            elif gt and not pr:  fn += 1
            elif not gt and pr:  fp += 1
            else:                tn += 1
        acc, prec, rec, spec, f1 = metrics(tp, fp, tn, fn)
        print(f"    {l:18s} acc={acc*100:6.2f}%  "
              f"recall={rec*100:6.2f}%  spec={spec*100:6.2f}%  F1={f1*100:6.2f}%  "
              f"(TP={tp} FP={fp} TN={tn} FN={fn})")

    print()
    print("  Note: CT-ICH has no ischemic ground truth — only hemorrhage scored.")
    print()


# ── Main ───────────────────────────────────────────────────────────────────

def find_dataset_root(path: Path) -> Path:
    """Locate the directory that contains Patients_CT/ + hemorrhage_diagnosis.csv."""
    candidates = [path] + [p for p in path.rglob("*") if p.is_dir()]
    for c in candidates:
        if (c / "Patients_CT").is_dir() and (c / "hemorrhage_diagnosis.csv").is_file():
            return c
    raise FileNotFoundError(
        f"Could not find Patients_CT/ + hemorrhage_diagnosis.csv under {path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    default_device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    parser.add_argument("--dataset-path", required=True,
                        help="Path to the CT-ICH dataset root (or a parent dir containing it)")
    parser.add_argument("--device", default=default_device,
                        choices=["cpu", "cuda", "mps"], help="Inference device")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N patients (smoke test)")
    parser.add_argument("--output-dir", default="output_ctich",
                        help="Where to write CSV / report")
    args = parser.parse_args()

    root = find_dataset_root(Path(args.dataset_path))
    print(f"Dataset root: {root}")

    # 1. Load labels
    labels = load_ctich_labels(root / "hemorrhage_diagnosis.csv")
    print(f"  Loaded GT for {len(labels)} patients")

    # 2. Discover patient folders
    patient_dirs = sorted(
        p for p in (root / "Patients_CT").iterdir()
        if p.is_dir() and (p / "brain").is_dir()
    )
    if args.limit is not None:
        patient_dirs = patient_dirs[:args.limit]
    print(f"  Found {len(patient_dirs)} patient folder(s)")

    # 3. Models
    device = torch.device(args.device)
    print(f"\nLoading models on {device} …")
    hem_models = load_hemorrhage_models(HEMORRHAGE_MODEL_DIR, device)
    isch_model = load_ischemic_model(ISCHEMIC_MODEL_PATH, device)
    if not hem_models:
        print("ERROR: no hemorrhage model folds loaded", file=sys.stderr)
        return 1

    # 4. Evaluate
    out_dir = Path(args.output_dir)
    print(f"\nRunning patient-level inference (batch size {args.batch_size}) …")
    csv_path, rows = evaluate(
        patient_dirs, labels, hem_models, isch_model,
        device, args.batch_size, out_dir,
    )

    # 5. Report
    print_report(rows)
    print(f"Per-patient predictions: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
