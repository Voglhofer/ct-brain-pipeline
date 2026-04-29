#!/usr/bin/env python3
"""
Evaluate the CT brain pipeline on the CQ500 dataset.
====================================================
Patient-level evaluation with max-pool aggregation across slices.

CQ500 (http://headctstudy.qure.ai/dataset) contains 491 head CT scans
labeled by 3 radiologists for intracranial hemorrhage and 5 subtypes
(IPH, IVH, SAH, EDH, SDH) plus fractures / mass effect / midline shift.

This script:
  1. Iterates over patient folders under --dataset-path
     (each folder = one patient = one CT study with multiple DICOM series)
  2. Per patient: filters to the best axial brain CT series, sorts slices
     by z-position, runs both models with batched inference.
  3. Patient-level diagnosis via MAX POOLING across slices
     (the existing pipeline.aggregate_patient_results does exactly this).
  4. Compares against majority vote of R1/R2/R3 reads from reads.csv.
  5. Writes per-patient CSV + per-class accuracy / confusion matrix.

Note: CQ500 has no ischemic stroke labels. Ischemic predictions are
recorded in the CSV for reference but only hemorrhage metrics are computed.

Usage:
  python evaluate_cq500.py \\
      --dataset-path /path/to/CQ500 \\
      --reads-csv /path/to/reads.csv

  # Quick smoke test (5 patients)
  python evaluate_cq500.py --dataset-path ... --reads-csv ... --limit 5

  # Skip already-processed patients (idempotent rerun)
  python evaluate_cq500.py ... --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pydicom
import torch

from pipeline import (
    HEMORRHAGE_LABELS,
    HEMORRHAGE_MODEL_DIR,
    ISCHEMIC_MODEL_PATH,
    aggregate_patient_results,
    collect_input_paths,
    dicom_to_hu,
    filter_dicom_series,
    load_hemorrhage_models,
    load_ischemic_model,
    run_pipeline_batched,
    sort_dicom_by_position,
)


# ── CQ500 ground-truth handling ────────────────────────────────────────────

# Map our model's hemorrhage subtype names to CQ500 reads.csv column suffixes
# (CQ500 does not have a separate "any" column — it's the ICH umbrella label.)
SUBTYPE_TO_CQ500 = {
    "any": "ICH",
    "intraparenchymal": "IPH",
    "intraventricular": "IVH",
    "subarachnoid": "SAH",
    "epidural": "EDH",
    "subdural": "SDH",
}


def load_cq500_labels(reads_csv: Path) -> dict[str, dict[str, int]]:
    """
    Parse CQ500 reads.csv and return {patient_id: {subtype: 0/1}} using
    majority vote across R1/R2/R3 (≥2 of 3 readers positive).
    """
    labels: dict[str, dict[str, int]] = {}
    with open(reads_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = (row.get("name") or row.get("Name") or "").strip()
            if not name:
                continue
            patient_labels: dict[str, int] = {}
            for our_label, cq_suffix in SUBTYPE_TO_CQ500.items():
                votes = 0
                for reader_id in ("R1", "R2", "R3"):
                    col = f"{reader_id}:{cq_suffix}"
                    val = row.get(col, "")
                    try:
                        if int(float(val)) == 1:
                            votes += 1
                    except (ValueError, TypeError):
                        pass
                patient_labels[our_label] = 1 if votes >= 2 else 0
            labels[name] = patient_labels
    return labels


def normalize_patient_id(name: str) -> str:
    """Normalise patient folder names so they match reads.csv `name` field.

    reads.csv uses 'CQ500-CT-0' (no zero padding). Folder names can vary:
      CQ500CT0, CQ500-CT-0, CQ500-CT-000, etc.
    """
    s = name.upper().replace("_", "-")
    # Strip everything up to "CT" and rebuild
    if "CT" in s:
        idx = s.rfind("CT")
        prefix = s[:idx + 2]
        suffix = s[idx + 2:].lstrip("-")
        # Strip leading zeros from suffix
        suffix = str(int(suffix)) if suffix.isdigit() else suffix
        # Always emit as CQ500-CT-N
        return f"CQ500-CT-{suffix}"
    return name


# ── Patient discovery ──────────────────────────────────────────────────────

def find_patient_folders(root: Path) -> list[Path]:
    """Return one folder per patient under root.

    Heuristic: any directory whose name (case-insensitive) contains 'CQ500'
    or 'CT-<num>' is treated as a patient folder. Otherwise we fall back to
    direct children of `root`.
    """
    patients: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if "CQ500" in p.name.upper() or p.name.upper().startswith("CT-"):
            patients.append(p)
    if not patients:
        patients = [p for p in sorted(root.iterdir()) if p.is_dir()]
    return patients


# ── Per-patient pipeline ───────────────────────────────────────────────────

def run_one_patient(
    patient_dir: Path,
    hem_models: list,
    isch_model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    no_filter: bool,
) -> dict | None:
    """Run the full pipeline on one CQ500 patient folder.

    Returns a dict with the patient-level (max-pooled) diagnosis, or None
    if the patient could not be processed.
    """
    paths = collect_input_paths([str(patient_dir)])
    if not paths:
        return None

    if not no_filter:
        paths = filter_dicom_series(paths, verbose=False)
    if not paths:
        return None

    paths = sort_dicom_by_position(paths)

    # Read all slices to HU
    images_hu: list[np.ndarray] = []
    for p in paths:
        try:
            images_hu.append(dicom_to_hu(str(p)))
        except Exception:
            continue
    if not images_hu:
        return None

    n = len(images_hu)
    per_slice = run_pipeline_batched(
        images_hu, hem_models, isch_model, device, batch_size=batch_size
    )

    # Wrap into format expected by aggregate_patient_results
    wrapped = [
        {"slice_index": i, "results": r}
        for i, r in enumerate(per_slice)
    ]
    agg = aggregate_patient_results(wrapped)
    agg["n_slices"] = n
    return agg


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(
    patient_dirs: list[Path],
    labels: dict[str, dict[str, int]],
    hem_models: list,
    isch_model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    no_filter: bool,
    out_dir: Path,
    resume: bool,
) -> tuple[Path, list[dict]]:
    csv_path = out_dir / "patient_predictions.csv"

    fieldnames = [
        "patient", "n_slices",
        "gt_any", "pred_any", "p_any",
        *[f"gt_{l}" for l in HEMORRHAGE_LABELS if l != "any"],
        *[f"pred_{l}" for l in HEMORRHAGE_LABELS if l != "any"],
        *[f"p_{l}" for l in HEMORRHAGE_LABELS if l != "any"],
        "pred_ischemic", "p_ischemic",
        "in_reads_csv",
    ]

    # Resume support: read already-processed patients
    done: set[str] = set()
    rows_existing: list[dict] = []
    if resume and csv_path.exists():
        with open(csv_path, newline="") as fh:
            for row in csv.DictReader(fh):
                done.add(row["patient"])
                rows_existing.append(row)
        print(f"  Resume: {len(done)} patient(s) already processed")

    mode = "a" if (resume and csv_path.exists()) else "w"
    fh = open(csv_path, mode, newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if mode == "w":
        writer.writeheader()

    n_total = len(patient_dirs)
    t0 = time.time()
    rows: list[dict] = list(rows_existing)
    try:
        for i, pdir in enumerate(patient_dirs, 1):
            pid = normalize_patient_id(pdir.name)
            if pid in done:
                continue
            t_p = time.time()
            try:
                agg = run_one_patient(
                    pdir, hem_models, isch_model, device, batch_size, no_filter
                )
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
                "in_reads_csv": int(gt is not None),
            }
            for l in HEMORRHAGE_LABELS:
                if l == "any":
                    continue
                row[f"gt_{l}"] = (gt or {}).get(l, "")
                row[f"pred_{l}"] = int(hem["subtypes"][l]["patient_positive"])
                row[f"p_{l}"] = hem["subtypes"][l]["max_probability"]

            writer.writerow(row)
            fh.flush()
            rows.append(row)

            elapsed = time.time() - t_p
            done_n = i
            rate = done_n / max(time.time() - t0, 1e-6)
            tag = "" if gt is not None else "  [no GT]"
            print(
                f"  [{i}/{n_total}] {pid}  "
                f"slices={agg['n_slices']:3d}  "
                f"p_any={hem['subtypes']['any']['max_probability']:.3f}  "
                f"pred={'POS' if hem['patient_positive'] else 'neg'}  "
                f"({elapsed:.1f}s, avg {rate*60:.1f} pat/min){tag}"
            )
    finally:
        fh.close()

    return csv_path, rows


# ── Reporting ──────────────────────────────────────────────────────────────

def _to_int(x) -> int | None:
    if x is None or x == "":
        return None
    try:
        return int(float(x))
    except (ValueError, TypeError):
        return None


def metrics(tp: int, fp: int, tn: int, fn: int) -> dict:
    n = tp + fp + tn + fn
    return {
        "n": n,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "acc":  (tp + tn) / max(n, 1),
        "prec": tp / max(tp + fp, 1),
        "rec":  tp / max(tp + fn, 1),
        "spec": tn / max(tn + fp, 1),
        "f1":   2 * tp / max(2 * tp + fp + fn, 1),
    }


def print_report(rows: list[dict], out_dir: Path) -> None:
    print("\n" + "=" * 76)
    print("  CQ500 PATIENT-LEVEL RESULTS  (max-pool aggregation)")
    print("=" * 76)

    summary: dict[str, dict] = {}
    for label in HEMORRHAGE_LABELS:
        tp = fp = tn = fn = 0
        for r in rows:
            gt = _to_int(r.get(f"gt_{label}"))
            pred = _to_int(r.get(f"pred_{label}"))
            if gt is None or pred is None:
                continue
            if gt == 1 and pred == 1: tp += 1
            elif gt == 1 and pred == 0: fn += 1
            elif gt == 0 and pred == 1: fp += 1
            else: tn += 1
        summary[label] = metrics(tp, fp, tn, fn)

    header = f"  {'Subtype':<20s} {'n':>5s}  {'acc':>7s} {'prec':>7s} {'recall':>7s} {'spec':>7s} {'F1':>7s}   {'TP':>4s} {'FP':>4s} {'TN':>4s} {'FN':>4s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label in HEMORRHAGE_LABELS:
        m = summary[label]
        if m["n"] == 0:
            continue
        print(
            f"  {label:<20s} {m['n']:>5d}  "
            f"{m['acc']*100:>6.2f}% {m['prec']*100:>6.2f}% {m['rec']*100:>6.2f}% "
            f"{m['spec']*100:>6.2f}% {m['f1']*100:>6.2f}%   "
            f"{m['tp']:>4d} {m['fp']:>4d} {m['tn']:>4d} {m['fn']:>4d}"
        )

    n_with_gt = sum(1 for r in rows if _to_int(r.get("gt_any")) is not None)
    n_total = len(rows)
    n_isch_pos = sum(1 for r in rows if _to_int(r.get("pred_ischemic")) == 1)
    print()
    print(f"  Patients evaluated:  {n_total}")
    print(f"    with GT in reads.csv: {n_with_gt}")
    print(f"    without GT:           {n_total - n_with_gt}")
    print(f"  Ischemic predictions (no CQ500 GT): {n_isch_pos}/{n_total} positive")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as fh:
        json.dump({
            "n_patients": n_total,
            "n_with_gt": n_with_gt,
            "metrics": summary,
        }, fh, indent=2)
    print(f"\n  Summary JSON:        {summary_path}")


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset-path", required=True,
                        help="Root folder containing CQ500-CT-* patient folders")
    parser.add_argument("--reads-csv", required=True,
                        help="Path to CQ500 reads.csv (radiologist labels)")
    default_device = (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    parser.add_argument("--device", default=default_device,
                        choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of patients to process")
    parser.add_argument("--no-filter", action="store_true",
                        help="Skip DICOM series filtering")
    parser.add_argument("--output-dir", default="output_cq500")
    parser.add_argument("--resume", action="store_true",
                        help="Skip patients already in patient_predictions.csv")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.dataset_path).expanduser()
    if not root.exists():
        print(f"ERROR: dataset path does not exist: {root}", file=sys.stderr)
        return 1
    reads_path = Path(args.reads_csv).expanduser()
    if not reads_path.exists():
        print(f"ERROR: reads.csv does not exist: {reads_path}", file=sys.stderr)
        return 1

    print(f"Dataset path: {root}")
    print(f"Reads CSV:    {reads_path}")
    labels = load_cq500_labels(reads_path)
    print(f"Loaded ground-truth labels for {len(labels)} patient(s)")

    patient_dirs = find_patient_folders(root)
    if args.limit is not None:
        patient_dirs = patient_dirs[:args.limit]
    print(f"Found {len(patient_dirs)} patient folder(s)")

    device = torch.device(args.device)
    print(f"\nLoading models on {device} …")
    hem_models = load_hemorrhage_models(HEMORRHAGE_MODEL_DIR, device)
    isch_model = load_ischemic_model(ISCHEMIC_MODEL_PATH, device)
    if not hem_models:
        print("ERROR: no hemorrhage model folds loaded", file=sys.stderr)
        return 1

    print(f"\nProcessing patients (batch size {args.batch_size})…")
    csv_path, rows = evaluate(
        patient_dirs, labels, hem_models, isch_model, device,
        args.batch_size, args.no_filter, out_dir, args.resume,
    )

    print(f"\nPer-patient predictions: {csv_path}")
    print_report(rows, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
