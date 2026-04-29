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

The dataset stores already brain-windowed JPG slices (NO HU values).
Preprocessing mirrors the Bachelor_RSNA_test SeuTao pipeline
(`run_ctich_full_ensemble.py`):

  * **Hemorrhage** — the JPG is read directly as uint8 grayscale and
    stacked with its previous and next neighbour slices to form the
    3-channel input that the hemorrhage DenseNet expects. ImageNet-style
    normalisation is then applied by `pipeline.to_tensor`. This matches
    the training-time preprocessing of the RSNA 2019 hemorrhage models.
    The full 3-backbone × 5-fold ensemble (15 models) is used when the
    `models/hemorrhage/{DenseNet121,DenseNet169,SE-ResNeXt101}/`
    subdirectories are present.

  * **Ischemic** — the JPG is converted back to pseudo-HU by inverting
    the brain window (linear map uint8 [0,255] -> HU [0,80]). Inside
    that range the round-trip is exact, so the brain-window and
    stroke-window (32/8) channels are reconstructed losslessly; only
    the soft-tissue window (40/120) is clipped at HU=80 and is
    therefore an approximation. CT-ICH has no ischemic ground truth,
    so these predictions are reported but not scored.

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
    prepare_ischemic_input,
)

# Brain-window parameters of the dataset (standard 40/80 brain window).
# Used to invert the JPG -> pseudo-HU for the ischemic preprocessing.
BRAIN_WINDOW_CENTER = 40.0
BRAIN_WINDOW_WIDTH = 80.0

# CT-ICH labels: subtype columns in hemorrhage_diagnosis.csv
SUBTYPE_COLS = {
    "intraventricular": "Intraventricular",
    "intraparenchymal": "Intraparenchymal",
    "subarachnoid":     "Subarachnoid",
    "epidural":         "Epidural",
    "subdural":         "Subdural",
}

# ── Image loading ─────────────────────────────────────────────

def load_image_uint8(path: Path) -> np.ndarray:
    """Load a brain-windowed JPG as a 512×512 uint8 grayscale array.

    The dataset is already brain-windowed (WL=40, WW=80). The original
    Bachelor_RSNA_test pipeline feeds these bytes straight into the
    DenseNet without any HU round-trip, which is what we replicate here.
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if img.shape != (512, 512):
        img = cv2.resize(img, (512, 512))
    return img


def make_3ch_neighbours(slices_uint8: list[np.ndarray], idx: int) -> np.ndarray:
    """Stack [prev, curr, next] grayscale slices as a HWC uint8 image.

    Mirrors `make_3ch` in run_ctich_full_ensemble.py: at scan boundaries
    the missing neighbour is replaced by the current slice (clamp).
    """
    prev_idx = max(0, idx - 1)
    next_idx = min(len(slices_uint8) - 1, idx + 1)
    return np.stack(
        [slices_uint8[prev_idx], slices_uint8[idx], slices_uint8[next_idx]],
        axis=-1,
    )


def uint8_to_pseudo_hu(img_uint8: np.ndarray) -> np.ndarray:
    """Invert the brain window to recover pseudo-HU in [0, 80].

    Used only for the ischemic model, which needs HU values to apply its
    multi-window preprocessing. The map is exact within the brain HU
    range; values outside [0, 80] are not recoverable from the JPG.
    """
    low = BRAIN_WINDOW_CENTER - BRAIN_WINDOW_WIDTH / 2   # 0 HU
    high = BRAIN_WINDOW_CENTER + BRAIN_WINDOW_WIDTH / 2  # 80 HU
    return img_uint8.astype(np.float32) / 255.0 * (high - low) + low


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
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
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
    """Run hemorrhage + ischemic ensembles on every slice and aggregate.

    Hemorrhage uses the SeuTao-faithful direct-uint8 + prev/curr/next
    neighbour stacking. Ischemic uses pseudo-HU inversion to recover
    HU values within the brain window range and then runs the standard
    multi-window preprocessing from `pipeline.prepare_ischemic_input`.
    """
    slices = list_patient_slices(patient_dir)
    if not slices:
        return None

    slices_uint8: list[np.ndarray] = []
    slice_indices: list[int] = []
    for idx, path in slices:
        try:
            slices_uint8.append(load_image_uint8(path))
            slice_indices.append(idx)
        except Exception as e:
            print(f"    SKIP slice {path.name}: {e}")

    if not slices_uint8:
        return None

    hem_inputs = [make_3ch_neighbours(slices_uint8, i) for i in range(len(slices_uint8))]
    hem_results = predict_hemorrhage_batch(hem_models, hem_inputs, device, batch_size)

    isch_inputs = [prepare_ischemic_input(uint8_to_pseudo_hu(s)) for s in slices_uint8]
    isch_results = predict_ischemic_batch(isch_model, isch_inputs, device, batch_size)

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
                # Ischemic predictions are reported but unscored
                # (CT-ICH has no ischemic ground truth).
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


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Mann-Whitney-U based AUC. Returns NaN if only one class present."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # Rank-sum AUC with tie correction.
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    # Average ties
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg_rank = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg_rank
        i = j + 1
    n_pos = pos.size
    n_neg = neg.size
    rank_sum_pos = ranks[y_true == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_auc_ci(
    y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 1000, seed: int = 42
) -> tuple[float, float]:
    """Patient-level percentile bootstrap 95% CI for AUC."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = roc_auc(y_true[idx], y_score[idx])
        if not np.isnan(a):
            aucs.append(a)
    if not aucs:
        return float("nan"), float("nan")
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


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
    y_true_any = np.array([int(r["gt_any"]) for r in eval_rows])
    y_score_any = np.array([float(r["p_any"]) for r in eval_rows])
    auc_any = roc_auc(y_true_any, y_score_any)
    ci_lo, ci_hi = bootstrap_auc_ci(y_true_any, y_score_any)
    print(f"\n  HEMORRHAGE (any)   n={n}")
    print(f"    AUC={auc_any:.4f}  [95% CI {ci_lo:.4f}, {ci_hi:.4f}]")
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
        y_true = np.array([int(r.get(f"gt_{l}") or 0) for r in eval_rows])
        y_score = np.array([float(r.get(f"p_{l}") or 0.0) for r in eval_rows])
        auc = roc_auc(y_true, y_score)
        auc_str = f"AUC={auc:.4f}" if not np.isnan(auc) else "AUC=  n/a "
        print(f"    {l:18s} {auc_str}  "
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
