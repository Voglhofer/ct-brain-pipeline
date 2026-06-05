#!/usr/bin/env python3
"""Derive per-class hemorrhage thresholds from RSNA out-of-fold predictions.

Reads the instrumented DenseNet121 training output (5 folds × best-epoch
val-prediction dumps) and computes Youden-optimal, F1-optimal and
sens@95-spec thresholds per class. Adds a 1000-resample bootstrap CI for
the Youden threshold so the report can show threshold stability.

This replaces the legacy approach where thresholds were fitted on CT-ICH
ground truth — that was test-set tuning and inflates the reported metrics.
The new thresholds use only RSNA validation predictions, which the model
has never seen during training (out-of-fold), so they can be frozen and
applied to CT-ICH / Kaggle / hospital data without leakage.

Input layout (the user assembles this from the hospital workstation):
  <input_dir>/fold0.npz       # contains outGT, outPRED, shape (~134k, 6)
  <input_dir>/fold0_best.json # {"epoch": N, "score": ..., "macro_AUC": ...}
  ...
  <input_dir>/fold4.npz
  <input_dir>/fold4_best.json

Class order in the npz: ["any", "epidural", "intraparenchymal",
                        "intraventricular", "subarachnoid", "subdural"].

Output:
  <output_path>  (default: ct-brain-pipeline/config/thresholds_rsna_val.json)

Usage:
  python scripts/derive_rsna_thresholds.py \
      --input-dir /path/to/dn121_val_preds \
      --output config/thresholds_rsna_val.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (precision_recall_curve, roc_auc_score, roc_curve,
                             average_precision_score)


HEMORRHAGE_CLASSES = ["any", "epidural", "intraparenchymal",
                      "intraventricular", "subarachnoid", "subdural"]


def find_youden_threshold(gt: np.ndarray, pred: np.ndarray) -> float:
    """Return the threshold maximising Youden's J = TPR - FPR."""
    fpr, tpr, thresholds = roc_curve(gt, pred)
    j = tpr - fpr
    return float(thresholds[int(np.argmax(j))])


def find_f1_threshold(gt: np.ndarray, pred: np.ndarray) -> float:
    """Return the threshold maximising F1 on the PR-curve."""
    precision, recall, thresholds = precision_recall_curve(gt, pred)
    # precision/recall have len(thresholds)+1; align by dropping last point.
    p, r, t = precision[:-1], recall[:-1], thresholds
    if len(t) == 0:
        return float("nan")
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[int(np.argmax(f1))])


def find_sens95_threshold(gt: np.ndarray, pred: np.ndarray) -> float:
    """Return the highest threshold that still achieves sensitivity ≥ 0.95.

    Picks the most specific operating point compatible with screening-style
    sensitivity requirements.
    """
    fpr, tpr, thresholds = roc_curve(gt, pred)
    valid = tpr >= 0.95
    if not valid.any():
        return float(thresholds[int(np.argmax(tpr))])
    idx = np.where(valid)[0]
    return float(thresholds[idx[int(np.argmax(thresholds[idx]))]])


def confusion_metrics(gt: np.ndarray, pred: np.ndarray, thr: float) -> dict:
    """Per-class binary metrics at a given threshold."""
    y = (pred >= thr).astype(np.int32)
    tp = int(((y == 1) & (gt == 1)).sum())
    fp = int(((y == 1) & (gt == 0)).sum())
    tn = int(((y == 0) & (gt == 0)).sum())
    fn = int(((y == 0) & (gt == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = (2 * ppv * sens / (ppv + sens)) if (ppv + sens) else 0.0
    return {
        "threshold": float(thr), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "sensitivity": sens, "specificity": spec, "PPV": ppv, "NPV": npv,
        "F1": f1, "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
    }


def bootstrap_youden_ci(gt: np.ndarray, pred: np.ndarray, n_resamples: int,
                        seed: int) -> tuple[float, float]:
    """95% percentile bootstrap CI for the Youden-optimal threshold."""
    rng = np.random.default_rng(seed)
    n = len(gt)
    samples = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        g, p = gt[idx], pred[idx]
        if g.sum() == 0 or g.sum() == n:
            samples[i] = np.nan
            continue
        samples[i] = find_youden_threshold(g, p)
    valid = samples[~np.isnan(samples)]
    if len(valid) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(valid, 2.5)), float(np.percentile(valid, 97.5))


def load_folds(input_dir: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load and concatenate val predictions from 5 folds.

    Returns (outGT, outPRED, per_fold_meta) where the first two are
    concatenated arrays of shape (sum_N, 6).
    """
    all_gt, all_pred, meta = [], [], []
    for f in range(5):
        npz_path = input_dir / f"fold{f}.npz"
        meta_path = input_dir / f"fold{f}_best.json"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing {npz_path}")
        with np.load(npz_path) as z:
            gt = z["outGT"]
            pred = z["outPRED"]
        # Some training runs write float32, others save with extra dims; squeeze.
        gt = np.asarray(gt, dtype=np.float32).reshape(-1, len(HEMORRHAGE_CLASSES))
        pred = np.asarray(pred, dtype=np.float32).reshape(-1, len(HEMORRHAGE_CLASSES))
        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch in {npz_path}: {gt.shape} vs {pred.shape}")
        fold_meta = {"fold": f, "n_samples": int(gt.shape[0])}
        if meta_path.exists():
            with open(meta_path) as mf:
                fold_meta.update(json.load(mf))
        all_gt.append(gt)
        all_pred.append(pred)
        meta.append(fold_meta)
        print(f"  fold {f}: {gt.shape[0]:,} slices "
              f"(best epoch {fold_meta.get('epoch', '?')})")
    return np.concatenate(all_gt, axis=0), np.concatenate(all_pred, axis=0), meta


def load_single_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Load a single npz (e.g., reconstructed ensemble OOF from one fold)."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing {npz_path}")
    with np.load(npz_path, allow_pickle=True) as z:
        gt = np.asarray(z["outGT"], dtype=np.float32).reshape(-1, len(HEMORRHAGE_CLASSES))
        pred = np.asarray(z["outPRED"], dtype=np.float32).reshape(-1, len(HEMORRHAGE_CLASSES))
    if gt.shape != pred.shape:
        raise ValueError(f"Shape mismatch in {npz_path}: {gt.shape} vs {pred.shape}")
    print(f"  loaded {gt.shape[0]:,} slices from {npz_path.name}")
    meta = [{"source": str(npz_path), "n_samples": int(gt.shape[0])}]
    return gt, pred, meta


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-dir", type=Path,
                       help="Directory with fold0..4.npz and fold0..4_best.json "
                            "(per-fold instrumented training output)")
    group.add_argument("--single-npz", type=Path,
                       help="Single npz with outGT + outPRED — use when you have "
                            "reconstructed ensemble OOF from one fold (see "
                            "reconstruct_ensemble_from_features.py)")
    ap.add_argument("--output", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "config" / "thresholds_rsna_val.json")
    ap.add_argument("--bootstrap-resamples", type=int, default=1000,
                    help="Bootstrap resamples for Youden CI (default 1000)")
    ap.add_argument("--seed", type=int, default=1992,
                    help="RNG seed for the bootstrap (default 1992 — matches SeuTao)")
    args = ap.parse_args()

    if args.input_dir is not None:
        print("Loading per-fold validation predictions ...")
        gt, pred, fold_meta = load_folds(args.input_dir)
        source_method = ("Youden's J on concatenated out-of-fold RSNA validation "
                         "predictions (DenseNet121, 5 folds, best epoch per fold).")
        n_folds = 5
    else:
        print(f"Loading single npz: {args.single_npz}")
        gt, pred, fold_meta = load_single_npz(args.single_npz)
        source_method = ("Youden's J on reconstructed 3-backbone ensemble OOF "
                         "predictions (DN121 + DN169 + SE-ResNeXt101, single "
                         "fold's val set, classifier head applied to saved CNN "
                         "features). See scripts/reconstruct_ensemble_from_features.py.")
        n_folds = 1
    n_total = gt.shape[0]
    print(f"Total out-of-fold predictions: {n_total:,} slices × {gt.shape[1]} classes")

    youden_thr, youden_ci, f1_thr, sens95_thr = {}, {}, {}, {}
    metrics_at_youden, metrics_at_f1 = {}, {}
    aucs, pr_aucs = {}, {}

    print(f"\nDeriving thresholds with {args.bootstrap_resamples} bootstrap resamples ...")
    for ci, cls in enumerate(HEMORRHAGE_CLASSES):
        g, p = gt[:, ci], pred[:, ci]
        n_pos, n_neg = int(g.sum()), int(len(g) - g.sum())

        if n_pos == 0 or n_neg == 0:
            print(f"  {cls:18s}  SKIPPED (n_pos={n_pos}, n_neg={n_neg})")
            youden_thr[cls] = float("nan")
            continue

        thr_y = find_youden_threshold(g, p)
        thr_f1 = find_f1_threshold(g, p)
        thr_s95 = find_sens95_threshold(g, p)
        auc = float(roc_auc_score(g, p))
        pr_auc = float(average_precision_score(g, p))

        ci_low, ci_high = bootstrap_youden_ci(g, p, args.bootstrap_resamples,
                                              args.seed + ci)

        youden_thr[cls] = thr_y
        youden_ci[cls] = [ci_low, ci_high]
        f1_thr[cls] = thr_f1
        sens95_thr[cls] = thr_s95
        aucs[cls] = auc
        pr_aucs[cls] = pr_auc
        metrics_at_youden[cls] = confusion_metrics(g, p, thr_y)
        metrics_at_f1[cls] = confusion_metrics(g, p, thr_f1)

        print(f"  {cls:18s}  Youden={thr_y:.4f}  CI=[{ci_low:.4f},{ci_high:.4f}]  "
              f"F1opt={thr_f1:.4f}  Sens95={thr_s95:.4f}  AUC={auc:.4f}  "
              f"n_pos={n_pos:,}")

    legacy = {"any": 0.3715, "epidural": 0.0247, "intraparenchymal": 0.1738,
              "intraventricular": 0.1018, "subarachnoid": 0.1967,
              "subdural": 0.2191}

    output_doc = {
        "method": source_method,
        "n_folds_concatenated": n_folds,
        "n_samples_total": int(n_total),
        "per_fold_meta": fold_meta,
        "bootstrap": {"resamples": args.bootstrap_resamples, "seed": args.seed,
                      "ci_level": 0.95},
        "class_order": HEMORRHAGE_CLASSES,
        "youden_thresholds": youden_thr,
        "youden_ci_95": youden_ci,
        "f1_thresholds": f1_thr,
        "sens95_thresholds": sens95_thr,
        "auc_per_class": aucs,
        "pr_auc_per_class": pr_aucs,
        "metrics_at_youden": metrics_at_youden,
        "metrics_at_f1": metrics_at_f1,
        "legacy_ctich_thresholds": legacy,
        "legacy_note": "The previous thresholds were fitted on CT-ICH ground "
                       "truth and applied to CT-ICH evaluation, which is "
                       "test-set tuning. These new thresholds replace them.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_doc, f, indent=2)
    print(f"\nWrote {args.output}")
    print("Update pipeline.py to load HEMORRHAGE_THRESHOLDS from this file "
          "and re-run evaluate_ctich.py / evaluate_kaggle.py.")


if __name__ == "__main__":
    main()
