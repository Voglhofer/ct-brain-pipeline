#!/usr/bin/env python3
"""Compare legacy vs RSNA-derived thresholds on CT-ICH slice predictions.

Reads the cached CT-ICH predictions (Bachelor_RSNA_test/ct_ich_results_full/
all_predictions.csv) and applies both threshold sets, printing a side-by-side
table per class so you can drop the numbers straight into the report.

This does NOT re-run the model — it operates on the predictions that were
already computed. For Kaggle / hospital, re-run evaluate_kaggle.py /
evaluate_ctich.py (they will pick up the new thresholds via pipeline.py).

Usage:
  python scripts/compare_thresholds.py
  python scripts/compare_thresholds.py \
      --predictions /Users/.../all_predictions.csv \
      --new-thresholds config/thresholds_rsna_val.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CLASSES = ["any", "epidural", "intraparenchymal", "intraventricular",
           "subarachnoid", "subdural"]


def confusion(gt: np.ndarray, pred: np.ndarray, thr: float) -> dict:
    y = (pred >= thr).astype(np.int32)
    g = gt.astype(np.int32)
    tp = int(((y == 1) & (g == 1)).sum())
    fp = int(((y == 1) & (g == 0)).sum())
    tn = int(((y == 0) & (g == 0)).sum())
    fn = int(((y == 0) & (g == 1)).sum())
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * ppv * sens / (ppv + sens)) if (ppv + sens) else 0.0
    return dict(TP=tp, FP=fp, TN=tn, FN=fn, sens=sens, spec=spec, PPV=ppv, F1=f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path,
                    default=Path("/Users/josefinevoglhofer/Documents/DTU/"
                                 "Bachelor_RSNA_test/ct_ich_results_full/"
                                 "all_predictions.csv"))
    ap.add_argument("--new-thresholds", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "config" / "thresholds_rsna_val.json")
    ap.add_argument("--legacy-thresholds", type=Path,
                    default=Path("/Users/josefinevoglhofer/Documents/DTU/"
                                 "Bachelor_RSNA_test/ct_ich_results_full/"
                                 "metrics_optimal_threshold.json"))
    args = ap.parse_args()

    with open(args.legacy_thresholds) as f:
        legacy = json.load(f)["thresholds"]
    with open(args.new_thresholds) as f:
        new_doc = json.load(f)
        new = new_doc["youden_thresholds"]
        new_ci = new_doc.get("youden_ci_95", {})

    print(f"Loading CT-ICH predictions from {args.predictions} ...")
    gt = {c: [] for c in CLASSES}
    pred = {c: [] for c in CLASSES}
    with open(args.predictions) as f:
        for row in csv.DictReader(f):
            for c in CLASSES:
                gt[c].append(int(row[f"gt_{c}"]))
                pred[c].append(float(row[f"pred_{c}"]))
    gt = {c: np.asarray(v, dtype=np.int32) for c, v in gt.items()}
    pred = {c: np.asarray(v, dtype=np.float32) for c, v in pred.items()}
    n = len(gt["any"])
    print(f"  {n:,} slices loaded")

    print("\nThresholds:")
    print(f"  {'class':<18} {'legacy (CT-ICH)':<18} {'new (RSNA val)':<28}")
    for c in CLASSES:
        ci = new_ci.get(c, [float('nan'), float('nan')])
        print(f"  {c:<18} {legacy[c]:<18.4f} "
              f"{new[c]:.4f}  CI=[{ci[0]:.4f},{ci[1]:.4f}]")

    print("\nCT-ICH metrics at each threshold:")
    print(f"  {'class':<18} | {'legacy sens':>11} {'spec':>6} {'F1':>6} "
          f"{'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5} | "
          f"{'new sens':>9} {'spec':>6} {'F1':>6} "
          f"{'TP':>5} {'FP':>5} {'TN':>5} {'FN':>5}")
    print("  " + "-" * 142)
    rows_out = {}
    for c in CLASSES:
        L = confusion(gt[c], pred[c], legacy[c])
        N = confusion(gt[c], pred[c], new[c])
        print(f"  {c:<18} | "
              f"{L['sens']:>11.3f} {L['spec']:>6.3f} {L['F1']:>6.3f} "
              f"{L['TP']:>5} {L['FP']:>5} {L['TN']:>5} {L['FN']:>5} | "
              f"{N['sens']:>9.3f} {N['spec']:>6.3f} {N['F1']:>6.3f} "
              f"{N['TP']:>5} {N['FP']:>5} {N['TN']:>5} {N['FN']:>5}")
        rows_out[c] = {"legacy": L, "new": N,
                       "legacy_threshold": legacy[c], "new_threshold": new[c]}

    out = args.new_thresholds.parent / "ctich_threshold_comparison.json"
    with open(out, "w") as f:
        json.dump({"n_slices": n, "per_class": rows_out}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
