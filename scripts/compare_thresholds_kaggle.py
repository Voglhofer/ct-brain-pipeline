#!/usr/bin/env python3
"""Compare legacy vs RSNA-derived thresholds on cached Kaggle predictions.

The Kaggle Brain Stroke CT dataset only labels slices as Bleeding /
Ischemia / Normal — no per-subtype ground truth. So we collapse the 6
hemorrhage predictions to a single "any hemorrhage" decision per slice
using the Youden threshold for the `any` class, then evaluate against
the binary `gt_hemorrhage` label.

Reads the cached predictions CSV (output_kaggle/predictions.csv or
output_kaggle_3slice/predictions.csv) so no model re-run is needed.

Usage:
  python scripts/compare_thresholds_kaggle.py
  python scripts/compare_thresholds_kaggle.py --predictions output_kaggle_3slice/predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


CLASSES = ["any", "epidural", "intraparenchymal", "intraventricular",
           "subarachnoid", "subdural"]
LEGACY = {"any": 0.3715, "epidural": 0.0247, "intraparenchymal": 0.1738,
          "intraventricular": 0.1018, "subarachnoid": 0.1967,
          "subdural": 0.2191}


def metrics(tp, fp, tn, fn):
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * ppv * sens / (ppv + sens)) if (ppv + sens) else 0.0
    return dict(TP=tp, FP=fp, TN=tn, FN=fn,
                sens=sens, spec=spec, PPV=ppv, F1=f1)


def evaluate(rows, thr_any):
    """Apply Youden 'any' threshold to p_any column and compare with
    binary Bleeding label."""
    tp = fp = tn = fn = 0
    for r in rows:
        g = int(r["gt_hemorrhage"])
        p = float(r["p_any"])
        y = 1 if p >= thr_any else 0
        if g and y: tp += 1
        elif g: fn += 1
        elif y: fp += 1
        else: tn += 1
    return metrics(tp, fp, tn, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "output_kaggle" / "predictions.csv")
    ap.add_argument("--new-thresholds", type=Path,
                    default=Path(__file__).resolve().parents[1]
                    / "config" / "thresholds_rsna_val.json")
    args = ap.parse_args()

    with open(args.new_thresholds) as f:
        new_doc = json.load(f)
        new_any = new_doc["youden_thresholds"]["any"]
        ci = new_doc.get("youden_ci_95", {}).get("any", [float("nan")] * 2)

    print(f"Loading {args.predictions} ...")
    with open(args.predictions) as f:
        # The CSV has a duplicate "p_any" column (header line shows two p_any)
        # — csv.DictReader keeps the last value under that key, which is the
        # second p_any. Inspect to confirm: both p_any columns hold the same
        # float in our generated files. We rely on DictReader's behaviour.
        rows = list(csv.DictReader(f))

    # Keep only slices that are part of the labelled dataset
    # (gt_hemorrhage column is empty/missing for unlabelled rows).
    labelled = [r for r in rows if r.get("gt_hemorrhage") not in (None, "")]
    print(f"  total rows: {len(rows):,}, labelled: {len(labelled):,}")

    L = evaluate(labelled, LEGACY["any"])
    N = evaluate(labelled, new_any)

    print(f"\nThresholds for 'any':")
    print(f"  legacy (CT-ICH-tuned): {LEGACY['any']:.4f}")
    print(f"  new (RSNA val):        {new_any:.4f}  CI=[{ci[0]:.4f},{ci[1]:.4f}]")

    print(f"\nKaggle metrics (binary 'any hemorrhage' on {len(labelled):,} labelled slices):")
    print(f"  {'metric':<10} {'legacy':>10} {'new':>10}")
    print("  " + "-" * 34)
    for k in ("sens", "spec", "PPV", "F1"):
        print(f"  {k:<10} {L[k]:>10.3f} {N[k]:>10.3f}")
    print(f"  {'TP':<10} {L['TP']:>10} {N['TP']:>10}")
    print(f"  {'FP':<10} {L['FP']:>10} {N['FP']:>10}")
    print(f"  {'TN':<10} {L['TN']:>10} {N['TN']:>10}")
    print(f"  {'FN':<10} {L['FN']:>10} {N['FN']:>10}")

    out_path = args.new_thresholds.parent / f"kaggle_threshold_comparison_{args.predictions.parent.name}.json"
    with open(out_path, "w") as f:
        json.dump({"n_labelled": len(labelled),
                   "predictions_file": str(args.predictions),
                   "legacy_threshold": LEGACY["any"],
                   "new_threshold": new_any,
                   "new_threshold_ci_95": ci,
                   "legacy_metrics": L,
                   "new_metrics": N}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
