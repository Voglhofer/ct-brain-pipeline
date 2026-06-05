#!/usr/bin/env python3
"""Parse SeuTao-format log.csv files from the March 2026 OLD training run.

The original SeuTao train.py writes a 12-column CSV with per-epoch train
loss and (every-5-epoch) val metrics for 5 folds concatenated in one file.
This script parses it into clean per-epoch DataFrames suitable for plotting
and per-fold summary tables for the thesis.

Input layout (the OLD reproduction):
  <root>/external/SeuTao_repo/2DNet/src/data_test/<backbone>/log.csv

Output (written next to this script's --output-dir):
  <out>/<backbone>_per_epoch.csv          # one row per (fold, epoch)
  <out>/all_backbones_per_epoch.csv       # concatenated across backbones
  <out>/best_epoch_per_fold.csv           # one row per (backbone, fold)
  <out>/training_summary.json             # cross-fold mean/std per backbone

CSV column structure (12 fields per epoch row in the SeuTao format):
  Epoch, LR, Time, TrainLoss, ValLoss,
  "auc:", "[6 floats]", "loss:", "[6 floats]", ValLoss

Val metrics are NaN except at epochs 0, 4, 9, 14, ..., 79 (every 5 epochs).

Usage:
  python scripts/parse_old_training_logs.py \\
      --reproduction-root /home/khma/bsc_hemorrage/rsna-seutao-reproduction \\
      --output-dir training_analysis/old_run
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


HEMORRHAGE_CLASSES = ["any", "epidural", "intraparenchymal",
                      "intraventricular", "subarachnoid", "subdural"]

BACKBONE_DIRS = {
    "DenseNet121": "DenseNet121_change_avg_256",
    "DenseNet169": "DenseNet169_change_avg_256",
    "SE-ResNeXt101": "se_resnext101_32x4d_256",
}


def parse_list(cell: str) -> list[float]:
    """Parse '[0.9, nan, 0.8]' into a list of floats. Returns 6 NaNs on failure."""
    s = cell.strip()
    if not s.startswith("[") or not s.endswith("]"):
        return [float("nan")] * 6
    inner = s[1:-1]
    out = []
    for item in inner.split(","):
        item = item.strip()
        if item.lower() == "nan" or item == "":
            out.append(float("nan"))
        else:
            try:
                out.append(float(item))
            except ValueError:
                out.append(float("nan"))
    while len(out) < 6:
        out.append(float("nan"))
    return out[:6]


def parse_log_csv(path: Path) -> pd.DataFrame:
    """Parse a SeuTao-format log.csv into a DataFrame.

    Returns columns:
      fold, epoch, lr, epoch_time_s, train_loss, val_loss,
      val_auc_<class> (6), val_loss_<class> (6).
    """
    rows = []
    current_fold = None
    with open(path) as f:
        reader = csv.reader(f)
        for fields in reader:
            if not fields:
                continue
            head = fields[0].strip()
            # Header row of the file.
            if head.lower() == "epoch":
                continue
            # Metadata lines accompanying each fold's start.
            if head.startswith("train dataset") or head.startswith("train_batch"):
                continue
            # Single-integer line = fold marker.
            if len(fields) == 1 and head.isdigit():
                current_fold = int(head)
                continue
            # Epoch row: must have ≥10 fields.
            if len(fields) < 10 or current_fold is None:
                continue
            try:
                epoch = int(fields[0])
                lr = float(fields[1])
                epoch_time = float(fields[2])
                train_loss = float(fields[3])
            except ValueError:
                continue
            try:
                val_loss = float(fields[4])
            except ValueError:
                val_loss = float("nan")
            aucs = parse_list(fields[6])
            losses = parse_list(fields[8])
            row = {
                "fold": current_fold, "epoch": epoch, "lr": lr,
                "epoch_time_s": epoch_time, "train_loss": train_loss,
                "val_loss": val_loss,
            }
            for c, a, l in zip(HEMORRHAGE_CLASSES, aucs, losses):
                row[f"val_auc_{c}"] = a
                row[f"val_loss_{c}"] = l
            rows.append(row)
    return pd.DataFrame(rows)


def best_epoch_per_fold(df: pd.DataFrame) -> pd.DataFrame:
    """For each fold, find the epoch with the highest val_auc_any."""
    out = []
    for fold, sub in df.groupby("fold"):
        valid = sub.dropna(subset=["val_auc_any"])
        if valid.empty:
            continue
        best = valid.loc[valid["val_auc_any"].idxmax()]
        rec = {"fold": fold, "best_epoch": int(best["epoch"]),
               "best_val_loss": float(best["val_loss"])}
        for c in HEMORRHAGE_CLASSES:
            rec[f"val_auc_{c}_at_best"] = float(best[f"val_auc_{c}"])
        rec["val_macro_auc_at_best"] = float(
            np.nanmean([best[f"val_auc_{c}"] for c in HEMORRHAGE_CLASSES]))
        rec["total_train_time_s"] = float(sub["epoch_time_s"].sum())
        rec["final_epoch"] = int(sub["epoch"].max())
        out.append(rec)
    return pd.DataFrame(out)


def cross_fold_summary(best_df: pd.DataFrame) -> dict:
    """Mean ± std across folds for the headline metrics."""
    if best_df.empty:
        return {}
    summary = {"n_folds": len(best_df)}
    for c in HEMORRHAGE_CLASSES:
        col = f"val_auc_{c}_at_best"
        summary[f"AUC_{c}_mean"] = float(best_df[col].mean())
        summary[f"AUC_{c}_std"] = float(best_df[col].std(ddof=1))
    summary["macro_AUC_mean"] = float(best_df["val_macro_auc_at_best"].mean())
    summary["macro_AUC_std"] = float(best_df["val_macro_auc_at_best"].std(ddof=1))
    summary["best_epochs"] = best_df["best_epoch"].tolist()
    summary["total_compute_hours"] = float(
        best_df["total_train_time_s"].sum() / 3600)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproduction-root", type=Path, required=True,
                    help="Path to the OLD rsna-seutao-reproduction directory")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to write the parsed CSVs and JSON summary")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_test = (args.reproduction_root.expanduser().resolve()
                 / "external" / "SeuTao_repo" / "2DNet" / "src" / "data_test")

    all_frames = []
    best_frames = []
    summary = {}

    for backbone, subdir in BACKBONE_DIRS.items():
        log_path = data_test / subdir / "log.csv"
        if not log_path.exists():
            print(f"[{backbone}] log.csv missing at {log_path} — skipped")
            continue
        print(f"[{backbone}] parsing {log_path} ...")
        df = parse_log_csv(log_path)
        if df.empty:
            print(f"[{backbone}] parsed 0 rows — skipped")
            continue
        df.insert(0, "backbone", backbone)
        per_path = args.output_dir / f"{backbone}_per_epoch.csv"
        df.to_csv(per_path, index=False)
        print(f"  wrote {per_path} ({len(df)} epoch rows, "
              f"{df['fold'].nunique()} folds)")
        all_frames.append(df)

        best = best_epoch_per_fold(df)
        best.insert(0, "backbone", backbone)
        best_frames.append(best)
        summary[backbone] = cross_fold_summary(best)
        print(f"  cross-fold val_auc_any (mean ± std): "
              f"{summary[backbone]['AUC_any_mean']:.4f} ± "
              f"{summary[backbone]['AUC_any_std']:.4f} "
              f"(n_folds={summary[backbone]['n_folds']})")

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        combined.to_csv(args.output_dir / "all_backbones_per_epoch.csv",
                        index=False)
        print(f"\nWrote {args.output_dir / 'all_backbones_per_epoch.csv'} "
              f"({len(combined)} rows)")
    if best_frames:
        best_combined = pd.concat(best_frames, ignore_index=True)
        best_combined.to_csv(args.output_dir / "best_epoch_per_fold.csv",
                             index=False)
        print(f"Wrote {args.output_dir / 'best_epoch_per_fold.csv'}")
    with open(args.output_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
