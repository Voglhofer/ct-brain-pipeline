#!/usr/bin/env python3
"""Generate fold-1 validation diagnostics from the reconstructed predictions.

Reads /tmp/ensemble_oof_fold1.npz (produced by
reconstruct_ensemble_from_features.py) which contains:
  outGT, outPRED, DN121_PRED, DN169_PRED, SERES_PRED, slice_ids

Writes a set of validation-quality plots for the OLD-training fold 1 to
the thesis: ROC, PR, calibration, score histograms, confusion matrices at
Youden-optimal threshold, per-class threshold comparison.

These are evaluated on 134k held-out (out-of-fold) RSNA slices that none
of the three models saw during training — so they are honest val metrics
for the deployed 3-backbone ensemble.

Usage:
  python scripts/plot_fold1_validation_metrics.py \\
      --npz data/ensemble_oof_fold1.npz \\
      --output-dir training_analysis/old_run/fold1_val
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score, precision_recall_curve,
                             average_precision_score, confusion_matrix)
from sklearn.calibration import calibration_curve


HEMORRHAGE_CLASSES = ["any", "epidural", "intraparenchymal",
                      "intraventricular", "subarachnoid", "subdural"]
CLASS_LABELS = {
    "any": "Any", "epidural": "Epidural", "intraparenchymal": "Intraparenchymal",
    "intraventricular": "Intraventricular", "subarachnoid": "Subarachnoid",
    "subdural": "Subdural",
}
COLORS = {"DenseNet121": "tab:blue", "DenseNet169": "tab:orange",
          "SE-ResNeXt101": "tab:green", "Ensemble": "tab:red"}


def youden_threshold(y: np.ndarray, p: np.ndarray):
    fpr, tpr, thr = roc_curve(y, p)
    idx = int(np.argmax(tpr - fpr))
    return float(thr[idx]), float(tpr[idx]), float(1 - fpr[idx])


def plot_roc_per_class(preds: dict, gt: np.ndarray, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (ci, cls) in zip(axes.ravel(), enumerate(HEMORRHAGE_CLASSES)):
        for name, p in preds.items():
            if p is None: continue
            try:
                fpr, tpr, _ = roc_curve(gt[:, ci], p[:, ci])
                auc = roc_auc_score(gt[:, ci], p[:, ci])
            except ValueError:
                continue
            ax.plot(fpr, tpr, color=COLORS.get(name, "gray"),
                    label=f"{name} (AUC={auc:.3f})", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.7, alpha=0.5)
        ax.set_title(CLASS_LABELS[cls])
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle("ROC curves — fold 1 validation (134,946 slices)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_pr_per_class(preds: dict, gt: np.ndarray, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (ci, cls) in zip(axes.ravel(), enumerate(HEMORRHAGE_CLASSES)):
        for name, p in preds.items():
            if p is None: continue
            try:
                prec, rec, _ = precision_recall_curve(gt[:, ci], p[:, ci])
                ap = average_precision_score(gt[:, ci], p[:, ci])
            except ValueError:
                continue
            ax.plot(rec, prec, color=COLORS.get(name, "gray"),
                    label=f"{name} (AP={ap:.3f})", linewidth=2)
        ax.set_title(CLASS_LABELS[cls])
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8); ax.set_ylim(0, 1.02)
    fig.suptitle("Precision-recall curves — fold 1 validation", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_calibration_per_class(preds: dict, gt: np.ndarray, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (ci, cls) in zip(axes.ravel(), enumerate(HEMORRHAGE_CLASSES)):
        for name, p in preds.items():
            if p is None: continue
            try:
                prob_true, prob_pred = calibration_curve(
                    gt[:, ci], p[:, ci], n_bins=10, strategy="quantile")
            except Exception:
                continue
            ax.plot(prob_pred, prob_true, "o-", color=COLORS.get(name, "gray"),
                    label=name, linewidth=2, markersize=5)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.7, alpha=0.5)
        ax.set_title(CLASS_LABELS[cls])
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed positive rate")
        ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=8)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.suptitle("Calibration curves — fold 1 validation", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_score_histograms(preds: dict, gt: np.ndarray, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    ens = preds.get("Ensemble")
    if ens is None:
        plt.close(fig); return
    for ax, (ci, cls) in zip(axes.ravel(), enumerate(HEMORRHAGE_CLASSES)):
        neg = ens[gt[:, ci] == 0, ci]
        pos = ens[gt[:, ci] == 1, ci]
        ax.hist(neg, bins=50, alpha=0.5, density=True, label=f"Negative (n={len(neg):,})",
                color="tab:blue")
        ax.hist(pos, bins=50, alpha=0.5, density=True, label=f"Positive (n={len(pos):,})",
                color="tab:red")
        thr, _, _ = youden_threshold(gt[:, ci], ens[:, ci])
        ax.axvline(thr, color="black", linestyle="--",
                   label=f"Youden thr = {thr:.3f}")
        ax.set_title(CLASS_LABELS[cls]); ax.set_xlabel("Ensemble probability")
        ax.set_ylabel("Density"); ax.legend(loc="best", fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Ensemble score distributions by ground truth — fold 1",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_confusion_matrices(ens: np.ndarray, gt: np.ndarray, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    for ax, (ci, cls) in zip(axes.ravel(), enumerate(HEMORRHAGE_CLASSES)):
        thr, _, _ = youden_threshold(gt[:, ci], ens[:, ci])
        pred = (ens[:, ci] >= thr).astype(int)
        cm = confusion_matrix(gt[:, ci], pred, labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=9)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred -", "Pred +"]); ax.set_yticklabels(["True -", "True +"])
        ax.set_title(f"{CLASS_LABELS[cls]} (thr={thr:.3f})")
    fig.suptitle("Confusion matrices at Youden thresholds (ensemble, fold 1)",
                 fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_metrics_table(preds: dict, gt: np.ndarray, out_csv: Path,
                        out_tex: Path):
    """One row per backbone × class with AUC, AP, Youden thr, sens, spec, F1."""
    rows = []
    for name, p in preds.items():
        if p is None: continue
        for ci, cls in enumerate(HEMORRHAGE_CLASSES):
            try:
                auc = roc_auc_score(gt[:, ci], p[:, ci])
                ap = average_precision_score(gt[:, ci], p[:, ci])
                thr, sens, spec = youden_threshold(gt[:, ci], p[:, ci])
                ybin = (p[:, ci] >= thr).astype(int)
                tp = int(((ybin == 1) & (gt[:, ci] == 1)).sum())
                fp = int(((ybin == 1) & (gt[:, ci] == 0)).sum())
                fn = int(((ybin == 0) & (gt[:, ci] == 1)).sum())
                ppv = tp / max(tp + fp, 1)
                f1 = 2 * ppv * sens / max(ppv + sens, 1e-9)
            except Exception:
                continue
            rows.append(dict(model=name, klass=cls, AUC=auc, AP=ap,
                             youden_thr=thr, sensitivity=sens, specificity=spec,
                             PPV=ppv, F1=f1))
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)

    # LaTeX table per backbone
    lines = ["\\begin{table}[t]\\centering\\small",
             "\\caption{Fold-1 validation metrics (Youden thresholds, "
             "134,946 slices).}",
             "\\label{tab:fold1_val}",
             "\\begin{tabular}{llccccc}",
             "\\toprule",
             "Model & Class & AUC & AP & Sens. & Spec. & F1 \\\\",
             "\\midrule"]
    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    for model in by_model:
        for r in by_model[model]:
            lines.append(f"{model} & {CLASS_LABELS[r['klass']]} & "
                         f"{r['AUC']:.3f} & {r['AP']:.3f} & "
                         f"{r['sensitivity']:.3f} & {r['specificity']:.3f} & "
                         f"{r['F1']:.3f} \\\\")
        lines.append("\\midrule")
    if lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}\\end{table}"]
    out_tex.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, required=True,
                    help="Output of reconstruct_ensemble_from_features.py")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.npz, allow_pickle=True) as z:
        gt = z["outGT"]
        ens = z["outPRED"]
        dn121 = z.get("DN121_PRED")
        dn169 = z.get("DN169_PRED")
        seres = z.get("SERES_PRED")
    preds = {"DenseNet121": dn121, "DenseNet169": dn169,
             "SE-ResNeXt101": seres, "Ensemble": ens}
    print(f"Loaded {gt.shape[0]:,} slices × {gt.shape[1]} classes from {args.npz}")
    print(f"Per-class positive count: "
          + ", ".join(f"{c}={int(gt[:, i].sum())}"
                      for i, c in enumerate(HEMORRHAGE_CLASSES)))
    print()

    plot_roc_per_class(preds, gt, args.output_dir / "roc_per_class.png")
    plot_pr_per_class(preds, gt, args.output_dir / "pr_per_class.png")
    plot_calibration_per_class(preds, gt, args.output_dir / "calibration_per_class.png")
    plot_score_histograms(preds, gt, args.output_dir / "score_histograms.png")
    plot_confusion_matrices(ens, gt, args.output_dir / "confusion_matrices.png")
    write_metrics_table(preds, gt,
                        args.output_dir / "fold1_val_metrics.csv",
                        args.output_dir / "fold1_val_metrics.tex")
    print(f"Wrote 5 PNG plots + CSV + LaTeX table to {args.output_dir}")


if __name__ == "__main__":
    main()
