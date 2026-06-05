#!/usr/bin/env python3
"""Generate thesis-ready training curve plots from the parsed OLD training logs.

Reads the CSVs produced by parse_old_training_logs.py and writes a set of
PNG plots + a LaTeX-ready summary table.

Plots (saved to --output-dir/plots/):
  1. learning_curves_<backbone>.png    train+val loss vs epoch, all 5 folds
  2. auc_curves_<backbone>.png         per-class val AUC vs epoch, mean ± std across folds
  3. backbone_comparison_loss.png      val loss vs epoch, mean across folds for each backbone
  4. backbone_comparison_auc_any.png   val_auc_any vs epoch, mean across folds for each backbone
  5. best_auc_per_fold_per_backbone.png  bar chart of final val AUC

Tables:
  best_metrics_table.tex   LaTeX longtable: backbone × class → mean ± std AUC
  convergence_table.tex    when each fold converged + total training time

Usage:
  python scripts/plot_old_training_curves.py \\
      --input-dir training_analysis/old_run \\
      --output-dir training_analysis/old_run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HEMORRHAGE_CLASSES = ["any", "epidural", "intraparenchymal",
                      "intraventricular", "subarachnoid", "subdural"]
CLASS_LABELS = {
    "any": "Any", "epidural": "Epidural", "intraparenchymal": "Intraparenchymal",
    "intraventricular": "Intraventricular", "subarachnoid": "Subarachnoid",
    "subdural": "Subdural",
}
BACKBONE_COLORS = {
    "DenseNet121": "tab:blue",
    "DenseNet169": "tab:orange",
    "SE-ResNeXt101": "tab:green",
}


def plot_learning_curves_per_backbone(df_bb: pd.DataFrame, backbone: str,
                                      out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for fold in sorted(df_bb["fold"].unique()):
        sub = df_bb[df_bb["fold"] == fold]
        axes[0].plot(sub["epoch"], sub["train_loss"], alpha=0.7,
                     label=f"fold {fold}")
        val = sub.dropna(subset=["val_loss"])
        axes[1].plot(val["epoch"], val["val_loss"], "o-", alpha=0.7,
                     label=f"fold {fold}")
    axes[0].set_title(f"{backbone}: training loss per fold")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train loss"); axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    axes[1].set_title(f"{backbone}: validation loss per fold (every 5 epochs)")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Val loss"); axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_auc_curves_per_backbone(df_bb: pd.DataFrame, backbone: str,
                                 out_path: Path):
    val = df_bb.dropna(subset=["val_auc_any"])
    if val.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = sorted(val["epoch"].unique())
    for cls in HEMORRHAGE_CLASSES:
        col = f"val_auc_{cls}"
        means, stds = [], []
        for e in epochs:
            vals = val.loc[val["epoch"] == e, col].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        means = np.asarray(means); stds = np.asarray(stds)
        ax.plot(epochs, means, "o-", label=CLASS_LABELS[cls], linewidth=1.5)
        ax.fill_between(epochs, means - stds, means + stds, alpha=0.15)
    ax.set_title(f"{backbone}: validation AUC per class (mean ± std across 5 folds)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation AUC"); ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, ncol=2)
    ax.set_ylim(0.5, 1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_backbone_comparison(all_df: pd.DataFrame, metric_col: str,
                             ylabel: str, title: str, out_path: Path,
                             every_5_only: bool):
    fig, ax = plt.subplots(figsize=(10, 6))
    for bb, color in BACKBONE_COLORS.items():
        sub = all_df[all_df["backbone"] == bb]
        if every_5_only:
            sub = sub.dropna(subset=[metric_col])
        if sub.empty:
            continue
        epochs = sorted(sub["epoch"].unique())
        means, stds = [], []
        for e in epochs:
            vals = sub.loc[sub["epoch"] == e, metric_col].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            stds.append(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        means = np.asarray(means); stds = np.asarray(stds)
        ax.plot(epochs, means, "o-" if every_5_only else "-",
                color=color, label=bb, linewidth=2)
        ax.fill_between(epochs, means - stds, means + stds,
                        alpha=0.2, color=color)
    ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3); ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_best_auc_bar(best_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    backbones = sorted(best_df["backbone"].unique())
    classes = HEMORRHAGE_CLASSES
    width = 0.8 / len(backbones)
    x = np.arange(len(classes))
    for i, bb in enumerate(backbones):
        sub = best_df[best_df["backbone"] == bb]
        means = [sub[f"val_auc_{c}_at_best"].mean() for c in classes]
        stds = [sub[f"val_auc_{c}_at_best"].std(ddof=1) for c in classes]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=4,
               label=bb, color=BACKBONE_COLORS.get(bb, "gray"))
    ax.set_xticks(x + width * (len(backbones) - 1) / 2)
    ax.set_xticklabels([CLASS_LABELS[c] for c in classes], rotation=20)
    ax.set_ylabel("Best validation AUC (mean ± std across 5 folds)")
    ax.set_title("Per-class best validation AUC by backbone")
    ax.set_ylim(0.85, 1.0)
    ax.grid(alpha=0.3, axis="y"); ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_latex_best_metrics(best_df: pd.DataFrame, out_path: Path):
    """LaTeX table: rows = backbones, columns = per-class AUC (mean ± std)."""
    rows = []
    for bb in sorted(best_df["backbone"].unique()):
        sub = best_df[best_df["backbone"] == bb]
        cells = [bb]
        for c in HEMORRHAGE_CLASSES + ["macro"]:
            col = f"val_auc_{c}_at_best" if c != "macro" else "val_macro_auc_at_best"
            m, s = sub[col].mean(), sub[col].std(ddof=1)
            cells.append(f"{m:.3f} $\\pm$ {s:.3f}")
        rows.append(" & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    header = ("\\begin{table}[t]\n\\centering\n\\small\n"
              "\\caption{Per-class validation AUC at best epoch, mean $\\pm$ "
              "std across 5 folds.}\n\\label{tab:training_auc}\n"
              "\\begin{tabular}{l" + "c" * (len(HEMORRHAGE_CLASSES) + 1) + "}\n"
              "\\toprule\nBackbone & "
              + " & ".join(CLASS_LABELS[c] for c in HEMORRHAGE_CLASSES)
              + " & Macro \\\\\n\\midrule\n")
    footer = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    out_path.write_text(header + body + footer)


def write_latex_convergence(best_df: pd.DataFrame, out_path: Path):
    rows = []
    for bb in sorted(best_df["backbone"].unique()):
        sub = best_df[best_df["backbone"] == bb]
        cells = [bb,
                 f"{sub['best_epoch'].mean():.1f} $\\pm$ {sub['best_epoch'].std(ddof=1):.1f}",
                 f"{sub['total_train_time_s'].sum() / 3600:.1f}",
                 f"{sub['final_epoch'].max() + 1}"]
        rows.append(" & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    header = ("\\begin{table}[t]\n\\centering\n\\small\n"
              "\\caption{Convergence statistics across folds.}\n"
              "\\label{tab:convergence}\n"
              "\\begin{tabular}{lccc}\n\\toprule\n"
              "Backbone & Best epoch (mean $\\pm$ std) & Total compute (h) & Epochs trained \\\\\n"
              "\\midrule\n")
    footer = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    out_path.write_text(header + body + footer)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    plot_dir = args.output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    all_df = pd.read_csv(args.input_dir / "all_backbones_per_epoch.csv")
    best_df = pd.read_csv(args.input_dir / "best_epoch_per_fold.csv")

    for backbone in sorted(all_df["backbone"].unique()):
        df_bb = all_df[all_df["backbone"] == backbone]
        plot_learning_curves_per_backbone(
            df_bb, backbone, plot_dir / f"learning_curves_{backbone}.png")
        plot_auc_curves_per_backbone(
            df_bb, backbone, plot_dir / f"auc_curves_{backbone}.png")
        print(f"  plotted {backbone} ({len(df_bb)} epoch rows)")

    plot_backbone_comparison(
        all_df, "val_loss", "Validation loss",
        "Validation loss vs epoch, mean ± std across folds",
        plot_dir / "backbone_comparison_loss.png", every_5_only=True)
    plot_backbone_comparison(
        all_df, "val_auc_any", "Validation AUC (any hemorrhage)",
        "Val AUC (any) vs epoch, mean ± std across folds",
        plot_dir / "backbone_comparison_auc_any.png", every_5_only=True)
    plot_best_auc_bar(best_df, plot_dir / "best_auc_per_fold_per_backbone.png")
    print(f"\nWrote 5 comparison plots to {plot_dir}")

    write_latex_best_metrics(best_df, args.output_dir / "best_metrics_table.tex")
    write_latex_convergence(best_df, args.output_dir / "convergence_table.tex")
    print(f"Wrote LaTeX tables: best_metrics_table.tex, convergence_table.tex")


if __name__ == "__main__":
    main()
