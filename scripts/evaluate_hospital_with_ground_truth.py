#!/usr/bin/env python3
"""Evaluate the hospital cohort against clinical ground truth at three
operating points: Youden, F1-optimal, and the legacy CT-ICH-tuned threshold.

Reads every patient's results.json (the original cached probabilities)
and applies all three threshold sets to the max-probability scan-level
rule (patient positive if any slice's p_any exceeds the threshold).
Reports sensitivity / specificity / PPV / NPV / F1 / accuracy against
the clinician-supplied ground truth.

Subtype-level metrics are skipped because the supplied ground truth is
binary at the patient level only (hemorrhage yes/no).

Usage:
  python scripts/evaluate_hospital_with_ground_truth.py \\
      --results-root /path/to/resultater \\
      --ground-truth data/hospital_ground_truth.csv \\
      --thresholds config/thresholds_rsna_val.json \\
      --output-dir training_analysis/hospital_eval
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def confusion(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Binary confusion + headline metrics."""
    tp = int(((pred == 1) & (gt == 1)).sum())
    fp = int(((pred == 1) & (gt == 0)).sum())
    tn = int(((pred == 0) & (gt == 0)).sum())
    fn = int(((pred == 0) & (gt == 1)).sum())
    n = tp + fp + tn + fn
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = (2 * ppv * sens / (ppv + sens)) if (ppv + sens) else 0.0
    acc = (tp + tn) / n if n else 0.0
    return dict(TP=tp, FP=fp, TN=tn, FN=fn,
                sensitivity=sens, specificity=spec,
                PPV=ppv, NPV=npv, F1=f1, accuracy=acc, n=n)


def load_ground_truth(path: Path) -> dict[int, int]:
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[int(row["patient_id"])] = int(row["gt_hemorrhage"])
    return out


def load_max_p_any(results_root: Path) -> dict[int, float]:
    """For every patient with a results.json, return max p_any across slices."""
    out = {}
    for pdir in sorted(results_root.iterdir(),
                       key=lambda p: int(p.name) if p.name.isdigit() else p.name):
        if not pdir.is_dir():
            continue
        rj = pdir / "results.json"
        if not rj.exists():
            continue
        try:
            pid = int(pdir.name)
        except ValueError:
            continue
        with open(rj) as f:
            data = json.load(f)
        probs = [s["results"]["hemorrhage"]["any"]["probability"]
                 for s in data.get("slices", [])]
        out[pid] = max(probs) if probs else 0.0
    return out


def plot_threshold_comparison(metrics_by_strategy: dict, out_path: Path):
    metrics_to_plot = ["sensitivity", "specificity", "PPV", "F1", "accuracy"]
    strategies = list(metrics_by_strategy.keys())
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(metrics_to_plot))
    width = 0.8 / len(strategies)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, name in enumerate(strategies):
        vals = [metrics_by_strategy[name][m] for m in metrics_to_plot]
        ax.bar(x + i * width, vals, width,
               label=f"{name} (thr={metrics_by_strategy[name]['threshold']:.3f})",
               color=colors[i % len(colors)])
    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels([m.title() for m in metrics_to_plot])
    ax.set_ylabel("Metric value")
    ax.set_title("Hospital cohort: scan-level 'any hemorrhage' detection")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_score_distribution(max_p: dict, gt: dict, thresholds: dict,
                            out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 5))
    pids = sorted(set(max_p) & set(gt))
    scores = [max_p[p] for p in pids]
    labels = [gt[p] for p in pids]
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    ax.eventplot([pos, neg], orientation="vertical", linelengths=0.7,
                 linewidths=2, colors=["#d62728", "#1f77b4"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"GT positive (n={len(pos)})",
                        f"GT negative (n={len(neg)})"])
    ax.set_ylabel("Max p_any across slices")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3, axis="y")
    for name, (thr, col) in thresholds.items():
        ax.axhline(thr, linestyle="--", color=col, linewidth=1.5,
                   label=f"{name} thr = {thr:.3f}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Per-patient max p_any vs ground truth, with operating-point thresholds")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def write_latex_table(metrics_by_strategy: dict, out_path: Path):
    rows = []
    for name, m in metrics_by_strategy.items():
        rows.append(f"{name} & {m['threshold']:.4f} & "
                    f"{m['sensitivity']:.3f} & {m['specificity']:.3f} & "
                    f"{m['PPV']:.3f} & {m['F1']:.3f} & {m['accuracy']:.3f} & "
                    f"{m['TP']} & {m['FP']} & {m['TN']} & {m['FN']} \\\\")
    body = "\n".join(rows)
    text = ("\\begin{table}[t]\\centering\\small\n"
            "\\caption{Hospital cohort evaluation against clinical ground "
            "truth (n=%d). Scan-level decision: patient positive if any "
            "slice's $p_\\text{any}$ exceeds the threshold.}\n" % m["n"]
            + "\\label{tab:hospital_eval}\n"
            "\\begin{tabular}{lcccccccccc}\n\\toprule\n"
            "Strategy & Thr. & Sens. & Spec. & PPV & F1 & Acc. & TP & FP & TN & FN \\\\\n"
            "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n\\end{table}")
    out_path.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", type=Path, required=True,
                    help="Directory with one folder per patient containing results.json")
    ap.add_argument("--ground-truth", type=Path, required=True)
    ap.add_argument("--thresholds", type=Path, required=True,
                    help="Path to thresholds_rsna_val.json")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(args.ground_truth)
    max_p = load_max_p_any(args.results_root)
    shared = sorted(set(gt) & set(max_p))
    missing_gt = sorted(set(max_p) - set(gt))
    missing_pred = sorted(set(gt) - set(max_p))
    print(f"Patients with predictions: {len(max_p)}")
    print(f"Patients with ground truth: {len(gt)}")
    print(f"Patients in both (evaluated): {len(shared)}")
    if missing_gt:
        print(f"  No ground truth (excluded): {missing_gt}")
    if missing_pred:
        print(f"  No prediction folder: {missing_pred}")

    gt_arr = np.asarray([gt[p] for p in shared], dtype=np.int32)
    p_arr = np.asarray([max_p[p] for p in shared], dtype=np.float32)
    print(f"\nPositive prevalence in evaluated cohort: "
          f"{int(gt_arr.sum())}/{len(gt_arr)} "
          f"({gt_arr.mean() * 100:.1f}%)")

    with open(args.thresholds) as f:
        thr_doc = json.load(f)
    legacy = thr_doc["legacy_ctich_thresholds"]["any"]
    youden = thr_doc["youden_thresholds"]["any"]
    f1opt = thr_doc["f1_thresholds"]["any"]
    sens95 = thr_doc["sens95_thresholds"]["any"]
    print(f"\nThresholds for 'any':")
    print(f"  Legacy (CT-ICH tuned): {legacy:.4f}")
    print(f"  Youden (RSNA OOF):     {youden:.4f}")
    print(f"  F1-optimal (RSNA OOF): {f1opt:.4f}")
    print(f"  Sens95 (RSNA OOF):     {sens95:.4f}")

    strategies = {
        "Legacy (CT-ICH)": legacy,
        "Youden (RSNA val)": youden,
        "F1 (RSNA val)": f1opt,
        "Sens95 (RSNA val)": sens95,
    }
    metrics_by_strategy = {}
    for name, thr in strategies.items():
        pred = (p_arr >= thr).astype(np.int32)
        m = confusion(gt_arr, pred)
        m["threshold"] = float(thr)
        metrics_by_strategy[name] = m

    print(f"\n{'Strategy':<22} {'Thr':>7} "
          f"{'Sens':>7} {'Spec':>7} {'PPV':>7} {'F1':>7} {'Acc':>7} "
          f"{'TP':>3} {'FP':>3} {'TN':>3} {'FN':>3}")
    print("-" * 100)
    for name, m in metrics_by_strategy.items():
        print(f"  {name:<22} {m['threshold']:>6.4f} "
              f"{m['sensitivity']:>7.3f} {m['specificity']:>7.3f} "
              f"{m['PPV']:>7.3f} {m['F1']:>7.3f} {m['accuracy']:>7.3f} "
              f"{m['TP']:>3} {m['FP']:>3} {m['TN']:>3} {m['FN']:>3}")

    # CSV
    csv_path = args.output_dir / "hospital_strategy_metrics.csv"
    fields = ["strategy", "threshold", "sensitivity", "specificity",
              "PPV", "NPV", "F1", "accuracy", "TP", "FP", "TN", "FN", "n"]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for name, m in metrics_by_strategy.items():
            w.writerow([name] + [m[k] for k in fields[1:]])
    print(f"\nWrote {csv_path}")

    # Plots
    plot_threshold_comparison(metrics_by_strategy,
                              args.output_dir / "strategy_comparison.png")
    plot_score_distribution(
        {p: max_p[p] for p in shared},
        {p: gt[p] for p in shared},
        {"Legacy": (legacy, "tab:blue"),
         "Youden": (youden, "tab:orange"),
         "F1": (f1opt, "tab:green")},
        args.output_dir / "score_distribution_with_thresholds.png")
    print(f"Wrote {args.output_dir / 'strategy_comparison.png'}")
    print(f"Wrote {args.output_dir / 'score_distribution_with_thresholds.png'}")

    # LaTeX
    write_latex_table(metrics_by_strategy, args.output_dir / "hospital_eval.tex")
    print(f"Wrote {args.output_dir / 'hospital_eval.tex'}")

    # Per-patient table for the appendix
    per_patient_path = args.output_dir / "hospital_per_patient.csv"
    with open(per_patient_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "gt_hemorrhage", "max_p_any",
                    "legacy_pos", "youden_pos", "f1_pos", "sens95_pos"])
        for pid in shared:
            mp = max_p[pid]
            w.writerow([pid, gt[pid], round(mp, 4),
                        int(mp >= legacy), int(mp >= youden),
                        int(mp >= f1opt), int(mp >= sens95)])
    print(f"Wrote {per_patient_path}")


if __name__ == "__main__":
    main()
