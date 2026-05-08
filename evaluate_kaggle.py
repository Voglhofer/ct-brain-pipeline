#!/usr/bin/env python3
"""
Evaluate the (current) ct-brain-pipeline on the Kaggle Brain Stroke CT
Dataset (ozguraslank/brain-stroke-ct-dataset).

This rewrite uses the **DICOM** files (with native HU) rather than the
pre-windowed PNG mirrors used by the original evaluate_kaggle.py. That
removes the lossy pseudo-HU reconstruction step.

The dataset is organised by class (Bleeding / Ischemia / Normal /
External_Test) with one DICOM file per slice (no patient series), so we
treat each .dcm as an independent slice and use the single-slice
preprocessing helpers from pipeline.py.

Usage:
  python evaluate_kaggle.py                    # all 3 classes, MPS/CUDA if available
  python evaluate_kaggle.py --limit 5          # smoke test
  python evaluate_kaggle.py --output-dir output_kaggle_v2
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pipeline import (
    HEMORRHAGE_LABELS,
    HEMORRHAGE_MODEL_DIR,
    HEMORRHAGE_THRESHOLDS,
    ISCHEMIC_MODEL_PATH,
    dicom_to_hu,
    load_hemorrhage_models,
    load_ischemic_model,
    predict_hemorrhage_batch,
    predict_ischemic_batch,
    prepare_hemorrhage_input,
    prepare_ischemic_input,
)

GT_CLASSES = {
    "Bleeding":  {"hemorrhage": True,  "ischemic": False},
    "Ischemia":  {"hemorrhage": False, "ischemic": True},
    "Normal":    {"hemorrhage": False, "ischemic": False},
}

# Ischemic model output threshold (matches pipeline default)
ISCHEMIC_THRESHOLD = 0.5


# ── AUC helpers (numpy-only, no sklearn) ─────────────────────────────────

def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_score[y_true == 1]; neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(y_score, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1)
    s = y_score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            for k in range(i, j + 1):
                ranks[order[k]] = avg
        i = j + 1
    n_pos = pos.size; n_neg = neg.size
    rs = ranks[y_true == 1].sum()
    return float((rs - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_auc_ci(y_true, y_score, n_boot=1000, seed=42):
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


# ── Image loading ────────────────────────────────────────────────────────

def collect_dicoms(root: Path, classes: list[str], limit: int | None):
    items = []
    for cls in classes:
        cls_dir = root / cls / "DICOM"
        if not cls_dir.is_dir():
            print(f"  WARNING: missing {cls_dir}")
            continue
        files = sorted(cls_dir.glob("*.dcm"))
        if limit:
            files = files[:limit]
        items.extend((p, cls) for p in files)
        print(f"  {cls}: {len(files)} DICOM(s)")
    return items


def find_dataset_root(download_path: Path) -> Path:
    for c in [download_path] + [p for p in download_path.rglob("*") if p.is_dir()]:
        if any((c / cls).is_dir() for cls in GT_CLASSES):
            return c
    raise FileNotFoundError(f"Could not find Bleeding/Ischemia/Normal under {download_path}")


# ── Evaluation ───────────────────────────────────────────────────────────

def evaluate(items, hem_models, isch_model, device, batch_size, csv_path):
    n = len(items)
    print(f"\nRunning inference on {n} slice(s) (batch size {batch_size})…")

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "file", "class",
        "gt_hemorrhage", "pred_hemorrhage", "p_any",
        *[f"p_{l}" for l in HEMORRHAGE_LABELS],
        "gt_ischemic", "pred_ischemic", "p_ischemic",
    ]

    rows = []
    t0 = time.time()
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()

        for start in range(0, n, batch_size):
            batch = items[start:start + batch_size]
            hem_inputs, isch_inputs, valid = [], [], []
            for path, cls in batch:
                try:
                    hu = dicom_to_hu(str(path))
                    hem_inputs.append(prepare_hemorrhage_input(hu))
                    isch_inputs.append(prepare_ischemic_input(hu))
                    valid.append((path, cls))
                except Exception as e:
                    print(f"  SKIP {path.name}: {e}")
            if not valid:
                continue

            hem_results = predict_hemorrhage_batch(hem_models, hem_inputs, device, batch_size)
            isch_results = predict_ischemic_batch(isch_model, isch_inputs, device, batch_size)

            for (path, cls), hem, isch in zip(valid, hem_results, isch_results):
                gt = GT_CLASSES[cls]
                pred_hem = hem["any"]["positive"]
                pred_isch = isch["ischemic_stroke"]["positive"]
                row = {
                    "file": str(path),
                    "class": cls,
                    "gt_hemorrhage": int(gt["hemorrhage"]),
                    "pred_hemorrhage": int(pred_hem),
                    "p_any": hem["any"]["probability"],
                    "gt_ischemic": int(gt["ischemic"]),
                    "pred_ischemic": int(pred_isch),
                    "p_ischemic": isch["ischemic_stroke"]["probability"],
                }
                for l in HEMORRHAGE_LABELS:
                    row[f"p_{l}"] = hem[l]["probability"]
                writer.writerow(row)
                rows.append(row)

            done = start + len(batch)
            if done % (batch_size * 10) == 0 or done >= n:
                rate = done / max(time.time() - t0, 1e-6)
                eta = (n - done) / max(rate, 1e-6)
                print(f"  [{done}/{n}]  {rate:.1f} img/s  eta {eta/60:.1f} min")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s ({n / max(elapsed, 1e-6):.1f} img/s)")
    return rows


def metrics(tp, fp, tn, fn):
    acc = (tp + tn) / max(tp + fp + tn + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return acc, prec, rec, spec, f1


def print_report(rows, log_lines):
    def p(s=""):
        print(s); log_lines.append(s)

    p("\n" + "=" * 72)
    p(f"  Kaggle Brain Stroke CT — Slice-level results (n={len(rows)})")
    p("=" * 72)

    classes = {}
    for r in rows:
        classes.setdefault(r["class"], []).append(r)
    p("\n  Per-class counts:")
    for cls, rs in classes.items():
        p(f"    {cls:14s} n={len(rs)}")

    p("\n  Per-class binary metrics (Youden thresholds from RSNA training):")
    for cls, rs in classes.items():
        if cls not in GT_CLASSES:
            continue
        tp=fp=tn=fn=0
        for r in rs:
            g, pr = int(r["gt_hemorrhage"]), int(r["pred_hemorrhage"])
            if g and pr: tp+=1
            elif g: fn+=1
            elif pr: fp+=1
            else: tn+=1
        a,_,_,_,_ = metrics(tp,fp,tn,fn)
        p(f"    {cls:14s} HEM   acc={a*100:6.2f}%  TP={tp} FP={fp} TN={tn} FN={fn}")
        tp=fp=tn=fn=0
        for r in rs:
            g, pr = int(r["gt_ischemic"]), int(r["pred_ischemic"])
            if g and pr: tp+=1
            elif g: fn+=1
            elif pr: fp+=1
            else: tn+=1
        a,_,_,_,_ = metrics(tp,fp,tn,fn)
        p(f"    {' ':14s} ISCH  acc={a*100:6.2f}%  TP={tp} FP={fp} TN={tn} FN={fn}")

    p("\n  AUC (slice-level discrimination):")
    hem_rows = [r for r in rows if r["class"] in ("Bleeding", "Normal")]
    if hem_rows:
        y = np.array([int(r["gt_hemorrhage"]) for r in hem_rows])
        ps = np.array([float(r["p_any"]) for r in hem_rows])
        auc = roc_auc(y, ps); lo, hi = bootstrap_auc_ci(y, ps)
        p(f"    Hemorrhage (Bleeding vs Normal, n={len(hem_rows)})  AUC={auc:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")
    isch_rows = [r for r in rows if r["class"] in ("Ischemia", "Normal")]
    if isch_rows:
        y = np.array([int(r["gt_ischemic"]) for r in isch_rows])
        ps = np.array([float(r["p_ischemic"]) for r in isch_rows])
        auc = roc_auc(y, ps); lo, hi = bootstrap_auc_ci(y, ps)
        p(f"    Ischemic   (Ischemia vs Normal, n={len(isch_rows)})  AUC={auc:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")
    all_rows = [r for r in rows if r["class"] in ("Bleeding", "Ischemia", "Normal")]
    if all_rows:
        y = np.array([1 if r["class"] in ("Bleeding","Ischemia") else 0 for r in all_rows])
        ps = np.array([max(float(r["p_any"]), float(r["p_ischemic"])) for r in all_rows])
        auc = roc_auc(y, ps); lo, hi = bootstrap_auc_ci(y, ps)
        p(f"    Any stroke (Bleeding+Ischemia vs Normal, n={len(all_rows)})  AUC={auc:.4f}  [95% CI {lo:.4f}, {hi:.4f}]")


# ── Plots ────────────────────────────────────────────────────────────────

def make_plots(rows, out_dir: Path):
    import matplotlib.pyplot as plt

    aucs, lo, hi, names = [], [], [], []
    for name, cls in [
        ("Hemorrhage\nvs Normal", ("Bleeding","Normal")),
        ("Ischemia\nvs Normal",   ("Ischemia","Normal")),
        ("Any stroke\nvs Normal", ("Bleeding","Ischemia","Normal")),
    ]:
        rs = [r for r in rows if r["class"] in cls]
        if name.startswith("Hemorrhage"):
            y = np.array([int(r["gt_hemorrhage"]) for r in rs]); ps = np.array([float(r["p_any"]) for r in rs])
        elif name.startswith("Ischemia"):
            y = np.array([int(r["gt_ischemic"]) for r in rs]); ps = np.array([float(r["p_ischemic"]) for r in rs])
        else:
            y = np.array([1 if r["class"] in ("Bleeding","Ischemia") else 0 for r in rs])
            ps = np.array([max(float(r["p_any"]), float(r["p_ischemic"])) for r in rs])
        a = roc_auc(y, ps); l_, h_ = bootstrap_auc_ci(y, ps)
        aucs.append(a); lo.append(l_); hi.append(h_); names.append(name)

    # 1. AUC barplot
    colors = ["#c0392b", "#2980b9", "#7f8c8d"]
    fig, ax = plt.subplots(figsize=(7,5))
    err = [[a-l for a,l in zip(aucs,lo)], [h-a for a,h in zip(aucs,hi)]]
    bars = ax.bar(names, aucs, color=colors, edgecolor="black", yerr=err, capsize=6)
    ax.axhline(0.5, color="gray", linestyle="--", lw=1, label="Chance (AUC = 0.5)")
    ax.set_ylim(0, 1.0); ax.set_ylabel("AUC (slice-level)")
    ax.set_title(f"Kaggle Brain Stroke CT — Slice-level AUC\n(ct-brain-pipeline, n={len(rows)} DICOM slices)")
    for bar, a in zip(bars, aucs):
        ax.text(bar.get_x()+bar.get_width()/2, a+0.03, f"{a:.3f}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); plt.savefig(out_dir / "kaggle_auc_barplot.png", dpi=200); plt.close()

    # 2. ROC curves
    def roc_curve(y, s):
        y = np.asarray(y); s = np.asarray(s)
        order = np.argsort(-s); y = y[order]
        tps = np.cumsum(y == 1); fps = np.cumsum(y == 0)
        P = (y == 1).sum(); N = (y == 0).sum()
        return np.concatenate([[0], fps/N]), np.concatenate([[0], tps/P])

    fig, ax = plt.subplots(figsize=(7,6))
    pairs = [
        ("Hemorrhage (Bleeding vs Normal)", ("Bleeding","Normal"), "gt_hemorrhage", "p_any", "#c0392b"),
        ("Ischemia (Ischemia vs Normal)",   ("Ischemia","Normal"), "gt_ischemic",   "p_ischemic", "#2980b9"),
    ]
    for nm, cls, gtk, pk, col in pairs:
        rs = [r for r in rows if r["class"] in cls]
        y = np.array([int(r[gtk]) for r in rs]); ps = np.array([float(r[pk]) for r in rs])
        fpr, tpr = roc_curve(y, ps)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{nm}  (AUC = {roc_auc(y,ps):.3f})")
    rs = [r for r in rows if r["class"] in ("Bleeding","Ischemia","Normal")]
    y = np.array([1 if r["class"] in ("Bleeding","Ischemia") else 0 for r in rs])
    ps = np.array([max(float(r["p_any"]), float(r["p_ischemic"])) for r in rs])
    fpr, tpr = roc_curve(y, ps)
    ax.plot(fpr, tpr, color="#7f8c8d", lw=2, label=f"Any stroke vs Normal  (AUC = {roc_auc(y,ps):.3f})")
    ax.plot([0,1],[0,1],"--",color="gray",lw=1,label="Chance")
    ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Kaggle Brain Stroke CT — ROC Curves\n(ct-brain-pipeline, slice-level)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    plt.tight_layout(); plt.savefig(out_dir / "kaggle_roc_curves.png", dpi=200); plt.close()

    # 3. Score distributions
    fig, axes = plt.subplots(1, 2, figsize=(13,5), sharey=True)
    bins = np.linspace(0, 1, 41)
    for cls, col, alpha in [("Normal","#27ae60",0.6),("Ischemia","#2980b9",0.6),("Bleeding","#c0392b",0.6)]:
        vals = [float(r["p_any"]) for r in rows if r["class"] == cls]
        axes[0].hist(vals, bins=bins, color=col, alpha=alpha,
                     label=f"{cls} (n={len(vals)})", edgecolor="black", linewidth=0.3)
    axes[0].set_xlabel("Hemorrhage probability  (p_any)")
    axes[0].set_ylabel("Number of slices")
    axes[0].set_title("Hemorrhage score distribution by class")
    axes[0].legend(frameon=False)
    axes[0].spines["top"].set_visible(False); axes[0].spines["right"].set_visible(False)
    for cls, col, alpha in [("Normal","#27ae60",0.6),("Bleeding","#c0392b",0.6),("Ischemia","#2980b9",0.6)]:
        vals = [float(r["p_ischemic"]) for r in rows if r["class"] == cls]
        axes[1].hist(vals, bins=bins, color=col, alpha=alpha,
                     label=f"{cls} (n={len(vals)})", edgecolor="black", linewidth=0.3)
    axes[1].set_xlabel("Ischemic probability  (p_ischemic)")
    axes[1].set_title("Ischemic score distribution by class")
    axes[1].legend(frameon=False)
    axes[1].spines["top"].set_visible(False); axes[1].spines["right"].set_visible(False)
    plt.suptitle("Kaggle Brain Stroke CT — Score distributions  (ct-brain-pipeline)", y=1.02)
    plt.tight_layout(); plt.savefig(out_dir / "kaggle_score_distributions.png", dpi=200, bbox_inches="tight")
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    default_device = ("cuda" if torch.cuda.is_available()
                      else ("mps" if torch.backends.mps.is_available() else "cpu"))
    parser.add_argument("--device", default=default_device, choices=["cpu","cuda","mps"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit DICOMs per class (smoke test)")
    parser.add_argument("--classes", nargs="+", default=list(GT_CLASSES.keys()),
                        choices=list(GT_CLASSES.keys()))
    parser.add_argument("--dataset-path", default=None,
                        help="Local path; if omitted uses kagglehub cache")
    parser.add_argument("--output-dir", default="output_kaggle_v2")
    args = parser.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset_path:
        dl_path = Path(args.dataset_path)
    else:
        try:
            import kagglehub
        except ImportError:
            print("ERROR: pip install kagglehub", file=sys.stderr); return 1
        dl_path = Path(kagglehub.dataset_download("ozguraslank/brain-stroke-ct-dataset"))
    print(f"Dataset cached at: {dl_path}")
    root = find_dataset_root(dl_path)
    print(f"Dataset root: {root}")

    print("\nCollecting DICOMs:")
    items = collect_dicoms(root, args.classes, args.limit)
    if not items:
        print("ERROR: no DICOMs found", file=sys.stderr); return 1

    device = torch.device(args.device)
    print(f"\nLoading models on {device} …")
    hem_models = load_hemorrhage_models(HEMORRHAGE_MODEL_DIR, device)
    isch_model = load_ischemic_model(ISCHEMIC_MODEL_PATH, device)
    if not hem_models:
        print("ERROR: no hemorrhage models", file=sys.stderr); return 1

    csv_path = out_dir / "predictions.csv"
    rows = evaluate(items, hem_models, isch_model, device, args.batch_size, csv_path)

    log_lines = []
    print_report(rows, log_lines)
    (out_dir / "results_summary.txt").write_text("\n".join(log_lines))

    print("\nGenerating plots …")
    make_plots(rows, out_dir)
    print(f"  → {out_dir}/kaggle_auc_barplot.png")
    print(f"  → {out_dir}/kaggle_roc_curves.png")
    print(f"  → {out_dir}/kaggle_score_distributions.png")
    print(f"\nPer-image predictions: {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
