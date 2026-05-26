"""Centralized metric computation and logging utilities.

Used by both 2DNet/src/train.py and SequenceModel/main.py to capture every
metric that can be computed from (ground_truth, predictions) pairs without
re-running inference.

Design constraints:
- Pure additive — never affects the training loop's gradients, optimizer
  state, RNG, or epoch ordering.
- All metrics are derived from outGT/outPRED, so there is no extra forward
  pass. The raw arrays are dumped to npz so any metric can be recomputed
  post-hoc without re-training.
"""

import csv
import json
import os
import time
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

try:
    from sklearn.calibration import calibration_curve
except ImportError:
    calibration_curve = None


HEMORRHAGE_CLASSES = ["any", "epidural", "intraparenchymal",
                     "intraventricular", "subarachnoid", "subdural"]


def _safe(fn, default=float("nan"), **kwargs):
    """Run a metric fn; return default if it raises (e.g., single-class y_true)."""
    try:
        return float(fn(**kwargs))
    except Exception:
        return default


def _per_class_threshold_metrics(gt: np.ndarray, pred: np.ndarray, thr: float):
    """Return TP/FP/TN/FN/precision/recall/F1/sens/spec at a given threshold."""
    y_pred = (pred >= thr).astype(np.int32)
    y_true = gt.astype(np.int32)

    # sklearn confusion_matrix requires both classes present; do it manually.
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sens = rec
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "precision": prec, "recall": rec, "F1": f1,
        "sensitivity": sens, "specificity": spec,
    }


def _youden_threshold(gt: np.ndarray, pred: np.ndarray):
    """Return (threshold, sens, spec) maximising J = sens + spec - 1."""
    if gt.sum() == 0 or gt.sum() == len(gt):
        return float("nan"), float("nan"), float("nan")
    fpr, tpr, thresholds = roc_curve(gt, pred)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thresholds[idx]), float(tpr[idx]), float(1 - fpr[idx])


def _f1_optimal_threshold(gt: np.ndarray, pred: np.ndarray):
    """Return (threshold, F1) maximising F1 over PR-curve thresholds."""
    if gt.sum() == 0:
        return float("nan"), float("nan")
    precision, recall, thresholds = precision_recall_curve(gt, pred)
    # precision/recall have len(thresholds)+1; align by dropping last element.
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    if len(f1) == 0:
        return float("nan"), float("nan")
    idx = int(np.argmax(f1))
    return float(thresholds[idx]), float(f1[idx])


def _calibration_data(gt: np.ndarray, pred: np.ndarray, n_bins: int = 10):
    """Return (prob_true, prob_pred) for calibration plot. Empty arrays if undefined."""
    if calibration_curve is None or gt.sum() == 0 or gt.sum() == len(gt):
        return [], []
    try:
        prob_true, prob_pred = calibration_curve(gt, pred, n_bins=n_bins, strategy="uniform")
        return prob_true.tolist(), prob_pred.tolist()
    except Exception:
        return [], []


def compute_full_metrics(gt: np.ndarray, pred: np.ndarray,
                         class_names=None, default_threshold: float = 0.5) -> dict:
    """Return a dict of every metric derivable from (gt, pred).

    Parameters
    ----------
    gt   : (N, C) ground-truth one-hot/binary array
    pred : (N, C) predicted probability (post-sigmoid) array
    class_names : list of C names. Defaults to RSNA's 6 hemorrhage labels.
    default_threshold : threshold used for "at 0.5" metrics.

    Returns
    -------
    dict with keys:
        per_class[class_name]: dict of all per-class metrics
        macro: dict of macro-averaged scalar metrics
        meta: array shapes, n_pos, n_neg per class
    """
    gt = np.asarray(gt)
    pred = np.asarray(pred)
    if gt.ndim == 1:
        gt = gt.reshape(-1, 1)
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    n, c = gt.shape
    if class_names is None:
        class_names = HEMORRHAGE_CLASSES[:c] if c <= len(HEMORRHAGE_CLASSES) else [f"c{i}" for i in range(c)]

    per_class = {}
    for i, name in enumerate(class_names):
        gi = gt[:, i].astype(np.float64)
        pi = pred[:, i].astype(np.float64)
        n_pos = int(gi.sum())
        n_neg = int(len(gi) - n_pos)

        auc = _safe(roc_auc_score, y_true=gi, y_score=pi)
        pr_auc = _safe(average_precision_score, y_true=gi, y_score=pi)

        thr05 = _per_class_threshold_metrics(gi, pi, default_threshold)

        thr_y, sens_y, spec_y = _youden_threshold(gi, pi)
        thr_f, f1_f = _f1_optimal_threshold(gi, pi)

        # Metrics at Youden threshold (if defined)
        thr_y_metrics = (_per_class_threshold_metrics(gi, pi, thr_y)
                        if np.isfinite(thr_y) else None)
        thr_f_metrics = (_per_class_threshold_metrics(gi, pi, thr_f)
                        if np.isfinite(thr_f) else None)

        calib_true, calib_pred = _calibration_data(gi, pi)

        per_class[name] = {
            "n_pos": n_pos,
            "n_neg": n_neg,
            "AUC": auc,
            "PR_AUC": pr_auc,
            f"at_{default_threshold}": thr05,
            "youden": {
                "threshold": thr_y,
                "sensitivity": sens_y,
                "specificity": spec_y,
                "full_metrics": thr_y_metrics,
            },
            "f1_optimal": {
                "threshold": thr_f,
                "F1": f1_f,
                "full_metrics": thr_f_metrics,
            },
            "calibration": {
                "prob_true": calib_true,
                "prob_pred": calib_pred,
            },
        }

    # Macro-averaged scalars
    aucs = [per_class[n]["AUC"] for n in class_names if np.isfinite(per_class[n]["AUC"])]
    pr_aucs = [per_class[n]["PR_AUC"] for n in class_names if np.isfinite(per_class[n]["PR_AUC"])]
    f1s = [per_class[n][f"at_{default_threshold}"]["F1"] for n in class_names]

    macro = {
        "macro_AUC": float(np.mean(aucs)) if aucs else float("nan"),
        "macro_PR_AUC": float(np.mean(pr_aucs)) if pr_aucs else float("nan"),
        "macro_F1_at_0.5": float(np.mean(f1s)) if f1s else float("nan"),
    }

    return {
        "per_class": per_class,
        "macro": macro,
        "meta": {
            "n_samples": int(n),
            "n_classes": int(c),
            "class_names": list(class_names),
            "default_threshold": float(default_threshold),
        },
    }


def flatten_for_csv(metrics: dict, prefix: str = "") -> dict:
    """Flatten compute_full_metrics() output into a single-level dict
    suitable for one CSV row.

    Keys follow the pattern e.g.:
      AUC_any, PR_AUC_any, F1@0.5_any, Sens@0.5_any, Spec@0.5_any,
      TP_any, FP_any, TN_any, FN_any,
      ThrYouden_any, Sens@Youden_any, Spec@Youden_any,
      ThrF1opt_any, F1@F1opt_any,
      Macro_AUC, Macro_PR_AUC, ...
    """
    out = {}
    for name, m in metrics["per_class"].items():
        suffix = f"_{name}"
        out[f"{prefix}AUC{suffix}"] = m["AUC"]
        out[f"{prefix}PR_AUC{suffix}"] = m["PR_AUC"]
        out[f"{prefix}n_pos{suffix}"] = m["n_pos"]
        out[f"{prefix}n_neg{suffix}"] = m["n_neg"]
        thr05 = m[f"at_{metrics['meta']['default_threshold']}"]
        out[f"{prefix}Prec@0.5{suffix}"] = thr05["precision"]
        out[f"{prefix}Rec@0.5{suffix}"] = thr05["recall"]
        out[f"{prefix}F1@0.5{suffix}"] = thr05["F1"]
        out[f"{prefix}Sens@0.5{suffix}"] = thr05["sensitivity"]
        out[f"{prefix}Spec@0.5{suffix}"] = thr05["specificity"]
        out[f"{prefix}TP{suffix}"] = thr05["TP"]
        out[f"{prefix}FP{suffix}"] = thr05["FP"]
        out[f"{prefix}TN{suffix}"] = thr05["TN"]
        out[f"{prefix}FN{suffix}"] = thr05["FN"]
        out[f"{prefix}ThrYouden{suffix}"] = m["youden"]["threshold"]
        out[f"{prefix}Sens@Youden{suffix}"] = m["youden"]["sensitivity"]
        out[f"{prefix}Spec@Youden{suffix}"] = m["youden"]["specificity"]
        out[f"{prefix}ThrF1opt{suffix}"] = m["f1_optimal"]["threshold"]
        out[f"{prefix}F1@F1opt{suffix}"] = m["f1_optimal"]["F1"]
    out[f"{prefix}Macro_AUC"] = metrics["macro"]["macro_AUC"]
    out[f"{prefix}Macro_PR_AUC"] = metrics["macro"]["macro_PR_AUC"]
    out[f"{prefix}Macro_F1@0.5"] = metrics["macro"]["macro_F1_at_0.5"]
    return out


class GradNormAccumulator:
    """Track gradient L2-norms across batches WITHOUT modifying gradients.

    Call .record(model) AFTER loss.backward() but BEFORE optimizer.step()
    or optimizer.zero_grad(). The norm is computed from the .grad tensors
    and does not clip them.
    """

    def __init__(self):
        self.values = []

    def record(self, model):
        if torch is None:
            return
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is None:
                continue
            param_norm = p.grad.detach().data.norm(2)
            total_norm_sq += float(param_norm.item()) ** 2
        self.values.append(total_norm_sq ** 0.5)

    def summary(self):
        if not self.values:
            return {"mean": float("nan"), "std": float("nan"), "max": float("nan"), "n": 0}
        arr = np.asarray(self.values)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "max": float(arr.max()),
            "n": int(len(arr)),
        }

    def reset(self):
        self.values = []


def system_metrics():
    """Return dict of GPU memory + cuda info. Safe if no GPU available."""
    out = {
        "gpu_peak_mem_GB": 0.0,
        "gpu_reserved_mem_GB": 0.0,
        "gpu_available": False,
    }
    if torch is None:
        return out
    if not torch.cuda.is_available():
        return out
    out["gpu_available"] = True
    out["gpu_peak_mem_GB"] = float(torch.cuda.max_memory_allocated() / 1e9)
    out["gpu_reserved_mem_GB"] = float(torch.cuda.memory_reserved() / 1e9)
    return out


def reset_system_metrics():
    """Reset CUDA peak-memory counter for the next epoch's window."""
    if torch is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


class MetricLogger:
    """Per-fold metric logger. One instance per (fold, backbone) combination.

    Writes:
      - <root>/log.csv             — flat row per epoch
      - <root>/epoch_<E>_val_predictions.npz — outGT, outPRED (raw)
      - <root>/epoch_<E>_metrics.json — structured metrics
      - <root>/train_metadata.json — written once at start
      - <root>/best_epoch.json     — updated when best val AUC improves
    """

    def __init__(self, root: str, csv_columns: list = None,
                 save_predictions: bool = True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.root / "log.csv"
        self.csv_columns = csv_columns
        self.save_predictions = save_predictions
        self.best_score = -float("inf")
        self.best_epoch = -1

    def write_metadata(self, payload: dict):
        with open(self.root / "train_metadata.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def write_csv_header(self, columns: list):
        """Write CSV header. Call ONCE per run BEFORE first log_epoch."""
        self.csv_columns = columns
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(columns)

    def log_epoch(self, epoch: int, row: dict, full_metrics: dict = None,
                  gt: np.ndarray = None, pred: np.ndarray = None):
        """Write one CSV row + optionally dump predictions and structured metrics."""
        # CSV
        if self.csv_columns is None:
            self.csv_columns = list(row.keys())
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.csv_columns)

        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([row.get(col, "") for col in self.csv_columns])

        # Structured JSON
        if full_metrics is not None:
            with open(self.root / f"epoch_{epoch:03d}_metrics.json", "w") as f:
                json.dump(full_metrics, f, indent=2, default=str)

        # Raw predictions
        if self.save_predictions and gt is not None and pred is not None:
            npz_path = self.root / f"epoch_{epoch:03d}_val_predictions.npz"
            np.savez_compressed(npz_path,
                                outGT=np.asarray(gt, dtype=np.float32),
                                outPRED=np.asarray(pred, dtype=np.float32))

    def update_best(self, epoch: int, score: float, extra: dict = None):
        """If score > previous best, record it. Returns True if improved."""
        if not np.isfinite(score):
            return False
        if score > self.best_score:
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            payload = {"epoch": int(epoch), "score": float(score)}
            if extra:
                payload.update(extra)
            with open(self.root / "best_epoch.json", "w") as f:
                json.dump(payload, f, indent=2, default=str)
            return True
        return False
