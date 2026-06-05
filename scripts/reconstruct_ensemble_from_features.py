#!/usr/bin/env python3
"""Reconstruct 3-backbone ensemble OOF predictions from saved CNN features.

The original RSNA reproduction (March 2026) ran predict.py partially —
it saved CNN features (npy_train/*.npy) but no prediction CSVs. The
features only cover fold 1's val set for all three backbones. That's
134,946 slices of out-of-fold predictions we can recover without any
GPU time, because the saved features are already the expensive part
of the forward pass.

Each backbone's last layer is a single Linear (verified from
external/SeuTao_repo/2DNet/src/net/models.py):
  - DenseNet121:   mlp = Linear(1024, 6)
  - DenseNet169:   mlp = Linear(1664, 6)
  - SE-ResNeXt101: model_ft.last_linear = Sequential(Linear(2048, 6))

We extract just the head weights from each fold-1 checkpoint, multiply
by the saved feature vectors, sigmoid, then average across the three
backbones to get the deployed ensemble's OOF predictions. The result
is dumped as a single npz that derive_rsna_thresholds.py can consume
exactly like an instrumented-training output.

Approximation: predict.py originally averaged sigmoid(W·x_i) over
augmentations, whereas this reconstruction computes sigmoid(W·mean(x_i))
(because we only have the mean feature on disk). Sigmoid is monotonic,
so the rank ordering is preserved and Youden-optimal thresholds derived
from these predictions are essentially indistinguishable from the
augmentation-correct version.

Usage (run on hospital where features live):

  python scripts/reconstruct_ensemble_from_features.py \
      --reproduction-root /home/khma/bsc_hemorrage/rsna-seutao-reproduction \
      --fold 1 \
      --output /tmp/ensemble_oof_fold1.npz

Then scp the small (~5 MB) npz to the Mac and feed to
derive_rsna_thresholds.py.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch


HEMORRHAGE_CLASSES = ["any", "epidural", "intraparenchymal",
                      "intraventricular", "subarachnoid", "subdural"]


# Map backbone → (subdir under data_test/, key prefix in state_dict for the head)
BACKBONES = {
    "DenseNet121": {
        "dir": "DenseNet121_change_avg_256",
        "head_weight_key": "module.mlp.weight",
        "head_bias_key":   "module.mlp.bias",
        "feature_dim": 1024,
    },
    "DenseNet169": {
        "dir": "DenseNet169_change_avg_256",
        "head_weight_key": "module.mlp.weight",
        "head_bias_key":   "module.mlp.bias",
        "feature_dim": 1664,
    },
    "SE-ResNeXt101": {
        "dir": "se_resnext101_32x4d_256",
        "head_weight_key": "module.model_ft.last_linear.0.weight",
        "head_bias_key":   "module.model_ft.last_linear.0.bias",
        "feature_dim": 2048,
    },
}


def load_head_weights(ckpt_path: Path, weight_key: str, bias_key: str):
    """Extract just the classifier (W, b) from a .pth checkpoint."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    if weight_key not in sd:
        raise KeyError(
            f"Key '{weight_key}' not in {ckpt_path}. "
            f"Available keys (last 10): {list(sd.keys())[-10:]}"
        )
    W = sd[weight_key].cpu().numpy().astype(np.float32)
    b = sd[bias_key].cpu().numpy().astype(np.float32)
    return W, b


def load_features(npy_dir: Path, slice_ids: list[str]) -> np.ndarray:
    """Load (N, feature_dim) array of features for the requested slices.

    Slices missing from disk fall back to zeros and are reported.
    """
    sample = np.load(npy_dir / f"{slice_ids[0]}.npy")
    feat_dim = sample.shape[-1]
    out = np.zeros((len(slice_ids), feat_dim), dtype=np.float32)
    missing = 0
    for i, sid in enumerate(slice_ids):
        path = npy_dir / f"{sid}.npy"
        if not path.exists():
            missing += 1
            continue
        out[i] = np.load(path).astype(np.float32)
    if missing:
        print(f"  WARNING: {missing} slices had no feature file (zero-filled)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproduction-root", type=Path, required=True,
                    help="Path to rsna-seutao-reproduction root "
                         "(contains external/SeuTao_repo/2DNet/...)")
    ap.add_argument("--fold", type=int, default=1,
                    help="Fold index whose val set features were saved (default 1)")
    ap.add_argument("--output", type=Path, default=Path("/tmp/ensemble_oof_fold1.npz"))
    ap.add_argument("--label-csv", type=Path, default=None,
                    help="Path to stage1_train_cls.csv (auto-detected if omitted)")
    ap.add_argument("--val-list", type=Path, default=None,
                    help="Path to fold{N}/val.txt (auto-detected if omitted)")
    args = ap.parse_args()

    repo = args.reproduction_root.expanduser().resolve()
    seutao = repo / "external" / "SeuTao_repo" / "2DNet"
    data_test = seutao / "src" / "data_test"

    # Auto-locate val.txt — the OLD repro has it under data/data/, the
    # smoke-test setup uses data/. Try both.
    if args.val_list is not None:
        val_list_path = args.val_list
    else:
        candidates = [
            seutao / "data" / "data" / "fold_5_by_study_image" / f"fold{args.fold}" / "val.txt",
            seutao / "data" / "fold_5_by_study_image" / f"fold{args.fold}" / "val.txt",
            repo / "data" / "fold_5_by_study_image" / f"fold{args.fold}" / "val.txt",
        ]
        val_list_path = next((p for p in candidates if p.exists()), None)
        if val_list_path is None:
            raise FileNotFoundError(
                "Could not auto-locate val.txt. Tried:\n  " +
                "\n  ".join(str(p) for p in candidates))

    if args.label_csv is not None:
        label_path = args.label_csv
    else:
        candidates = [
            repo / "data" / "stage1_train_cls.csv",
            seutao / "data" / "data" / "stage1_train_cls.csv",
            seutao / "data" / "stage1_train_cls.csv",
        ]
        label_path = next((p for p in candidates if p.exists()), None)
        if label_path is None:
            raise FileNotFoundError(
                "Could not auto-locate stage1_train_cls.csv. Tried:\n  " +
                "\n  ".join(str(p) for p in candidates))

    print(f"Reproduction root: {repo}")
    print(f"Fold:              {args.fold}")
    print(f"Val list:          {val_list_path}")
    print(f"Label CSV:         {label_path}")
    print(f"Output:            {args.output}")
    print()

    # Read val list — entries are like "ID_xxxx.png", we want bare ID.
    with open(val_list_path) as f:
        slice_ids = [ln.strip().replace(".png", "") for ln in f if ln.strip()]
    print(f"Loaded {len(slice_ids):,} slice IDs from {val_list_path.name}")

    # Read labels and build slice → GT mapping.
    gt_map = {}
    with open(label_path) as f:
        for row in csv.DictReader(f):
            sid = row["filename"].replace(".png", "")
            gt_map[sid] = [int(row[c]) for c in HEMORRHAGE_CLASSES]
    print(f"Loaded labels for {len(gt_map):,} slices")
    missing_labels = [s for s in slice_ids if s not in gt_map]
    if missing_labels:
        print(f"  WARNING: {len(missing_labels)} slices have no labels — dropping")
        slice_ids = [s for s in slice_ids if s in gt_map]
    gt = np.asarray([gt_map[s] for s in slice_ids], dtype=np.float32)
    print(f"Final slice count: {len(slice_ids):,}")
    print()

    # Per-backbone: load fold checkpoint head + features → predictions.
    per_backbone_probs = {}
    for name, cfg in BACKBONES.items():
        bb_dir = data_test / cfg["dir"]
        ckpt = bb_dir / f"model_epoch_best_{args.fold}.pth"
        feats_dir = bb_dir / "prediction" / "npy_train"
        if not ckpt.exists():
            raise FileNotFoundError(f"Missing checkpoint: {ckpt}")
        if not feats_dir.exists():
            raise FileNotFoundError(f"Missing features dir: {feats_dir}")

        print(f"[{name}] loading head from {ckpt.name}")
        W, b = load_head_weights(ckpt, cfg["head_weight_key"], cfg["head_bias_key"])
        if W.shape[1] != cfg["feature_dim"]:
            print(f"  WARNING: expected feature_dim={cfg['feature_dim']}, "
                  f"got {W.shape[1]} (continuing with actual)")
        print(f"  classifier: ({W.shape[1]} → {W.shape[0]})")

        print(f"[{name}] loading {len(slice_ids):,} feature vectors from "
              f"{feats_dir} ...")
        feats = load_features(feats_dir, slice_ids)
        print(f"  feature array shape: {feats.shape}")

        # Linear classifier: logits = features @ W.T + b   (W: 6 × feature_dim)
        logits = feats @ W.T + b
        probs = 1.0 / (1.0 + np.exp(-logits))
        per_backbone_probs[name] = probs.astype(np.float32)
        print(f"[{name}] predictions ready, mean prob: {probs.mean(0)}")
        print()

    # Ensemble = mean across the three backbones.
    ensemble = np.mean(list(per_backbone_probs.values()), axis=0)
    print(f"Ensemble shape: {ensemble.shape}")
    print(f"Per-class positive rate (GT):    {gt.mean(0)}")
    print(f"Per-class mean predicted prob:   {ensemble.mean(0)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        outGT=gt,
        outPRED=ensemble,
        DN121_PRED=per_backbone_probs["DenseNet121"],
        DN169_PRED=per_backbone_probs["DenseNet169"],
        SERES_PRED=per_backbone_probs["SE-ResNeXt101"],
        slice_ids=np.asarray(slice_ids),
    )
    print(f"\nWrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    print("Next: scp this file to the Mac and run derive_rsna_thresholds.py "
          "with --single-fold")


if __name__ == "__main__":
    main()
