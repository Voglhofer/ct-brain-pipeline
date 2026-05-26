import argparse
from pathlib import Path
import numpy as np
import pandas as pd

MODELS = [
    "DenseNet121_change_avg_256",
    "DenseNet169_change_avg_256",
    "se_resnext101_32x4d_256",
]

def normalize_name(x):
    x = str(x)
    return x.replace(".png", "").replace(".dcm", "")

def load_val_names(csv_root: Path):
    names = []
    for fold in range(5):
        df = pd.read_csv(csv_root / f"val_fold{fold}.csv")
        names.extend([normalize_name(v) for v in df["filename"].tolist()])
    return names

def load_test_names(csv_root: Path):
    df = pd.read_csv(csv_root / "stage_2_sample_submission.csv")
    df["filename"] = df["ID"].apply(lambda s: "ID_" + s.split("_")[1])
    return list(pd.unique(df["filename"].map(normalize_name)))

def load_train_features(npy_dir: Path, ordered_names):
    rows = []
    for name in ordered_names:
        f = npy_dir / f"{name}.npy"
        if not f.exists():
            raise FileNotFoundError(f"Mangler train feature: {f}")
        rows.append(np.load(f).astype(np.float32))
    return np.stack(rows).astype(np.float16)

def load_test_features(npy_dir: Path, ordered_names):
    rows = []
    for name in ordered_names:
        files = sorted(npy_dir.glob(f"{name}_*.npy"))
        if not files:
            raise FileNotFoundError(f"Mangler test-features for {name} i {npy_dir}")
        arrs = [np.load(f).astype(np.float32) for f in files]
        rows.append(np.mean(arrs, axis=0))
    return np.stack(rows).astype(np.float16)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    csv_root = repo_root / "SequenceModel" / "csv"
    feature_root = repo_root / "SequenceModel" / "features" / "stage2_finetune"
    feature_root.mkdir(parents=True, exist_ok=True)

    val_names = load_val_names(csv_root)
    test_names = load_test_names(csv_root)

    for model_name in MODELS:
        pred_dir = repo_root / "2DNet" / "src" / "data_test" / model_name / "prediction"
        if not pred_dir.exists():
            raise FileNotFoundError(f"Mangler prediction-dir: {pred_dir}")

        out_dir = feature_root / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        val_csv = pred_dir / "val_aug_10.csv"
        test_csv = pred_dir / "test_aug_10.csv"
        if not val_csv.exists() or not test_csv.exists():
            raise FileNotFoundError(f"Mangler val/test csv i {pred_dir}")

        pd.read_csv(val_csv).to_csv(out_dir / f"{model_name}_val_prob_TTA.csv", index=False)
        pd.read_csv(test_csv).to_csv(out_dir / f"{model_name}_test_prob_TTA_stage2.csv", index=False)

        val_fea = load_train_features(pred_dir / "npy_train", val_names)
        test_fea = load_test_features(pred_dir / "npy_test", test_names)

        np.save(out_dir / f"{model_name}_val_oof_feature_TTA.npy", val_fea)
        np.save(out_dir / f"{model_name}_test_feature_TTA_stage2.npy", test_fea)

        print(f"Byggede sequence-inputs for {model_name}")

if __name__ == "__main__":
    main()