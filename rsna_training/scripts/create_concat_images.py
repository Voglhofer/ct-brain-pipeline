import argparse
import os
from pathlib import Path
import pandas as pd
import cv2
from tqdm import tqdm

def build_concat(df: pd.DataFrame, png_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    skipped = 0

    for study, sdf in tqdm(df.groupby("study_instance_uid"), desc=f"build {out_dir.name}"):
        sdf = sdf.copy()
        if "slice_id" in sdf.columns:
            sdf["slice_num"] = sdf["slice_id"].astype(str).str.split("_").str[-1].astype(int)
        else:
            # stage2_test_cls.csv lacks slice_id; use image_position (z-coord) instead
            sdf["slice_num"] = pd.to_numeric(sdf["image_position"], errors="coerce").fillna(0)
        sdf = sdf.sort_values("slice_num").reset_index(drop=True)

        files = sdf["filename"].tolist()
        for i, fname in enumerate(files):
            prev_f = files[i - 1] if i > 0 else fname
            next_f = files[i + 1] if i < len(files) - 1 else fname

            img_prev = cv2.imread(str(png_dir / prev_f), 0)
            img_cur = cv2.imread(str(png_dir / fname), 0)
            img_next = cv2.imread(str(png_dir / next_f), 0)

            if img_prev is None or img_cur is None or img_next is None:
                skipped += 1
                continue

            img_prev = cv2.resize(img_prev, (512, 512))
            img_cur = cv2.resize(img_cur, (512, 512))
            img_next = cv2.resize(img_next, (512, 512))

            merged = cv2.merge([img_prev, img_cur, img_next])
            merged = cv2.resize(merged, (256, 256))
            cv2.imwrite(str(out_dir / fname), merged)

    if skipped:
        print(f"  ⚠ Skipped {skipped} slices due to missing PNGs in {out_dir.name}")

def load_config():
    """Load config.env and expand variables."""
    config_path = Path(os.getcwd()) / "config.env"
    if not config_path.exists():
        return {}
    env = {}
    for line in config_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = value.strip()
    env["PROJECT_ROOT"] = str(Path(os.getcwd()))
    for _ in range(10):
        changed = False
        for k in list(env.keys()):
            for k2, v2 in env.items():
                token = "${" + k2 + "}"
                if token in env[k]:
                    env[k] = env[k].replace(token, v2)
                    changed = True
        if not changed:
            break
    return env

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    cfg = load_config()

    stage1_csv = repo_root / "2DNet" / "data" / "stage1_train_cls.csv"
    stage2_csv = repo_root / "2DNet" / "data" / "stage2_test_cls.csv"

    train_png = Path(cfg.get("RSNA_TRAIN_PNG_DIR", str(repo_root / "2DNet" / "train_png")))
    test_png = Path(cfg.get("RSNA_TEST_PNG_DIR", str(repo_root / "2DNet" / "test_png")))

    train_out = Path(cfg.get("RSNA_CONCAT_TRAIN_DIR", str(repo_root / "2DNet" / "train_concat_3images_256")))
    test_out = Path(cfg.get("RSNA_CONCAT_TEST_DIR", str(repo_root / "2DNet" / "stage2_test_concat_3images")))

    df1 = pd.read_csv(stage1_csv)
    df2 = pd.read_csv(stage2_csv)

    build_concat(df1, train_png, train_out)
    build_concat(df2, test_png, test_out)

if __name__ == "__main__":
    main()