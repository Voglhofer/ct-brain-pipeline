#!/usr/bin/env python
"""
Download RSNA Intracranial Hemorrhage Detection dataset via kagglehub.

Requires:
  pip install kagglehub
  Kaggle credentials set up (KAGGLE_USERNAME + KAGGLE_KEY env vars, or ~/.kaggle/kaggle.json)

Creates symlinks from the kagglehub cache into data/rsna_raw/ so the pipeline
can find the DICOM directories and CSV files.

NOTE: Kaggle only provides Stage 2 data now.  The actual directory layout is:
  .../rsna-intracranial-hemorrhage-detection/
      rsna-intracranial-hemorrhage-detection/
          stage_2_train/            (DICOM files — used as "train" images)
          stage_2_test/             (DICOM files — used as "test"  images)
          stage_2_train.csv
          stage_2_sample_submission.csv
"""

import os
import sys
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("kagglehub ikke installeret. Kør: pip install kagglehub", file=sys.stderr)
    sys.exit(1)


def _find_data_root(base: Path) -> Path:
    """
    kagglehub kan returnere en sti hvor det faktiske indhold ligger ét eller
    to niveauer nede.  Find den mappe der indeholder stage_2_train eller
    stage_2_train.csv.
    """
    # Tjek base selv
    if (base / "stage_2_train").is_dir() or (base / "stage_2_train.csv").is_file():
        return base
    # Tjek ét niveau ned
    for child in base.iterdir():
        if child.is_dir():
            if (child / "stage_2_train").is_dir() or (child / "stage_2_train.csv").is_file():
                return child
            # To niveauer ned
            for grandchild in child.iterdir():
                if grandchild.is_dir():
                    if (grandchild / "stage_2_train").is_dir() or (grandchild / "stage_2_train.csv").is_file():
                        return grandchild
    return base


def _link(src: Path, dst: Path):
    """Opret symlink, skip hvis det allerede findes."""
    if dst.exists() or dst.is_symlink():
        print(f"  [skip] {dst} findes allerede")
        return
    if src.exists():
        os.symlink(src, dst)
        print(f"  [link] {src} -> {dst}")
    else:
        print(f"  [WARN] Kilde ikke fundet: {src}")


def main():
    project_root = Path(os.environ.get("PROJECT_ROOT", ".")).resolve()
    raw_dir = project_root / "data" / "rsna_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=== Downloader RSNA dataset via kagglehub ===")
    print("(Dette kan tage lang tid ved første kørsel)")
    dataset_path = kagglehub.competition_download("rsna-intracranial-hemorrhage-detection")
    dataset_path = Path(dataset_path)
    print(f"kagglehub returnerede: {dataset_path}")

    # Find den faktiske data-rod (kan være nested)
    data_root = _find_data_root(dataset_path)
    print(f"Data-rod fundet: {data_root}")
    print(f"  Indhold: {sorted(p.name for p in data_root.iterdir())}")

    # --- Symlink DICOM-mapper ---
    # Kaggle har kun stage_2 data. Vi linker:
    #   stage_2_train  ->  RSNA_TRAIN_DIR  (bruges til træning)
    #   stage_2_test   ->  RSNA_TEST_DIR   (bruges til test-inference)
    _link(data_root / "stage_2_train", raw_dir / "stage_2_train")
    _link(data_root / "stage_2_test",  raw_dir / "stage_2_test")

    # --- Symlink CSV-filer ---
    for csv_name in ["stage_2_train.csv", "stage_2_sample_submission.csv"]:
        _link(data_root / csv_name, raw_dir / csv_name)

    print("Kaggle data klar.")


if __name__ == "__main__":
    main()
