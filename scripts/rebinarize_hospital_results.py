#!/usr/bin/env python3
"""Re-apply new RSNA-derived thresholds to the hospital cohort's cached
per-slice probabilities (no GPU re-run needed).

The hospital evaluation directory contains one folder per patient with
`results.json`. The JSON stores per-slice probabilities for all 6
hemorrhage subtypes and the ischemic class — alongside the threshold
that was used when the pipeline first produced the file. This script
rebinarizes those probabilities with the new threshold set and writes
out a parallel directory tree of updated JSONs plus a cohort-level
comparison CSV and summary.

What we recompute per patient (per hemorrhage subtype):
  * n_positive_slices
  * positive_slice_indices
  * patient_positive
  * max_probability is unchanged (it's a property of the probabilities,
    not the threshold)

Ischemic predictions are untouched because we have not derived a new
threshold for that branch (it is still 0.5 in the deployed pipeline).

Usage (run on hospital where the data lives):

  python scripts/rebinarize_hospital_results.py \\
      --input-root /home/khma/ct-brain-pipeline/resultater \\
      --thresholds /tmp/thresholds_rsna_val.json \\
      --output-root /tmp/resultater_new_thresholds

Output:
  /tmp/resultater_new_thresholds/
      <patient>/results.json                  # updated per-patient JSON
      cohort_comparison.csv                   # one row per (patient, class)
      cohort_summary.json                     # totals + discordance counts
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HEMORRHAGE_CLASSES = ["any", "epidural", "intraparenchymal",
                      "intraventricular", "subarachnoid", "subdural"]


def rebinarize_patient(data: dict, new_thr: dict) -> tuple[dict, dict]:
    """Return (updated_data, per_class_diff) for one patient.

    per_class_diff is a dict with one entry per hemorrhage subtype
    capturing the old vs new flag and slice-count changes.
    """
    out = json.loads(json.dumps(data))  # deep copy
    slices = out.get("slices", [])

    # Per-class recompute from the per-slice probabilities.
    per_class = {}
    for cls in HEMORRHAGE_CLASSES:
        thr = new_thr[cls]
        probs = []
        positive_idx = []
        for s in slices:
            p = s["results"]["hemorrhage"][cls]["probability"]
            probs.append(p)
            is_pos = p >= thr
            s["results"]["hemorrhage"][cls]["threshold"] = thr
            s["results"]["hemorrhage"][cls]["positive"] = bool(is_pos)
            if is_pos:
                positive_idx.append(s["slice_index"])
        max_p = max(probs) if probs else 0.0
        per_class[cls] = {
            "max_probability": round(float(max_p), 4),
            "n_positive_slices": len(positive_idx),
            "positive_slice_indices": positive_idx,
            "patient_positive": len(positive_idx) > 0,
        }

    # Rebuild the patient-level summary.
    out["hemorrhage_thresholds"] = {c: float(new_thr[c])
                                    for c in HEMORRHAGE_CLASSES}
    any_pos = per_class["any"]["patient_positive"]
    out["patient_diagnosis"]["hemorrhage"]["patient_positive"] = any_pos
    out["patient_diagnosis"]["hemorrhage"]["n_positive_slices"] = \
        per_class["any"]["n_positive_slices"]
    out["patient_diagnosis"]["hemorrhage"]["subtypes"] = per_class

    # Build the diff record from old summary (which is still in `data`).
    old_summary = data["patient_diagnosis"]["hemorrhage"]["subtypes"]
    diff = {}
    for cls in HEMORRHAGE_CLASSES:
        old = old_summary.get(cls, {})
        new = per_class[cls]
        diff[cls] = {
            "max_probability": new["max_probability"],
            "old_threshold": data["hemorrhage_thresholds"].get(cls),
            "new_threshold": new_thr[cls],
            "old_patient_positive": bool(old.get("patient_positive", False)),
            "new_patient_positive": new["patient_positive"],
            "old_n_positive_slices": int(old.get("n_positive_slices", 0)),
            "new_n_positive_slices": new["n_positive_slices"],
        }
    return out, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True,
                    help="Directory with one subfolder per patient")
    ap.add_argument("--thresholds", type=Path, required=True,
                    help="Path to thresholds_rsna_val.json")
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    with open(args.thresholds) as f:
        thr_doc = json.load(f)
    new_thr = thr_doc["youden_thresholds"]
    print("Using NEW Youden thresholds (from RSNA val OOF):")
    for c in HEMORRHAGE_CLASSES:
        print(f"  {c:18s} {new_thr[c]:.4f}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    patients = sorted([p for p in args.input_root.iterdir() if p.is_dir()],
                      key=lambda p: int(p.name) if p.name.isdigit() else p.name)
    print(f"\nProcessing {len(patients)} patient folders ...")

    rows = []
    cohort_summary = {c: {"old_positive_patients": 0,
                          "new_positive_patients": 0,
                          "newly_positive": 0,
                          "newly_negative": 0,
                          "unchanged_positive": 0,
                          "unchanged_negative": 0} for c in HEMORRHAGE_CLASSES}

    for pdir in patients:
        rj = pdir / "results.json"
        if not rj.exists():
            print(f"  [{pdir.name}] no results.json — skipped")
            continue
        with open(rj) as f:
            data = json.load(f)
        updated, diff = rebinarize_patient(data, new_thr)
        out_dir = args.output_root / pdir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "results.json", "w") as f:
            json.dump(updated, f, indent=2)

        # Cohort stats
        for cls in HEMORRHAGE_CLASSES:
            d = diff[cls]
            old_pos, new_pos = d["old_patient_positive"], d["new_patient_positive"]
            if old_pos: cohort_summary[cls]["old_positive_patients"] += 1
            if new_pos: cohort_summary[cls]["new_positive_patients"] += 1
            if not old_pos and new_pos:
                cohort_summary[cls]["newly_positive"] += 1
            elif old_pos and not new_pos:
                cohort_summary[cls]["newly_negative"] += 1
            elif old_pos and new_pos:
                cohort_summary[cls]["unchanged_positive"] += 1
            else:
                cohort_summary[cls]["unchanged_negative"] += 1
            rows.append({
                "patient_id": pdir.name,
                "patient_label": data.get("patient_metadata", {})
                                     .get("PatientID", "Unknown"),
                "klass": cls,
                "max_probability": d["max_probability"],
                "old_threshold": d["old_threshold"],
                "new_threshold": d["new_threshold"],
                "old_positive": int(d["old_patient_positive"]),
                "new_positive": int(d["new_patient_positive"]),
                "old_n_positive_slices": d["old_n_positive_slices"],
                "new_n_positive_slices": d["new_n_positive_slices"],
                "flag_changed": int(d["old_patient_positive"]
                                    != d["new_patient_positive"]),
            })

    # Cohort comparison CSV
    csv_path = args.output_root / "cohort_comparison.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nWrote {csv_path} ({len(rows)} rows = {len(patients)} patients × "
          f"{len(HEMORRHAGE_CLASSES)} classes)")

    # Cohort summary JSON
    summary_path = args.output_root / "cohort_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "n_patients": len(patients),
            "new_thresholds": {c: float(new_thr[c]) for c in HEMORRHAGE_CLASSES},
            "per_class": cohort_summary,
        }, f, indent=2)
    print(f"Wrote {summary_path}")

    print("\n=== Cohort-level changes (38 patients) ===")
    print(f"  {'class':<18} {'old +':>6} {'new +':>6} "
          f"{'new+':>6} {'new-':>6} {'unchg+':>7} {'unchg-':>7}")
    for cls in HEMORRHAGE_CLASSES:
        s = cohort_summary[cls]
        print(f"  {cls:<18} {s['old_positive_patients']:>6} "
              f"{s['new_positive_patients']:>6} "
              f"{s['newly_positive']:>6} {s['newly_negative']:>6} "
              f"{s['unchanged_positive']:>7} {s['unchanged_negative']:>7}")


if __name__ == "__main__":
    main()
