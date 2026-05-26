# Hospital workflow — re-train hemorrhage CNN + SequenceModel

This subdirectory contains the SeuTao RSNA2019 reproduction code with
**metric instrumentation added** for capturing all training metrics
(advisor request). The training procedure itself is unchanged from
SeuTao's original — only logging was added.

## What gets re-trained

- 3 × CNN backbones (DenseNet121, DenseNet169, SE-ResNeXt101)
  × 5 folds × 80 epochs each
- 1 × SequenceModel (5 folds × 40 epochs)

Estimated wall-clock on RTX 5090: ~10-15 days sequential.

## First-time setup (run once on the hospital workstation)

```bash
cd rsna_training/

# 1. Create conda environment
conda env create -f environment.yml
conda activate rsna-seutao

# 2. Download data + pretrained weights
#    - Needs Kaggle credentials (~/.kaggle/kaggle.json)
#    - Needs to have joined the RSNA2019 competition on Kaggle
bash download_all.sh

# 3. Verify everything is in place
bash smoke_test.sh
```

After this, `data/` (RSNA labels + folds), `downloads/` (pretrained
weights from Google Drive), and PNG-converted DICOMs will exist locally.
None of these are tracked in git.

## Run the smoke test (10-15 min, do this FIRST)

```bash
bash run_full_training.sh --smoke
```

This trains 1 fold × 2 epochs × small subset. Inspect the output:

```bash
ls external/SeuTao_repo/2DNet/src/data_test/*_smoke/fold0/
# Expect: log.csv (with ~110 columns), epoch_000_*.npz, epoch_000_*.json,
#         train_metadata.json, best_epoch.json
```

Verify the CSV row has numbers in every column (no empty cells, no NaN).
If anything looks wrong, fix it BEFORE running the full pipeline — there
is only one shot at the full training.

## Run the full training (10-15 days)

```bash
# Run in background so SSH disconnects don't kill it
nohup bash run_full_training.sh > full_run.log 2>&1 &

# Monitor progress
tail -f full_run.log
tail -f external/SeuTao_repo/2DNet/src/data_test/DenseNet121_change_avg_256/fold0/log.csv
```

The script runs sequentially:
1. DenseNet121 — 5 folds × 80 epochs (~2 days)
2. DenseNet169 — same (~2-3 days)
3. SE-ResNeXt101 — same (~3-4 days)
4. `predict.py` × 3 backbones — generates predictions + features (~1-2 days)
5. `build_sequence_inputs.py` — assembles SM inputs (minutes)
6. `SequenceModel/main.py` — 5 folds × 40 epochs (~5-8 hours)
7. `analyze_training.py` — generates learning curves + LaTeX tables

## Output locations (NOT in git)

- CNN logs: `external/SeuTao_repo/2DNet/src/data_test/<backbone>_change_avg_256/fold<N>/`
  - `log.csv` — flat metrics, one row per epoch
  - `epoch_<E>_metrics.json` — full structured metrics
  - `epoch_<E>_val_predictions.npz` — raw GT + predictions (for recomputing any metric post-hoc)
  - `train_metadata.json` — hyperparameters + dataset info
  - `best_epoch.json` — best val AUC + epoch
  - `model_best_<N>.pth` — checkpoint at best val AUC

- SM logs: `external/SeuTao_repo/FinalSubmission/version3_debug/fold<N>/`
- Plots + LaTeX: `training_analysis/`

## When training is done

Copy these back to a personal laptop (anonymized — no patient data
is involved in this training, only the public RSNA data):

```bash
# Compress all log/metrics output (NOT the model checkpoints)
tar -czf training_metrics_$(date +%Y%m%d).tar.gz \
  external/SeuTao_repo/2DNet/src/data_test/*/fold*/log.csv \
  external/SeuTao_repo/2DNet/src/data_test/*/fold*/*.json \
  external/SeuTao_repo/2DNet/src/data_test/*/fold*/*.npz \
  external/SeuTao_repo/FinalSubmission/version3_debug/fold*/log.csv \
  external/SeuTao_repo/FinalSubmission/version3_debug/fold*/*.json \
  training_analysis/
```

Then run `analyze_training.py` locally (or use the plots/tables that
were generated on the hospital machine).

## Key files modified for metric instrumentation

| File | What was added |
|---|---|
| `external/SeuTao_repo/shared/metric_logger.py` | NEW: centralized metrics + logging |
| `external/SeuTao_repo/2DNet/src/train.py` | Per-class metrics, gradient norms, best-model checkpoint, npz dumps |
| `external/SeuTao_repo/SequenceModel/main.py` | Same, plus separate SM1/SM2 tracking |
| `external/SeuTao_repo/analyze_training.py` | NEW: post-hoc plots + LaTeX tables |
| `run_full_training.sh` | NEW: orchestrates full re-training |

**Important:** the training procedure itself (hyperparameters, optimizer,
scheduler, data, augmentation, seeds) is unchanged. Only logging was added.
Results should be statistically reproducible with the previous run.

## If something goes wrong

- `external/SeuTao_repo/2DNet/src/data_test/*/log_legacy.csv` is the
  unchanged SeuTao-original CSV format — preserved as fallback.
- `external/SeuTao_repo/FinalSubmission/*/log.txt` is the unchanged
  SeuTao SM log.
- Even if the new instrumentation fails, the legacy logs + checkpoints
  are still produced.
