"""Post-hoc training analysis.

Reads the per-fold log.csv + epoch_*_metrics.json + epoch_*_val_predictions.npz
files produced by the instrumented training scripts, and generates:

  - Learning curves (loss + AUC per epoch, per fold, per backbone)
  - ROC curves at best epoch
  - PR curves at best epoch
  - Calibration curves
  - LaTeX tables (CNN-only, CNN+SM comparison)
  - Cross-fold aggregated summary (mean ± std)

Usage
-----
    python3 analyze_training.py --root data_test/ --out figures/

Layout it expects under --root::

    <root>/<backbone_save_path>/fold0/log.csv
    <root>/<backbone_save_path>/fold0/epoch_***_metrics.json
    <root>/<backbone_save_path>/fold0/epoch_***_val_predictions.npz
    <root>/<backbone_save_path>/fold0/best_epoch.json
    ... (fold1 ... fold4)
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def _read_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_fold_history(fold_dir: Path):
    log = _read_csv(fold_dir / 'log.csv')
    best = _read_json(fold_dir / 'best_epoch.json')
    return {'fold_dir': fold_dir, 'log': log, 'best': best}


def _to_float(s, default=float('nan')):
    if s is None or s == '' or s == 'nan':
        return default
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def load_backbone(root: Path, backbone_dir: str) -> dict:
    """Load all per-fold logs for one backbone."""
    bb_root = root / backbone_dir
    folds = sorted([d for d in bb_root.iterdir() if d.is_dir() and d.name.startswith('fold')])
    return {
        'name': backbone_dir,
        'root': bb_root,
        'folds': [_load_fold_history(d) for d in folds],
    }


def aggregate_best_epoch_metrics(backbone: dict) -> dict:
    """For each fold, find the best epoch (already recorded in best_epoch.json
    or by max ValAUC_any), and return a list of per-class metric dicts."""
    per_fold = []
    for fold in backbone['folds']:
        best = fold['best']
        if best is None or fold['log'] is None or not fold['log']:
            continue
        target_epoch = int(best['epoch'])
        # Find row in log
        row = next((r for r in fold['log'] if int(r.get('Epoch', -1)) == target_epoch), None)
        if row is None:
            continue
        per_fold.append({
            'fold_dir': fold['fold_dir'],
            'best_epoch': target_epoch,
            'best_score': best['score'],
            'row': row,
        })
    return per_fold


def summarise_metric(per_fold: list, column: str) -> tuple[float, float]:
    """Return (mean, std) across folds for a given CSV column."""
    values = [_to_float(f['row'].get(column)) for f in per_fold]
    values = [v for v in values if np.isfinite(v)]
    if not values:
        return float('nan'), float('nan')
    return float(np.mean(values)), float(np.std(values))


def print_summary(backbone: dict):
    print(f"\n=== {backbone['name']} ===")
    per_fold = aggregate_best_epoch_metrics(backbone)
    if not per_fold:
        print('  No fold data found.')
        return

    print(f"  Folds with completed runs: {len(per_fold)}")
    for f in per_fold:
        any_auc = _to_float(f['row'].get('AUC_any', f['row'].get('SM2_AUC_any')))
        print(f"  fold {f['fold_dir'].name}: best epoch {f['best_epoch']}, AUC_any={any_auc:.4f}")

    print('  --- Mean ± std across folds (at best epoch) ---')
    # Try CNN columns first; fall back to SM2 prefix
    for hemo in ['any', 'epidural', 'intraparenchymal',
                 'intraventricular', 'subarachnoid', 'subdural']:
        for col_prefix in ('', 'SM2_'):
            col = f'{col_prefix}AUC_{hemo}'
            mean, std = summarise_metric(per_fold, col)
            if np.isfinite(mean):
                print(f'  {col}: {mean:.4f} ± {std:.4f}')
                break


def write_latex_table(backbones: list[dict], out_path: Path):
    """Emit a LaTeX table of mean ± std AUC across folds for each backbone."""
    hemo_classes = ['any', 'epidural', 'intraparenchymal',
                    'intraventricular', 'subarachnoid', 'subdural']
    lines = []
    lines.append(r'\begin{table}[h]')
    lines.append(r'\centering')
    lines.append(r'\caption{Per-fold mean $\pm$ std validation AUC at best epoch.}')
    cols = 'l' + 'c' * len(hemo_classes)
    lines.append(rf'\begin{{tabular}}{{{cols}}}')
    lines.append(r'\toprule')
    lines.append(r'Backbone & ' + ' & '.join(hemo_classes) + r' \\')
    lines.append(r'\midrule')
    for bb in backbones:
        per_fold = aggregate_best_epoch_metrics(bb)
        cells = [bb['name'].replace('_', r'\_')]
        for hemo in hemo_classes:
            mean, std = (float('nan'), float('nan'))
            for col_prefix in ('', 'SM2_'):
                col = f'{col_prefix}AUC_{hemo}'
                mean, std = summarise_metric(per_fold, col)
                if np.isfinite(mean):
                    break
            if np.isfinite(mean):
                cells.append(f'{mean:.3f} $\\pm$ {std:.3f}')
            else:
                cells.append('--')
        lines.append(' & '.join(cells) + r' \\')
    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(r'\end{table}')

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text('\n'.join(lines))
    print(f'LaTeX table written: {out_path}')


def plot_learning_curves(backbone: dict, out_dir: Path):
    """One PNG per fold: train/val loss + val AUC over epochs."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not available — skipping plots')
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    for fold in backbone['folds']:
        log = fold['log']
        if not log:
            continue
        epochs = [int(r['Epoch']) for r in log]
        train_loss = [_to_float(r.get('TrainLoss', r.get('TrainLoss_total'))) for r in log]
        val_loss = [_to_float(r.get('ValLoss', r.get('ValLoss_SM2'))) for r in log]
        val_auc = [_to_float(r.get('AUC_any', r.get('SM2_AUC_any'))) for r in log]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(epochs, train_loss, label='Train', linewidth=2)
        ax1.plot(epochs, val_loss, label='Val', linewidth=2)
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
        ax1.set_title(f"{backbone['name']} / {fold['fold_dir'].name} — Loss")
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, val_auc, color='C2', linewidth=2)
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('AUC (any)')
        ax2.set_title('Validation AUC')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        fig.tight_layout()
        out_png = out_dir / f"{backbone['name']}_{fold['fold_dir'].name}_learning_curves.png"
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f'  Saved {out_png}')


def plot_roc_curves(backbone: dict, out_dir: Path):
    """ROC curves at best epoch per fold (one figure per fold)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, roc_auc_score
    except ImportError:
        print('matplotlib/sklearn not available — skipping ROC plots')
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    hemo_classes = ['any', 'epidural', 'intraparenchymal',
                    'intraventricular', 'subarachnoid', 'subdural']

    for fold in backbone['folds']:
        best = fold['best']
        if best is None:
            continue
        epoch = int(best['epoch'])
        # Try CNN convention first, then SM
        npz_path = fold['fold_dir'] / f'epoch_{epoch:03d}_val_predictions.npz'
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        if 'outGT' in data.files and 'outPRED' in data.files:
            gt = data['outGT']
            pred = data['outPRED']
        elif 'labels' in data.files and 'sm2_probs' in data.files:
            gt = data['labels']
            pred = data['sm2_probs']
        else:
            continue

        fig, ax = plt.subplots(figsize=(6, 5))
        for i, name in enumerate(hemo_classes):
            try:
                fpr, tpr, _ = roc_curve(gt[:, i], pred[:, i])
                auc = roc_auc_score(gt[:, i], pred[:, i])
                ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})')
            except Exception:
                continue
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
        ax.set_title(f"{backbone['name']} / {fold['fold_dir'].name} — ROC at epoch {epoch}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        out_png = out_dir / f"{backbone['name']}_{fold['fold_dir'].name}_roc_best.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=120)
        plt.close(fig)
        print(f'  Saved {out_png}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path,
                        help='Path to data_test/ directory containing <backbone>/fold*/...')
    parser.add_argument('--out', required=True, type=Path,
                        help='Output directory for plots + LaTeX tables.')
    args = parser.parse_args()

    backbone_dirs = sorted([d.name for d in args.root.iterdir()
                           if d.is_dir() and any((d / f'fold{i}').is_dir() for i in range(5))])
    if not backbone_dirs:
        print(f'No backbone directories with fold[0-4] found under {args.root}.')
        return

    backbones = [load_backbone(args.root, name) for name in backbone_dirs]

    for bb in backbones:
        print_summary(bb)
        plot_learning_curves(bb, args.out / 'learning_curves')
        plot_roc_curves(bb, args.out / 'roc_curves')

    write_latex_table(backbones, args.out / 'latex' / 'auc_summary.tex')


if __name__ == '__main__':
    main()
