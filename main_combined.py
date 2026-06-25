"""
Train the full PINN (solution_u + dynamical_F) on a CONCATENATED MIT + HUST
training set, then report MAPE on held-out cells from each dataset separately
and combined.

Both datasets are A123-class LFP cells with 1.1 Ah nominal capacity, so SOH
(= capacity / 1.1) is directly comparable and the cells can share one model.
Normalisation is per-cell (inside read_one_csv), so concatenating cells across
datasets is sound.

Held-out cells are split at the WHOLE-CELL level (a test cell's cycles never
appear in train/valid), so the reported MAPE has no cross-cycle leakage.

The checkpoint saved here (solution_u + dynamical_F) is the source model for the
VoltUp fine-tuning step (main_finetune_voltup.py), which freezes dynamical_F and
adapts solution_u.

Usage (run inside the torch env):
    /opt/anaconda3/envs/voltup_ml/bin/python main_combined.py --epochs 80
"""
import argparse
import json
import os

import numpy as np

from dataloader.dataloader import DF
from Model.Model import PINN
from utils.util import eval_metrix

HUST_ROOT = 'data/HUST data'
MIT_ROOT = 'data/MIT data'

# Same held-out HUST cells the original main_HUST.py uses.
HUST_TEST_ID = ['1-4', '1-8', '2-4', '2-8', '3-4', '3-8', '4-4', '4-8',
                '5-4', '5-7', '6-4', '6-8', '7-4', '7-8', '8-4', '8-8',
                '9-4', '9-8', '10-4', '10-8']

MIT_BATCHES = ['2017-05-12', '2017-06-30', '2018-04-12']
# Hold out every 5th cell (sorted) per MIT batch -> ~20% of cells, whole-cell, deterministic.
MIT_TEST_EVERY = 5


def hust_split():
    """Return (train_paths, test_paths) for HUST, split by whole cell."""
    train, test = [], []
    for f in sorted(os.listdir(HUST_ROOT)):
        if not f.lower().endswith('.csv'):
            continue
        stem = f[:-4]
        path = os.path.join(HUST_ROOT, f)
        (test if stem in HUST_TEST_ID else train).append(path)
    return train, test


def mit_split():
    """Return (train_paths, test_paths) for MIT, split by whole cell, ~20% held out per batch."""
    train, test = [], []
    for batch in MIT_BATCHES:
        bdir = os.path.join(MIT_ROOT, batch)
        if not os.path.isdir(bdir):
            continue
        cells = sorted(f for f in os.listdir(bdir) if f.lower().endswith('.csv'))
        for i, f in enumerate(cells):
            path = os.path.join(bdir, f)
            (test if i % MIT_TEST_EVERY == 0 else train).append(path)
    return train, test


def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for combined MIT+HUST training')
    parser.add_argument('--data', type=str, default='COMBINED')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=80, help='epoch')
    parser.add_argument('--early_stop', type=int, default=15, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=2e-3, help='warmup lr')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-4, help='final lr')
    parser.add_argument('--lr_F', type=float, default=5e-4, help='lr of F')

    # model related
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    # loss related
    parser.add_argument('--alpha', type=float, default=0.5, help='loss = l_data + alpha*l_PDE + beta*l_physics')
    parser.add_argument('--beta', type=float, default=0.2, help='loss = l_data + alpha*l_PDE + beta*l_physics')

    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log file name')
    parser.add_argument('--save_folder', type=str, default='results/COMBINED results/Experiment1', help='save folder')

    # debugging / scaling
    parser.add_argument('--max_train_cells', type=int, default=None,
                        help='cap train cells (per dataset) for quick timing probes')
    parser.add_argument('--experiments', type=int, default=1, help='number of repeated experiments')
    return parser.parse_args()


def _split_train_valid(cells, valid_frac=0.2):
    """Whole-cell train/valid split (cells are deterministically ordered upstream)."""
    k = max(1, int(round(len(cells) * valid_frac)))
    return cells[k:], cells[:k]  # (train cells, valid cells)


def build_loaders(args):
    """Concatenate HUST+MIT train cells; hold out whole cells for valid and for test."""
    hust_train, hust_test = hust_split()
    mit_train, mit_test = mit_split()

    if args.max_train_cells is not None:
        hust_train = hust_train[:args.max_train_cells]
        mit_train = mit_train[:args.max_train_cells]
        hust_test = hust_test[:max(1, args.max_train_cells // 4)]
        mit_test = mit_test[:max(1, args.max_train_cells // 4)]

    # Whole-cell train/valid split per dataset (so valid stays representative of both,
    # and no cell's cycles appear in both train and valid).
    hust_tr, hust_va = _split_train_valid(hust_train)
    mit_tr, mit_va = _split_train_valid(mit_train)
    train_cells = hust_tr + mit_tr
    valid_cells = hust_va + mit_va

    reader = DF(args)  # base reader; both datasets normalised at nominal 1.1 Ah per cell

    print(f'[data] train cells: HUST={len(hust_tr)} MIT={len(mit_tr)} (total {len(train_cells)})')
    print(f'[data] valid cells: HUST={len(hust_va)} MIT={len(mit_va)} (total {len(valid_cells)})')
    print(f'[data] test  cells: HUST={len(hust_test)} MIT={len(mit_test)}')

    NOMINAL = 1.1
    loaders = {
        'train': reader.make_loader(train_cells, NOMINAL, shuffle=True),
        'valid': reader.make_loader(valid_cells, NOMINAL, shuffle=False),
        'test_combined': reader.make_loader(hust_test + mit_test, NOMINAL, shuffle=False),
        'test_hust': reader.make_loader(hust_test, NOMINAL, shuffle=False),
        'test_mit': reader.make_loader(mit_test, NOMINAL, shuffle=False),
    }
    return loaders


def evaluate(pinn, loaders):
    """Load best weights and report MAPE on each held-out test set."""
    pinn.solution_u.load_state_dict(pinn.best_model['solution_u'])
    pinn.dynamical_F.load_state_dict(pinn.best_model['dynamical_F'])
    out = {}
    for name in ['test_hust', 'test_mit', 'test_combined']:
        true_label, pred_label = pinn.Test(loaders[name])
        # correct sklearn order: (y_true, y_pred) so MAPE divides by true SOH
        MAE, MAPE, MSE, RMSE = eval_metrix(true_label, pred_label)
        out[name] = {'MAE': float(MAE), 'MAPE': float(MAPE),
                     'MSE': float(MSE), 'RMSE': float(RMSE), 'n': int(len(true_label))}
    return out


def run_one(args):
    os.makedirs(args.save_folder, exist_ok=True)
    loaders = build_loaders(args)
    pinn = PINN(args)
    pinn.Train(trainloader=loaders['train'],
               validloader=loaders['valid'],
               testloader=loaders['test_combined'])
    metrics = evaluate(pinn, loaders)

    print('\n===== held-out MAPE (combined MIT+HUST model) =====')
    for name, m in metrics.items():
        print(f'{name:14s}  MAPE={m["MAPE"]*100:.3f}%  MAE={m["MAE"]:.5f}  '
              f'RMSE={m["RMSE"]:.5f}  (n={m["n"]})')

    with open(os.path.join(args.save_folder, 'metrics.json'), 'w') as fh:
        json.dump(metrics, fh, indent=2)
    return metrics


def main():
    args = get_args()
    base = args.save_folder
    all_metrics = []
    for e in range(args.experiments):
        args.save_folder = base if args.experiments == 1 else f'{base.rstrip("/")}_{e + 1}'
        all_metrics.append(run_one(args))
    if len(all_metrics) > 1:
        # average MAPE across experiments
        for name in all_metrics[0]:
            mapes = [m[name]['MAPE'] for m in all_metrics]
            print(f'[avg over {len(all_metrics)}] {name}: MAPE={np.mean(mapes)*100:.3f}% '
                  f'(std {np.std(mapes)*100:.3f}%)')


if __name__ == '__main__':
    main()
