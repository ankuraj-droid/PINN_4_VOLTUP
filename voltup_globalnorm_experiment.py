"""
EXPERIMENT: does GLOBAL feature normalization (instead of the loader's per-cell min-max)
let the fine-tuned PINN actually RANK held-out VoltUp cells (R^2 > 0)?

Per-cell min-max scales every cell's features to [-1,1] independently, erasing the absolute
magnitude that distinguishes a strong cell from a weak one -> the model collapses to the fleet
mean on held-out cells. Here we fit ONE global min-max on the 16 physical features using TRAIN
cells only (no leakage) and apply it to all splits; the cycle-index "time" variable stays per-cell
(it means 'fraction of observed life'). Everything else (debiased labels, whole-cell split, freeze
dynamical_F, adapt solution_u) is identical to main_finetune_voltup.
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from dataloader.dataloader import DF
from main_finetune_voltup import AdaPINN, _voltup_split
from utils.util import eval_metrix

ROOT = 'data/VOLTUP DATA'
NOMINAL = 30.0


def build_args():
    a = argparse.Namespace()
    a.pretrain_model = 'results/COMBINED results/Experiment1/model.pth'
    a.save_folder = 'results/FINETUNE voltup globalnorm'
    a.log_dir = 'logging.txt'
    a.normalization_method = 'min-max'
    a.batch_size = 128
    a.epochs = 1; a.warmup_epochs = 1
    a.warmup_lr = 2e-3; a.lr = 1e-2; a.final_lr = 2e-4; a.lr_F = 5e-4
    a.F_layers_num = 3; a.F_hidden_dim = 60
    a.alpha = 0.7; a.beta = 0.2
    a.adaptation_lr = 4e-4; a.adaptation_epochs = 100; a.early_stop = 15
    return a


def load_cell_raw(path):
    """Cleaned raw frame: 16 features + 'cycle index' + 'capacity' (no normalization)."""
    reader = DF(argparse.Namespace(normalization_method='min-max', log_dir=None, save_folder=None))
    df = reader.read_one_csv(path, nominal_capacity=None)  # None -> no scaling/normalization
    return df


def cellset(files):
    return [load_cell_raw(os.path.join(ROOT, f)) for f in files]


def make_loader_global(dfs, fmin, frange, feat_cols, batch_size, shuffle):
    """Apply GLOBAL min-max to physical features, per-cell min-max to cycle index, build pairs."""
    X1, X2, Y1, Y2 = [], [], [], []
    for df in dfs:
        feats = df[feat_cols].to_numpy(float)
        feats = 2 * (feats - fmin) / frange - 1            # GLOBAL min-max -> [-1,1]
        ci = df['cycle index'].to_numpy(float)             # per-cell time -> [-1,1]
        ci = 2 * (ci - ci.min()) / (ci.max() - ci.min() + 1e-9) - 1
        x = np.column_stack([feats, ci])                   # 16 feats + cycle index = 17
        y = df['capacity'].to_numpy(float) / NOMINAL       # SOH label
        X1.append(x[:-1]); X2.append(x[1:]); Y1.append(y[:-1]); Y2.append(y[1:])
    X1 = np.concatenate(X1); X2 = np.concatenate(X2)
    Y1 = np.concatenate(Y1); Y2 = np.concatenate(Y2)
    ds = TensorDataset(torch.from_numpy(X1).float(), torch.from_numpy(X2).float(),
                       torch.from_numpy(Y1).float().view(-1, 1),
                       torch.from_numpy(Y2).float().view(-1, 1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def main():
    args = build_args()
    os.makedirs(args.save_folder, exist_ok=True)
    files = sorted(f for f in os.listdir(ROOT) if f.lower().endswith('.csv'))
    train_f, valid_f, test_f = _voltup_split(files)
    print(f'cells: train={len(train_f)} valid={len(valid_f)} test={len(test_f)}')

    feat_cols = None
    train_dfs = cellset(train_f); valid_dfs = cellset(valid_f); test_dfs = cellset(test_f)
    # feature columns = everything except cycle index and capacity (the 16 physical features)
    cols = list(train_dfs[0].columns)
    feat_cols = [c for c in cols if c not in ('cycle index', 'capacity')]
    assert len(feat_cols) == 16, feat_cols

    # GLOBAL min-max fit on TRAIN cells only
    allfeat = np.concatenate([df[feat_cols].to_numpy(float) for df in train_dfs])
    fmin = allfeat.min(0); fmax = allfeat.max(0); frange = np.where(fmax > fmin, fmax - fmin, 1.0)

    loaders = {
        'train': make_loader_global(train_dfs, fmin, frange, feat_cols, args.batch_size, True),
        'valid': make_loader_global(valid_dfs, fmin, frange, feat_cols, args.batch_size, False),
        'test': make_loader_global(test_dfs, fmin, frange, feat_cols, args.batch_size, False),
    }

    model = AdaPINN(args)
    t, p = model.Test(loaders['test'])
    MAE0, MAPE0, _, RMSE0 = eval_metrix(t, p)
    print(f'[source-only] MAPE={MAPE0*100:.3f}%')

    model.Adaptation(trainloader=loaders['train'], validloader=loaders['valid'], testloader=loaders['test'])

    if model.best_model is not None:
        model.solution_u.load_state_dict(model.best_model['solution_u'])
    t, p = model.Test(loaders['test'])
    t = t.ravel(); p = p.ravel()
    MAE1, MAPE1, _, RMSE1 = eval_metrix(t, p)
    base = np.mean(np.abs((t - t.mean()) / t)) * 100
    r2 = 1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2)
    corr = np.corrcoef(t, p)[0, 1]
    print(f'\n===== GLOBAL-NORM RESULT (n={len(t)}) =====')
    print(f'  source-only MAPE = {MAPE0*100:.2f}%')
    print(f'  fine-tuned  MAPE = {MAPE1*100:.2f}%   (per-cell-norm baseline was 9.30%)')
    print(f'  predict-mean MAPE= {base:.2f}%')
    print(f'  R^2              = {r2:.3f}   (per-cell-norm was ~0.00)')
    print(f'  corr(pred,true)  = {corr:.3f}')


if __name__ == '__main__':
    main()
