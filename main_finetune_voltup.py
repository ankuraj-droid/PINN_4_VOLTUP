"""
Fine-tune the combined (MIT+HUST) PINN onto VoltUp/AmpUp cells.

Transfer recipe (matches the paper's domain-adaptation design):
  * dynamical_F  -- the learned degradation DYNAMICS (du/dt = F(...)). This is the
                    universal physics, shared across chemistries. It is FROZEN.
                    (In the plan's wording this is "G", the governing dynamics.)
  * solution_u   -- the feature -> SOH mapping. This is cell/fleet specific and is
                    FINE-TUNED on the target data.
                    (In the plan's wording this is "F", the predictor.)

So: load combined source model -> freeze dynamical_F -> adapt solution_u on VoltUp.

VoltUp cells are 30 Ah nominal (vs 1.1 Ah for MIT/HUST). Per-cell feature normalisation
puts every cell's inputs on a common scale, but a real domain shift remains (different
chemistry/format/operating regime) — which is exactly why we FINE-TUNE solution_u rather
than use the source model directly. The frozen dynamical_F supplies a physics prior; the
adaptation re-fits the feature->SOH map to the VoltUp domain.

Run (inside the torch env), once data/VOLTUP DATA/ has CSVs:
    /opt/anaconda3/envs/voltup_ml/bin/python main_finetune_voltup.py \
        --pretrain_model "results/COMBINED results/Experiment1/model.pth" \
        --target VOLTUP

To validate the freeze/fine-tune WIRING before VoltUp data exists, use a stand-in
target carved from held-out MIT cells (proves only solution_u changes, F stays fixed):
    ... main_finetune_voltup.py --pretrain_model <model.pth> --target STANDIN_MIT
"""
import argparse
import os

import numpy as np
import torch

from dataloader.dataloader import DF, VOLTUPdata
from Model.Model import PINN
from utils.util import AverageMeter, eval_metrix, write_to_txt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AdaPINN(PINN):
    """PINN that adapts only solution_u while dynamical_F is frozen."""

    def __init__(self, args):
        super(AdaPINN, self).__init__(args)
        self.load_model(model_path=args.pretrain_model)
        self.ada_optimizer = torch.optim.Adam(self.solution_u.parameters(), lr=args.adaptation_lr)

    def freeze_dynamics(self):
        frozen = 0
        for p in self.dynamical_F.parameters():
            p.requires_grad = False
            frozen += p.numel()
        trainable = sum(p.numel() for p in self.solution_u.parameters() if p.requires_grad)
        self.logger.info(f'[freeze] dynamical_F frozen params={frozen}; '
                         f'solution_u trainable params={trainable}')
        return frozen, trainable

    def adaptation_one_epoch(self, epoch, dataloader):
        self.solution_u.train()
        l1, l2, l3 = AverageMeter(), AverageMeter(), AverageMeter()
        for it, (x1, x2, y1, y2) in enumerate(dataloader):
            x1, x2, y1, y2 = x1.to(device), x2.to(device), y1.to(device), y2.to(device)
            u1, f1 = self.forward(x1)
            u2, f2 = self.forward(x2)
            loss1 = 0.5 * self.loss_func(u1, y1) + 0.5 * self.loss_func(u2, y2)
            f_target = torch.zeros_like(f1)
            loss2 = 0.5 * self.loss_func(f1, f_target) + 0.5 * self.loss_func(f2, f_target)
            loss3 = self.relu(torch.mul(u2 - u1, y1 - y2)).sum()
            loss = loss1 + self.alpha * loss2 + self.beta * loss3

            self.ada_optimizer.zero_grad()
            loss.backward()
            self.ada_optimizer.step()
            l1.update(loss1.item()); l2.update(loss2.item()); l3.update(loss3.item())
        return l1.avg, l2.avg, l3.avg

    def Adaptation(self, trainloader, validloader=None, testloader=None):
        self.freeze_dynamics()
        min_valid_mse, valid_mse, early_stop = 10, 10, 0
        for e in range(1, self.args.adaptation_epochs + 1):
            early_stop += 1
            loss1, loss2, loss3 = self.adaptation_one_epoch(e, trainloader)
            self.logger.info('[Adapt] epoch:{}, data:{:.6f} PDE:{:.6f} phys:{:.6f}'.format(
                e, loss1, loss2, loss3))
            if validloader is not None:
                valid_mse = self.Valid(validloader)
                self.logger.info('[Valid] epoch:{}, MSE:{}'.format(e, valid_mse))
            if valid_mse < min_valid_mse and testloader is not None:
                min_valid_mse = valid_mse
                true_label, pred_label = self.Test(testloader)
                MAE, MAPE, MSE, RMSE = eval_metrix(true_label, pred_label)
                self.logger.info('[Test] MSE:{:.8f} MAE:{:.6f} MAPE:{:.6f} RMSE:{:.6f}'.format(
                    MSE, MAE, MAPE, RMSE))
                early_stop = 0
                self.best_model = {'solution_u': self.solution_u.state_dict(),
                                   'dynamical_F': self.dynamical_F.state_dict()}
                if self.args.save_folder is not None:
                    np.save(os.path.join(self.args.save_folder, 'true_label.npy'), true_label)
                    np.save(os.path.join(self.args.save_folder, 'pred_label.npy'), pred_label)
            if self.args.early_stop is not None and early_stop > self.args.early_stop:
                self.logger.info('early stop at epoch {}'.format(e))
                break
        self.clear_logger()
        if self.args.save_folder is not None and self.best_model is not None:
            torch.save(self.best_model, os.path.join(self.args.save_folder, 'finetune model.pth'))


# --------------------------------------------------------------------------- #
# Target data loaders
# --------------------------------------------------------------------------- #
def _voltup_split(files):
    """Whole-cell train/valid/test split (~60/20/20) over sorted VoltUp cell files."""
    n = len(files)
    n_test = max(1, n // 5)
    n_valid = max(1, n // 5)
    test = files[-n_test:]
    valid = files[-(n_test + n_valid):-n_test] or files[:1]
    train = files[:-(n_test + n_valid)] or files[:1]
    return train, valid, test


def load_voltup_target(args):
    """VoltUp cells (30 Ah nominal). Whole-cell train/valid/test split (no intra-cell leak)."""
    root = args.data_root
    files = sorted(f for f in os.listdir(root) if f.lower().endswith('.csv'))
    if not files:
        raise FileNotFoundError(
            f'No CSVs in {root}. VoltUp/AmpUp data not collected yet — '
            f'use --target STANDIN_MIT to validate the wiring.')
    data = VOLTUPdata(root=root, args=args)
    nominal = data.nominal_capacity
    train_f, valid_f, test_f = _voltup_split(files)
    p = lambda fs: [os.path.join(root, f) for f in fs]
    return {'train': data.make_loader(p(train_f), nominal, shuffle=True),
            'valid': data.make_loader(p(valid_f), nominal, shuffle=False),
            'test': data.make_loader(p(test_f), nominal, shuffle=False)}


def load_standin_mit_target(args):
    """Stand-in 'new fleet' carved from held-out MIT cells (1.1 Ah).

    WIRING CHECK ONLY. Its test cells are the combined model's held-out MIT test cells,
    so the resulting MAPE is NOT an independent generalization number — it exists purely
    to prove the freeze/fine-tune machinery runs end-to-end (dynamical_F stays fixed,
    solution_u adapts). NOT a substitute for real VoltUp data.
    """
    print('[standin] WIRING CHECK ONLY — MAPE here is not an independent held-out metric.')
    bdir = os.path.join('data/MIT data', '2018-04-12')
    cells = sorted(f for f in os.listdir(bdir) if f.lower().endswith('.csv'))
    held = [os.path.join(bdir, f) for i, f in enumerate(cells) if i % 5 == 0]
    train_list, valid_list, test_list = held[:-3], held[-3:-1], held[-1:]
    reader = DF(args)
    return {'train': reader.make_loader(train_list, 1.1, shuffle=True),
            'valid': reader.make_loader(valid_list, 1.1, shuffle=False),
            'test': reader.make_loader(test_list, 1.1, shuffle=False)}


def get_args():
    p = argparse.ArgumentParser('Fine-tune combined PINN onto VoltUp (freeze dynamical_F)')
    p.add_argument('--pretrain_model', type=str, required=True,
                   help='path to combined source model.pth (solution_u + dynamical_F)')
    p.add_argument('--target', type=str, default='VOLTUP', choices=['VOLTUP', 'STANDIN_MIT'])
    p.add_argument('--data_root', type=str, default='data/VOLTUP DATA')
    p.add_argument('--save_folder', type=str, default='results/FINETUNE voltup')

    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--normalization_method', type=str, default='min-max')

    # PINN/base args (needed to rebuild the architecture before loading weights)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--warmup_epochs', type=int, default=1)
    p.add_argument('--warmup_lr', type=float, default=2e-3)
    p.add_argument('--lr', type=float, default=1e-2)
    p.add_argument('--final_lr', type=float, default=2e-4)
    p.add_argument('--lr_F', type=float, default=5e-4)
    p.add_argument('--F_layers_num', type=int, default=3)
    p.add_argument('--F_hidden_dim', type=int, default=60)
    p.add_argument('--alpha', type=float, default=0.7)
    p.add_argument('--beta', type=float, default=0.2)

    # adaptation args
    p.add_argument('--adaptation_lr', type=float, default=4e-4)
    p.add_argument('--adaptation_epochs', type=int, default=100)
    p.add_argument('--early_stop', type=int, default=15)

    p.add_argument('--log_dir', type=str, default='logging.txt')
    return p.parse_args()


def main():
    args = get_args()
    os.makedirs(args.save_folder, exist_ok=True)

    target_loaders = (load_voltup_target(args) if args.target == 'VOLTUP'
                      else load_standin_mit_target(args))

    model = AdaPINN(args)

    # 1) source-only performance on the target (before any adaptation)
    true_label, pred_label = model.Test(target_loaders['test'])
    MAE0, MAPE0, MSE0, RMSE0 = eval_metrix(true_label, pred_label)
    print(f'[source-only on {args.target}] MAPE={MAPE0 * 100:.3f}%  MAE={MAE0:.5f}  RMSE={RMSE0:.5f}')

    # snapshot frozen-net weights to prove they don't move
    f_before = torch.cat([p.detach().flatten() for p in model.dynamical_F.parameters()]).clone()
    u_before = torch.cat([p.detach().flatten() for p in model.solution_u.parameters()]).clone()

    # 2) adapt (freeze dynamical_F, fine-tune solution_u)
    model.Adaptation(trainloader=target_loaders['train'],
                     validloader=target_loaders['valid'],
                     testloader=target_loaders['test'])

    f_after = torch.cat([p.detach().flatten() for p in model.dynamical_F.parameters()])
    u_after = torch.cat([p.detach().flatten() for p in model.solution_u.parameters()])
    f_delta = (f_after - f_before).abs().max().item()
    u_delta = (u_after - u_before).abs().max().item()
    print(f'[freeze check] dynamical_F max|Δ|={f_delta:.2e} (expect 0.0)   '
          f'solution_u max|Δ|={u_delta:.2e} (expect > 0)')

    # 3) post-adaptation performance (best model already reloaded inside Adaptation save path)
    if model.best_model is not None:
        model.solution_u.load_state_dict(model.best_model['solution_u'])
    true_label, pred_label = model.Test(target_loaders['test'])
    MAE1, MAPE1, MSE1, RMSE1 = eval_metrix(true_label, pred_label)
    print(f'[after fine-tune on {args.target}] MAPE={MAPE1 * 100:.3f}%  MAE={MAE1:.5f}  RMSE={RMSE1:.5f}')
    print(f'>>> MAPE {MAPE0 * 100:.3f}% -> {MAPE1 * 100:.3f}%')

    save_name = os.path.join(args.save_folder, args.log_dir)
    write_to_txt(save_name, f'target={args.target} source_MAPE={MAPE0:.6f} '
                            f'finetuned_MAPE={MAPE1:.6f} F_delta={f_delta:.3e} U_delta={u_delta:.3e}')


if __name__ == '__main__':
    main()
