import argparse
import os

from dataloader.dataloader import VOLTUPdata
from Model.Model import PINN

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_args():
    parser = argparse.ArgumentParser('Hyper Parameters for VOLTUP dataset')
    parser.add_argument('--data', type=str, default='VOLTUP', help='XJTU, HUST, MIT, TJU, VOLTUP')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max,z-score')

    # scheduler related
    parser.add_argument('--epochs', type=int, default=200, help='epoch')
    parser.add_argument('--early_stop', type=int, default=20, help='early stop')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epoch')
    parser.add_argument('--warmup_lr', type=float, default=2e-3, help='warmup lr')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-4, help='final lr')
    parser.add_argument('--lr_F', type=float, default=5e-4, help='lr of F')

    # model related
    parser.add_argument('--F_layers_num', type=int, default=3, help='the layers num of F')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='the hidden dim of F')

    # loss related
    parser.add_argument('--alpha', type=float, default=0.5, help='loss = l_data + alpha * l_PDE + beta * l_physics')
    parser.add_argument('--beta', type=float, default=0.2, help='loss = l_data + alpha * l_PDE + beta * l_physics')

    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log dir, if None, do not save')
    parser.add_argument('--save_folder', type=str, default='results/VOLTUP results', help='save folder')
    parser.add_argument('--data_root', type=str, default='data/VOLTUP DATA', help='VOLTUP data folder')

    return parser.parse_args()


def load_VOLTUP_data(args, small_sample=None, test_ratio=0.2):
    data = VOLTUPdata(root=args.data_root, args=args)

    files = sorted([f for f in os.listdir(args.data_root) if f.lower().endswith('.csv')])
    if len(files) == 0:
        raise FileNotFoundError(f'No CSV files found in {args.data_root}')

    split_index = int(len(files) * (1 - test_ratio))
    train_files = files[:split_index]
    test_files = files[split_index:]

    train_list = [os.path.join(args.data_root, f) for f in train_files]
    test_list = [os.path.join(args.data_root, f) for f in test_files]

    if small_sample is not None:
        train_list = train_list[:small_sample]

    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {
        'train': trainloader['train_2'],
        'valid': trainloader['valid_2'],
        'test': testloader['test_3']
    }
    return dataloader


def main():
    args = get_args()
    for e in range(10):
        args.save_folder = os.path.join('results/VOLTUP results', f'Experiment{e + 1}')
        os.makedirs(args.save_folder, exist_ok=True)

        dataloader = load_VOLTUP_data(args)
        pinn = PINN(args)
        pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])


def small_sample():
    args = get_args()
    for n in [1, 2, 3, 4]:
        for e in range(10):
            args.save_folder = os.path.join('results/VOLTUP results (small sample {})'.format(n), f'Experiment{e + 1}')
            args.batch_size = 128
            os.makedirs(args.save_folder, exist_ok=True)

            dataloader = load_VOLTUP_data(args, small_sample=n)
            pinn = PINN(args)
            pinn.Train(trainloader=dataloader['train'], validloader=dataloader['valid'], testloader=dataloader['test'])


if __name__ == '__main__':
    main()
    # small_sample()
  