import argparse
import dataclasses
import json
import random
import shutil

import matplotlib
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from trainers.classification import site_classification
from configurations.configs import *
from datasets.cc359_dataset import CC359Ds
from datasets.msm_dataset import MultiSiteMri
from dpipe.io import load
from models.unet import UNet2D
from utils.sort_ds import create_sorted_dataset
from torch.utils import data

from trainers.curriculum import curriculum
from utils.utils import load_model


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")

    parser.add_argument("--num-workers", type=int, default=1,
                        help="number of workers for multithread dataloading.")
    # lr params
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Regularisation parameter for L2-loss.")

    parser.add_argument("--random-seed", type=int, default=1234,
                        help="Random seed to have reproducible results.")

    parser.add_argument("--num-classes", type=int, default=2,
                        help="Number of classes to predict (including background).")

    parser.add_argument("--gpu", type=int, default=6,
                        help="choose gpu device.")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--target", type=int, default=2)
    parser.add_argument("--algo", type=str)
    parser.add_argument('--exp_name', default='')
    parser.add_argument('--msm', action='store_true')
    parser.add_argument("--mode", type=str, default='pretrain', help='pretrain or clustering_finetune')

    return parser.parse_args()


def main(args):
    if args.msm:
        args.source = args.target
        print(f"changed source to {args.target}")
        if args.mode == 'clustering_finetune':
            config = MsmConfigFinetuneClustering()
        elif args.mode == 'pretrain':
            config = MsmPretrainConfig()
        else:
            config = MsmConfigFinetuneClustering()

    else:
        if 'debug' in args.exp_name:
            config = DebugConfigCC359()
        elif args.mode == 'clustering_finetune' or args.mode == "pseudo_labeling" or args.mode == "pseudo_no_cluster":
            config = CC359ConfigFinetuneClustering()
        elif args.mode == 'pretrain':
            config = CC359ConfigPretrain()
        else:
            config = CC359ConfigFinetuneClustering()
    if args.exp_name == '':
        args.exp_name = args.mode

    cudnn.enabled = True
    model = UNet2D(config.n_channels, n_chans_out=config.n_chans_out)
    if args.mode != 'pretrain':
        if args.exp_name != '':
            config.exp_dir = Path(config.base_res_path) / f'source_{args.source}_target_{args.target}' / args.exp_name
        else:
            config.exp_dir = Path(config.base_res_path) / f'source_{args.source}_target_{args.target}' / args.mode

        ckpt_path = Path(
            config.base_res_path) / f'source_{args.source}_target_{args.target}' / 'clustering_finetune' / 'best_model.pth'
        model = load_model(model, ckpt_path, config.msm)
        if config.msm:
            optimizer = optim.Adam(model.parameters(),
                                   lr=config.lr, weight_decay=args.weight_decay)
        else:
            optimizer = optim.SGD(model.parameters(),
                                  lr=config.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
            
    if config.exp_dir.exists():
        shutil.rmtree(config.exp_dir)

    config.exp_dir.mkdir(parents=True, exist_ok=True)
    json.dump(dataclasses.asdict(config), open(config.exp_dir / 'config.json', 'w'))
    model.train()
    if not torch.cuda.is_available():
        print('training on cpu')
        args.gpu = 'cpu'
        config.parallel_model = False
        torch.cuda.manual_seed_all(args.random_seed)

    model.to(args.gpu)
    if config.parallel_model:
        model = torch.nn.DataParallel(model, device_ids=[6,7])
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if config.msm:
        assert args.source == args.target
        source_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.source}t/train_ids.json'))
        target_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.target}/train_ids.json'))
        val_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.target}/val_ids.json'), yield_id=True,
                              test=True)
        val_ds_source = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.source}t/val_ids.json'), yield_id=True,
                                     test=True)
        test_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.target}/test_ids.json'), yield_id=True,
                               test=True)
    else:
        source_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.source}/train_ids.json')[:config.data_len],
                            site=args.source)
        target_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.target}/train_ids.json')[:config.data_len],
                            site=args.target)
        val_ds = val_ds_source = test_ds = None
        val_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.target}/test_ids.json'), site=args.target,
                         yield_id=True, slicing_interval=1)
        val_ds_source = CC359Ds(load(f'{config.base_splits_path}/site_{args.source}/val_ids.json'), site=args.source,
                                yield_id=True, slicing_interval=1)
        test_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.target}/test_ids.json'), site=args.target,
                          yield_id=True, slicing_interval=1)

    trainloader = data.DataLoader(source_ds, batch_size=config.source_batch_size, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=config.drop_last)
    targetloader = data.DataLoader(target_ds, batch_size=config.target_batch_size, shuffle=True,
                                   num_workers=args.num_workers, pin_memory=True, drop_last=config.drop_last)

    if args.mode == "sort_ds":
        create_sorted_dataset(ckpt_path, args, config)
    elif args.mode == "classify":
        site_classification(ckpt_path, trainloader, targetloader, val_ds, test_ds, val_ds_source, args, config)
    elif args.mode == "curriculum":
        curriculum(ckpt_path, trainloader, targetloader, val_ds, test_ds, val_ds_source, args, config)


if __name__ == '__main__':
    main(get_arguments())