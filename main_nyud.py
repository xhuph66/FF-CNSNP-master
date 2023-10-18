#!/user/bin/python
# -*- encoding: utf-8 -*-

import os, sys
import argparse
import time
import datetime
import torch
from data.data_loader import MyDataLoader, NYUD_Loader


from models.models import Network

from models.optimizer import Optimizer

from torch.utils.data import DataLoader, sampler
from utils import Logger, Averagvalue, save_checkpoint
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
from configs import Config
from train import train
from test import test, multiscale_test


parser = argparse.ArgumentParser(description='Mode Selection')
parser.add_argument('--mode', default='train', type=str, choices={"train", "test"},
                    help="Setting models for training or testing")
args = parser.parse_args()

cfg = Config()

cfg.data = "nyud"
cfg.resume = "./pretrained/{}.pth".format(cfg.data)
cfg.dataset = "./data/{}".format(cfg.data)
cfg.save_pth = join("./output", cfg.data)


if cfg.gpu:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

THIS_DIR = abspath(dirname(__file__))
TMP_DIR = join(THIS_DIR, cfg.save_pth)

if not isdir(TMP_DIR):
    os.makedirs(TMP_DIR)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch Classification Model Training')
    parser.add_argument(
        '--work-dir',
        type=str,
        help='path for get training config and saving log/models')
    parser.add_argument(
        '--local_rank',
        type=int,
        default=0,
        help='LOCAL_PROCESS_RANK in DistributedDataParallel model')

    return parser.parse_args()


def main():
    # model
    model = Network(cfg)
    print('=> Load model')

    model.to(device)
    print('=> Cuda used')

    test_dataset = MyDataLoader(root=cfg.dataset, split="test")
    test_loader = DataLoader(test_dataset, batch_size=1,
                             num_workers=100, shuffle=False)

    if args.mode == "test":
        assert isfile(cfg.resume), "No checkpoint is found at '{}'".format(cfg.resume)

        model.load_checkpoint()
        test(cfg, model, test_loader, save_dir=join(TMP_DIR, "test", "sing_scale_test"))

        if cfg.multi_aug:
            multiscale_test(model, test_loader, save_dir=join(TMP_DIR, "test", "multi_scale_test"))

    else:
        train_dataset = MyDataLoader(root=cfg.dataset, split="train", transform=cfg.aug)
        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size,
                                  num_workers=100, drop_last=True, shuffle=True)

        print(model)
        model.init_weight()

        if cfg.resume:
            model.load_checkpoint()

        model.train()

        # optimizer
        optim, scheduler = Optimizer(cfg)(model)

        # log
        log = Logger(join(TMP_DIR, "%s-%d-log-nyud.txt" % ("sgd", cfg.lr)))
        sys.stdout = log

        train_loss = []
        train_loss_detail = []

        for epoch in range(0, cfg.max_epoch):
            tr_avg_loss, tr_detail_loss = train(cfg,
                                                train_loader, model, optim, scheduler, epoch,
                                                save_dir=join(TMP_DIR, "train-nyud-cffmci-v2", "epoch-%d-training-record-nyud" % epoch))

            test(cfg, model, test_loader, save_dir=join(TMP_DIR, "train-nyud-cffmci-v2", "epoch-%d-testing-record-nyud-view" % epoch))

            log.flush()

            train_loss.append(tr_avg_loss)
            train_loss_detail += tr_detail_loss

if __name__ == '__main__':
    main()
