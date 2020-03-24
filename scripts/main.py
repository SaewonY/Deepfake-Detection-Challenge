import gc
import os
import cv2
import sys
import time
import argparse
import numpy as np
import pandas as pd

import torch
from torch import nn, cuda
import torch.nn.functional as F
import torchvision
from model import build_model
from dataset import build_dataset
from utils import seed_everything
from train import train_model
from optimizer import build_optimizer, build_scheduler

import warnings
warnings.filterwarnings('ignore')


def arg_parser():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--PATH', type=str, default='../input')
    arg('--weight_path', type=str, default='../input/weights/resnext50_32x4d-7cdf4587.pth')
    arg('--train_df_path', type=str, default='../input/train_df.csv')
    arg('--valid_df_path', type=str, default='../input/valid_df.csv')
    arg('--n_epochs', type=int, default=5)
    arg('--model', type=str, default='resnet50')
    arg('--model_path', type=str)
    arg('--batch_size', type=int, default=32)
    arg('--num_workers', type=int, default=6)
    arg('--optimizer', type=str, default='Adam')
    arg('--scheduler', type=str, default='Steplr')
    arg('--learning_rate', type=float, default=1e-3)
    arg('--weight_decay', type=float, default=0.)
    arg('--dropout', type=float, default=0.3)
    arg('--seed', type=int, default=42)
    arg('--unfreeze', action='store_true', help='unfreeze layers after 1 epoch')
    arg('--DEBUG', action='store_true', help='debug mode')
    arg('--cpu', action='store_true', help='use cpu')
    args = parser.parse_args()
    return args


def main():

    args = arg_parser()

    seed_everything(args.seed)

    if cuda.is_available() and not args.cpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    print(device)

    train_df = pd.read_csv(args.train_df_path)
    valid_df = pd.read_csv(args.valid_df_path)
    valid_df_sub = valid_df.sample(frac=1.0, random_state=42).reset_index(drop=True)[:40000]
    valid_df_sub1 = valid_df.sample(frac=1.0, random_state=52).reset_index(drop=True)[:40000]
    valid_df_sub2 = valid_df.sample(frac=1.0, random_state=62).reset_index(drop=True)[:40000]
    del valid_df; gc.collect()

    if args.DEBUG:
        train_df = train_df[:1000]
        valid_df_sub = valid_df_sub[:1000]
        valid_df_sub1 = valid_df_sub1[:1000]
        valid_df_sub2 = valid_df_sub2[:1000]

    train_loader = build_dataset(args, train_df, is_train=True)
    batch_num = len(train_loader)
    valid_loader = build_dataset(args, valid_df_sub, is_train=False)
    valid_loader1 = build_dataset(args, valid_df_sub1, is_train=False)
    valid_loader2 = build_dataset(args, valid_df_sub2, is_train=False)


    model = build_model(args, device)

    if args.model == 'resnet50':
        save_path = os.path.join(args.PATH, 'weights', f'resnet50_best.pt')
    if args.model == 'resnext':
        save_path = os.path.join(args.PATH, 'weights', f'resnext_best.pt')
    elif args.model == 'xception':
        save_path = os.path.join(args.PATH, 'weights', f'xception_best.pt')
    else:
        NotImplementedError

    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer, batch_num)

    train_cfg = {
    'train_loader':train_loader,
    'valid_loader':valid_loader,
    'valid_loader1':valid_loader1,
    'valid_loader2':valid_loader2,
    'model':model,
    'criterion':nn.BCEWithLogitsLoss(),
    'optimizer':optimizer,
    'scheduler':scheduler,
    'save_path':save_path,
    'device':device
    }                                                                                                                                                

    train_model(args, train_cfg)


if __name__ == '__main__':
    main()
