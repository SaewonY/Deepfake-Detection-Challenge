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
import albumentations
from albumentations.augmentations.transforms import ShiftScaleRotate, HorizontalFlip, Normalize, RandomBrightnessContrast, \
                                                    MotionBlur, Blur, GaussNoise, JpegCompression
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
    arg('--n_epochs', type=int, default=4)
    arg('--model', type=str, default='resnet50')
    arg('--model_type', type=str, default='cnn')
    arg('--model_path', type=str)
    arg('--batch_size', type=int, default=32)
    arg('--num_workers', type=int, default=6)
    arg('--optimizer', type=str, default='Adam')
    arg('--scheduler', type=str, default='Steplr')
    arg('--learning_rate', type=float, default=1e-3)
    arg('--weight_decay', type=float, default=0.)
    arg('--dropout', type=float, default=0.3)
    arg('--seed', type=int, default=42)
    arg('--preprocess', action='store_true', help='remove outliers')
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

    if args.model_type == 'cnn':
        if args.preprocess:
            train_df = pd.read_csv('../input/preprocessed_train_df.csv')
            train_df = pd.read_csv('../input/preprocessed_valid_df.csv')
        else:
            train_df = pd.read_csv('../input/train_df.csv')
            valid_df = pd.read_csv('../input/valid_df.csv')
        sample_num = 40000
    
    elif args.model_type == 'lrcn':
        if args.preprocess:
            train_df = pd.read_pickle('../input/preprocessed_lrcn_train_df.pkl')
            valid_df = pd.read_pickle('../input/preprocessed_lrcn_train_df.pkl')
        else:
            train_df = pd.read_pickle('../input/lrcn_train_df.pkl')
            valid_df = pd.read_pickle('../input/lrcn_valid_df.pkl')
        sample_num = 15000

    print("number of train data {}".format(len(train_df)))
    print("number of valid data {}\n".format(len(valid_df)))


    valid_df_sub = valid_df.sample(frac=1.0, random_state=42).reset_index(drop=True)[:sample_num]
    valid_df_sub1 = valid_df.sample(frac=1.0, random_state=52).reset_index(drop=True)[:sample_num]
    valid_df_sub2 = valid_df.sample(frac=1.0, random_state=62).reset_index(drop=True)[:sample_num]
    del valid_df; gc.collect()

    if args.DEBUG:
        train_df = train_df[:1000]
        valid_df_sub = valid_df_sub[:1000]
        valid_df_sub1 = valid_df_sub1[:1000]
        valid_df_sub2 = valid_df_sub2[:1000]


    if args.model_type == 'cnn':
        train_transforms = albumentations.Compose([
                                            HorizontalFlip(p=0.3),
                                            #   ShiftScaleRotate(p=0.3, scale_limit=0.25, border_mode=1, rotate_limit=25),
                                            #   RandomBrightnessContrast(p=0.2, brightness_limit=0.25, contrast_limit=0.5),
                                            #   MotionBlur(p=0.2),
                                              GaussNoise(p=0.3),
                                              JpegCompression(p=0.3, quality_lower=50),
                                            #   Normalize()
        ])
        valid_transforms = albumentations.Compose([
                                                HorizontalFlip(p=0.2),
                                                albumentations.OneOf([
                                                    JpegCompression(quality_lower=8, quality_upper=30, p=1.0),
                                                    GaussNoise(p=1.0),
                                                ], p=0.22),
                                                #   Normalize()
        ])
    elif args.model_type == 'lrcn':
        train_transforms = None
        valid_transforms = None


    train_loader = build_dataset(args, train_df, transforms=train_transforms, is_train=True)
    batch_num = len(train_loader)
    valid_loader = build_dataset(args, valid_df_sub, transforms=valid_transforms, is_train=False)
    valid_loader1 = build_dataset(args, valid_df_sub1, transforms=valid_transforms, is_train=False)
    valid_loader2 = build_dataset(args, valid_df_sub2, transforms=valid_transforms, is_train=False)


    model = build_model(args, device)

    if args.model == 'resnet50':
        save_path = os.path.join(args.PATH, 'weights', f'resnet50_best.pt')
    elif args.model == 'resnext':
        save_path = os.path.join(args.PATH, 'weights', f'resnext_best.pt')
    elif args.model == 'xception':
        save_path = os.path.join(args.PATH, 'weights', f'xception_best.pt')
    else:
        NotImplementedError

    if args.model_type == 'lrcn':
        save_path = os.path.join(args.PATH, 'weights', f'lrcn_best.pt')

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
