import os
import gc
import cv2
import glob
import time
import tqdm
import math
import pickle
import random
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict

import warnings
warnings.filterwarnings('ignore')


def arg_parser():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--PATH', type=str, default='../input')
    arg('--mode', type=str, default='train')
    arg('--outlier_df_path', type=str, default='../input/fake_outlier_list.txt')
    arg('--train_df_path', type=str, default='../input/train_df.csv')
    arg('--valid_df_path', type=str, default='../input/valid_df.csv')
    arg('--train_images_path', type=str, default='../input/train_images_v2')
    arg('--valid_images_path', type=str, default='../input/valid_images_v2')
    arg('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def make_df(args, fake_outlier_list):

    PATH = args.PATH
    if args.mode == 'train':
        print("start making train_df")
        IMAGE_PATH = args.train_images_path
        start = 0
        end = 40

    else:
        print("start making valid_df")
        IMAGE_PATH = args.valid_images_path
        start = 40
        end = 50

    VIDEOS = [f'dfdc_train_part_{i}' for i in range(start, end)]
    META = [f'{i}.json' for i in range(start, end)]

    sampled_df = pd.DataFrame(columns=['REAL', 'FAKE'])

    for i, (chunk, meta) in tqdm(enumerate(zip(VIDEOS, META)), total=len(VIDEOS)):
            
    #     print(f"{chunk}")
        df = pd.read_json(f'../input/metadata/{meta}').transpose().reset_index().rename(columns={'index':'file_name'})
        df.drop(columns=['split'], inplace=True)
        df.drop(columns=['label'], inplace=True)
        df.dropna(inplace=True)
        
        # removing outliers
        # df = df.loc[~(df['file_name'].isin(fake_outlier_list))]
        
        df['original'] = df['original'].apply(lambda x: x.split('.')[0])
        df['file_name'] = df['file_name'].apply(lambda x: x.split('.')[0])

        df = df.sort_values(by=['original'], ascending=True).reset_index(drop=True)
        
        unique_original_list = df['original'].unique()

        ####################################################################################
        
        original_files_dict = {}
        
        for origin_video in unique_original_list:

            # 모든 real 이미지 파일 경로 추출 
            origin_list = glob.glob(os.path.join(PATH, IMAGE_PATH, f'{chunk}', f'{origin_video}*.jpg'))
                    
            frames_list = []
            for origin in origin_list:
                frame_num = origin.split('/')[-1].split('.')[0].split('_')[-1]
                frames_list.append(frame_num)
                
            temp_df = df.loc[df['original'] == origin_video]
            fake_video_list = temp_df['file_name'].values            
            
            for i, frame_num in enumerate(frames_list):

                fake_frame_list = []
                
                for fake_video in fake_video_list:
                    fake_list = glob.glob(os.path.join(PATH, IMAGE_PATH, f'{chunk}', f'{fake_video}_{frame_num}.jpg'))
                    fake_frame_list = fake_frame_list + fake_list
                
                try:
                    rand_frame = random.choice(fake_frame_list)
                except:
                    continue
        
                original_files_dict[origin_list[i]] = rand_frame
            
        chunk_df = pd.DataFrame.from_dict(original_files_dict, orient='index', columns=['FAKE'])
        chunk_df = chunk_df.reset_index().rename(columns={'index':'REAL'})

        sampled_df = sampled_df.append(chunk_df)
                
    sampled_df = sampled_df.reset_index(drop=True)
    return sampled_df


def main():

    args = arg_parser()

    random.seed(args.seed)

    # load outliers
    with open(args.outlier_df_path, "rb") as fp:   
        fake_outlier_list = pickle.load(fp)

    if args.mode == 'train':
        train_df = make_df(args, fake_outlier_list)
        train_df.to_csv(args.train_df_path, index=False)
        print("saved train_df")
    elif args.mode == 'valid':
        valid_df = make_df(args, fake_outlier_list)
        valid_df.to_csv(args.valid_df_path, index=False)
        print("saved valid_df")


if __name__ == '__main__':
    main()