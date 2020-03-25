import os
import re
import glob
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import seed_everything


def make_df(args, fake_outlier_list):

    if args.mode == 'train':
        print("start making train_df")
        IMAGE_PATH = args.train_images_path
        start = 0; end = 40

    else:
        print("start making valid_df")
        IMAGE_PATH = args.valid_images_path
        start = 40; end = 50

    VIDEOS = [f'dfdc_train_part_{i}' for i in range(start, end)]
    META = [f'{i}.json' for i in range(start, end)]

    real_list = []
    fake_list = []

    for i, (chunk, meta) in tqdm(enumerate(zip(VIDEOS, META)), total=len(VIDEOS)):
        df = pd.read_json(f'../input/metadata/{meta}').transpose().reset_index().rename(columns={'index':'file_name'})
        df.drop(columns=['split'], inplace=True)
        df.drop(columns=['label'], inplace=True)
        df.dropna(inplace=True)

        if args.preprocess:
            df = df.loc[~(df['file_name'].isin(fake_outlier_list))]

        df['original'] = df['original'].apply(lambda x: x.split('.')[0])
        df['file_name'] = df['file_name'].apply(lambda x: x.split('.')[0])
        df = df.sort_values(by=['original'], ascending=True).reset_index(drop=True)

        unique_original_list = df['original'].unique()

        for origin_video in unique_original_list:
            origin_frames_list = glob.glob(os.path.join(IMAGE_PATH, f'{chunk}', f'{origin_video}*.jpg'))
            
            # frame 순으로 정렬
            sorted_frame_origin_list = sorted(origin_frames_list, key=lambda x: int(re.findall(r'\d+', x.split('/')[-1])[0]))

            for i in range(0, len(sorted_frame_origin_list)-args.extract_frame_num+1, 2): # test
                real_list.append(sorted_frame_origin_list[i:i+args.extract_frame_num])
                
            # original에 대응하는 frame 한 개 랜덤하게 샘플링
            temp_df = df.loc[df['original'] == origin_video].reset_index(drop=True)
            rand_row_index = random.randint(0, len(temp_df)-1)
            fake_video = temp_df['file_name'][rand_row_index]
            fake_frames_list = glob.glob(os.path.join(IMAGE_PATH, f'{chunk}', f'{fake_video}*.jpg'))
            if len(fake_frames_list) == 0:
                continue
            
            # frame 순으로 정렬
            sorted_frame_fake_list = sorted(fake_frames_list, key=lambda x: int(re.findall(r'\d+', x.split('/')[-1])[0]))

            for i in range(0, len(sorted_frame_fake_list)-args.extract_frame_num+1, 2):
                fake_list.append(sorted_frame_fake_list[i:i+args.extract_frame_num])

    real_df = pd.DataFrame(columns=['REAL'])
    fake_df = pd.DataFrame(columns=['FAKE'])
    
    assert len(real_list) == len(fake_list)

    for i, (frames, frames1) in tqdm(enumerate(zip(real_list, fake_list)), total=len(real_list)):
        real_df.loc[i, 'REAL'] = frames
        fake_df.loc[i, 'FAKE'] = frames1

    real_fake_df = pd.concat([real_df, fake_df], axis=1).reset_index(drop=True)

    return real_fake_df


def arg_parser():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--mode', type=str, default='train')
    arg('--outlier_df_path', type=str, default='../input/fake_outlier_list.txt')
    arg('--preprocess', action='store_true', help='remove outliers')
    arg('--extract_frame_num', type=int, default=10, help='sequential frames nums to extract')
    arg('--train_df_save_path', type=str, default='../input/lrcn_train_df.pkl')
    arg('--valid_df_save_path', type=str, default='../input/lrcn_valid_df.pkl')
    arg('--train_images_path', type=str, default='../input/train_images_v3')
    arg('--valid_images_path', type=str, default='../input/valid_images_v3')
    arg('--seed', type=int, default=42)
    args = parser.parse_args()
    return args


def main():

    args = arg_parser()

    seed_everything(args.seed)

    # load outliers
    with open(args.outlier_df_path, "rb") as fp:   
        fake_outlier_list = pickle.load(fp)

    if args.mode == 'train':
        train_df = make_df(args, fake_outlier_list)

        if args.preprocess:
            train_df.to_pickle('../input/preprocessed_lrcn_train_df.pkl')
        else:
            train_df.to_pickle(args.train_df_save_path)
        print("saved train_df")

    elif args.mode == 'valid':
        valid_df = make_df(args, fake_outlier_list)

        if args.preprocess:
            valid_df.to_pickle('../input/preprocessed_lrcn_valid_df.pkl')
        else:
            valid_df.to_pickle(args.valid_df_save_path)
        print("saved valid_df")


if __name__ == '__main__':
    main()