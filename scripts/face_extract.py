import os
import sys
import cv2
import time
import glob
import random
import pickle
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from credential import token
from slacker import Slacker

import warnings
warnings.filterwarnings('ignore')



def isotropically_resize_image(img, size, resample=cv2.INTER_AREA):
    h, w = img.shape[:2]
    if w > h:
        h = h * size // w
        w = size
    else:
        w = w * size // h
        h = size
    resized = cv2.resize(img, (w, h), interpolation=resample)
    return resized

def make_square_image(img):
    h, w = img.shape[:2]
    size = max(h, w)
    t = 0
    b = size - h
    l = 0
    r = size - w
    return cv2.copyMakeBorder(img, t, b, l, r, cv2.BORDER_CONSTANT, value=0)

def center_crop(img, crop_size):
    width, height = img.shape[:2]
    if height < crop_size or width < crop_size:
        raise ValueError

    x1, y1, x2, y2 = get_center_crop_coords(height, width, crop_size)
    img = img[y1:y2, x1:x2]
    return img

def get_center_crop_coords(height, width, crop_size):
    y1 = (height - crop_size) // 2
    y2 = y1 + crop_size
    x1 = (width - crop_size) // 2
    x2 = x1 + crop_size
    return x1, y1, x2, y2


def get_mobilenet_face(args, image, preprocess=False):
    global boxes,scores, num_detections
    (im_height, im_width) = image.shape[:-1]
    imgs = np.array([image])
    (boxes, scores) = sess.run(
        [boxes_tensor, scores_tensor],
        feed_dict={image_tensor: imgs})
    max_ = np.where(scores==scores.max())[0][0]
    box = boxes[0][max_]
    ymin, xmin, ymax, xmax = box
    
    (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                ymin * im_height, ymax * im_height)
    left, right, top, bottom = int(left), int(right), int(top), int(bottom)
    
    width = right - left
    height = bottom - top

    w_margin = width//5
    h_margin = height//8

    left = left - w_margin 
    right = right + w_margin
    top = top - h_margin
    bottom = bottom + h_margin

    left = 0 if left < 0 else left
    right = im_width if right > im_width else right
    top = 0 if top < 0 else top 
    bottom = im_height if bottom > im_height else bottom

        
    return left, right, top, bottom


def crop_image(frame,bbox):
    left, right, top, bottom=bbox
    return frame[top:bottom,left:right]

def get_img(args, frame, frame_idx, save_path, file_name, frames_boxes_dict=None, is_fake=False, preprocess=False):
    
    if is_fake:
        box = frames_boxes_dict[frame_idx]
        cropped_face = crop_image(frame, box)
        # resized_face = cv2.resize(cropped_face, (args.resize, args.resize))
        # center_cropped = center_crop(resized_face, args.crop_size)
        
        resized = isotropically_resize_image(np.array(cropped_face), 256)
        squared = make_square_image(resized)
        center_cropped = center_crop(squared, 224)

        file_save_name = f'{file_name}_{frame_idx}.jpg'
        cv2.imwrite(os.path.join(save_path, file_save_name), (cv2.cvtColor(center_cropped, cv2.COLOR_RGB2BGR)))

        
    else:
        
        box = get_mobilenet_face(args, frame, preprocess)
        cropped_face = crop_image(frame, box)

        # resized_face = cv2.resize(cropped_face, (args.resize, args.resize))
        # center_cropped = center_crop(resized_face, args.crop_size)
        
        resized = isotropically_resize_image(np.array(cropped_face), 256)
        squared = make_square_image(resized)
        center_cropped = center_crop(squared, 224)
    
        file_save_name = f'{file_name}_{frame_idx}.jpg'
        cv2.imwrite(os.path.join(save_path, file_save_name), (cv2.cvtColor(center_cropped, cv2.COLOR_RGB2BGR)))
        return box



def detect_video(args, video, save_path, file_name, preprocess=False, is_fake=False, frames_boxes_dict=None):
            
    capture = cv2.VideoCapture(video)
    v_len = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if is_fake:
        
        for frame_idx in range(int(v_len)):
            success = capture.grab()

            if not success:
                pass

            if frame_idx in frames_boxes_dict.keys():
                
                success, frame = capture.retrieve()
                
                if not success or frame is None:
                    print(f"{frame_idx} not returned")
                    pass
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    if preprocess and args.mode == 'valid':
                        width, height = frame.shape[:-1]
                        frame = cv2.resize(frame, (width//4, height//4) , fx=0, fy=0, interpolation = cv2.INTER_CUBIC)

                    try:
                        get_img(args, frame, frame_idx, save_path, file_name, frames_boxes_dict, is_fake=True)
                    except Exception as err:
                        print(err)
                        continue
        
    else:
    
        frame_count = args.extract_num
        minimum = 15
        frames_to_extract = np.linspace(0, v_len, frame_count, endpoint=False, dtype=np.int)

        frames_boxes_dict = {}

        i = 0
        for frame_idx in range(int(v_len)):
            success = capture.grab()

            if not success:
                pass

            if frame_idx in frames_to_extract:

                success, frame = capture.retrieve()
                if not success or frame is None:
                    pass
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    try:
                        if preprocess and args.mode == 'valid':
                            width, height = frame.shape[:-1]
                            frame = cv2.resize(frame, (width//4, height//4) , fx=0, fy=0, interpolation = cv2.INTER_CUBIC)
                            box = get_img(args, frame, frame_idx, save_path, file_name, is_fake=False, preprocess=True)
                    
                        else:
                            box = get_img(args, frame, frame_idx, save_path, file_name, is_fake=False)
                    except Exception as err:
                        print(err)
                        continue

                    frames_boxes_dict[frame_idx] = box

                    i += 1

        if i < minimum:
            print(f"file {file_name} {i} faces returned")
            return []

        return frames_boxes_dict



def extract_faces(args, SAVE_PATH, start, end):

    VIDEOS = [f'dfdc_train_part_{i}' for i in range(start, end)]
    META = [f'{i}.json' for i in range(start, end)]

    os.makedirs(SAVE_PATH, exist_ok=True)            

    for i, (chunk, meta) in tqdm(enumerate(zip(VIDEOS, META)), total=len(VIDEOS)):

        print(f"{chunk} extracting starts\n")
        
        os.makedirs(os.path.join(SAVE_PATH, f'{chunk}'), exist_ok=True)            
        save_path = os.path.join(SAVE_PATH, f'{chunk}')
        
        # load metadata
        metadata = pd.read_json(f'../input/metadata/{meta}').transpose().reset_index().rename(columns={'index':'filename'})
        metadata.dropna(inplace=True)
        
        unique_list = metadata['original'].unique()
        
        for original in tqdm(unique_list, total=len(unique_list)):
            temp_df = metadata.loc[metadata['original'] == original].reset_index(drop=True)
            
            real_path = os.path.join(args.PATH, 'unzipped_videos', f'{chunk}', temp_df['original'][0])
            file_name = real_path.split('/')[-1].split('.')[0]

        
            # real video로부터 프레임과 해당 프레임의 box좌표들 반환
            if args.mode == 'train':
                frames_boxes_dict = detect_video(args, real_path, save_path, file_name)
                preprocess = False

            elif args.mode == 'valid':

                # valid일 경우 랜덤하게 1/4 resolution 적용
                random_prob = random.random()
                if random_prob < 2/9:
                    preprocess = True
                else:
                    preprocess = False

                frames_boxes_dict = detect_video(args, real_path, save_path, file_name, preprocess)
        
            if not frames_boxes_dict:
                continue

            fake_list = temp_df['filename'].values
            
            # real로부터 반환된 box 좌표를 사용하여 얼굴 검출
            for fake in fake_list:
                fake_path = os.path.join(args.PATH, 'unzipped_videos', f'{chunk}', fake)
                file_name = fake_path.split('/')[-1].split('.')[0]
                detect_video(args, fake_path, save_path, file_name, preprocess, is_fake=True, frames_boxes_dict=frames_boxes_dict)

        # slack notice
        slack = Slacker(token)
        slack.chat.post_message('#deepfake-train', '{}  time {}'.format(
                                                    chunk, datetime.now().replace(second=0, microsecond=0)
                                ))



def arg_parser():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--PATH', type=str, default='../input')
    arg('--mode', type=str, default='valid')
    arg('--train_images_path', type=str, default='../input/train_images_v3')
    arg('--valid_images_path', type=str, default='../input/valid_images_v3')
    arg('--start', type=int, default=0)
    arg('--extract_num', type=int, default=30)
    arg('--resize', type=int, default=256)
    arg('--crop_size', type=int, default=224)
    args = parser.parse_args()
    return args


def main():

    args = arg_parser()

    if args.mode == 'train':
        print("start extracting train images")
        SAVE_PATH = args.train_images_path
        start, end = 0, 40

    elif args.mode == 'valid':
        print("start extracting valid images")
        SAVE_PATH = args.valid_images_path
        start, end = 40, 50

    if args.start:
        start = args.start
    
    extract_faces(args, SAVE_PATH, start, end)


if __name__ == '__main__':

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile('../input/frozen_inference_graph_face.pb', 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess=tf.compat.v1.Session(graph=detection_graph, config=config)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes_tensor = detection_graph.get_tensor_by_name('detection_boxes:0')    
        scores_tensor = detection_graph.get_tensor_by_name('detection_scores:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    main()