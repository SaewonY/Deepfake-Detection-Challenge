import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations
from albumentations.augmentations.transforms import ShiftScaleRotate, HorizontalFlip, Normalize, RandomBrightnessContrast, \
                                                    MotionBlur, Blur, GaussNoise, JpegCompression


class DeepFake_Dataset(Dataset):
    def __init__(self, df, transforms=None):

        self.df = df
        self.transforms = transforms
        
    def __len__(self):            
        return len(self.df)
            
    def __getitem__(self, index):
        
        real_path = self.df['REAL'][index]
        fake_path = self.df['FAKE'][index]
        
        real_img = cv2.imread(real_path)
        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        fake_img = cv2.imread(fake_path)
        fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            real_img = self.transforms(image=real_img)
            real_img = real_img['image']
            fake_img = self.transforms(image=fake_img)
            fake_img = fake_img['image']
        
        real_img = np.rollaxis(real_img, 2, 0)            
        fake_img = np.rollaxis(fake_img, 2, 0)
                    
        return (real_img, torch.tensor(0.)), (fake_img, torch.tensor(1.)) 


class LRCN_Dataset(Dataset):
    def __init__(self, df, transforms=None):

        self.df = df
        self.transforms = transforms
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):

        real_paths = self.df['REAL'][index]
        fake_paths = self.df['FAKE'][index]
        real_images = []
        for real_path in real_paths:
            image = cv2.imread(real_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            real_images.append(image)
        fake_images = []
        for fake_path in fake_paths:
            image = cv2.imread(fake_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            fake_images.append(image)
        return (np.stack(real_images), torch.tensor(0.)), (np.stack(fake_images), torch.tensor(1.))
        

train_transforms = albumentations.Compose([
                                          HorizontalFlip(p=0.3),
                                        #   ShiftScaleRotate(p=0.3, scale_limit=0.25, border_mode=1, rotate_limit=25),
                                        #   RandomBrightnessContrast(p=0.2, brightness_limit=0.25, contrast_limit=0.5),
                                        #   MotionBlur(p=0.2),
                                        #   GaussNoise(p=0.3),
                                        #   JpegCompression(p=0.3, quality_lower=50),
                                        #   Normalize()
])
valid_transforms = albumentations.Compose([
                                          HorizontalFlip(p=0.2),
                                          albumentations.OneOf([
                                              JpegCompression(quality_lower=8, quality_upper=30, p=1.0),
                                              # Downscale(scale_min=0.25, scale_max=0.75, p=1.0),
                                              GaussNoise(p=1.0),
                                          ], p=0.22),
                                        #   Normalize()
])


def build_dataset(args, df, is_train=False, transforms=None):

    shuffle = True if is_train else False

    if args.model_type == 'cnn':
            dataset = DeepFake_Dataset(df, transforms=transforms)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True)
    
    elif args.model_type == 'lrcn':

        dataset = LRCN_Dataset(df, transforms=transforms)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True)

    return loader