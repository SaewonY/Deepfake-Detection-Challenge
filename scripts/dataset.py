import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


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
        random_prob = random.random()
        if random_prob < 2/9: 
            flip = True
        else:
            flip = False
        for real_path in real_paths:
            image = cv2.imread(real_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if flip:
                image = cv2.flip(image, 1)
            real_images.append(image)

        fake_images = []
        random_prob = random.random()
        if random_prob < 2/9: 
            flip = True
        else:
            flip = False
        for fake_path in fake_paths:
            image = cv2.imread(fake_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if flip:
                image = cv2.flip(image, 1)
            fake_images.append(image)
            
        return (np.stack(real_images), torch.tensor(0.)), (np.stack(fake_images), torch.tensor(1.))


def build_dataset(args, df, is_train=False, transforms=None):

    shuffle = True if is_train else False

    if args.model_type == 'cnn':
            dataset = DeepFake_Dataset(df, transforms=transforms)
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True)
    
    elif args.model_type == 'lrcn':

        dataset = LRCN_Dataset(df, transforms=transforms)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True)

    return loader