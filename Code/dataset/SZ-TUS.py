import numpy as np
import torch
import os
import random
from scipy.ndimage import zoom
from PIL import Image
from copy import deepcopy
from torch.utils.data import Dataset
from torchvision import transforms
from Code.dataset.transform import random_rot_flip, random_rotate, blur



class SZ_TUS_Dataset(Dataset):
    def __init__(self, data_path, image_size=None, mode=None):
        super(SZ_TUS_Dataset, self).__init__()
        self.size = image_size
        self.mode = mode
        if mode=='train_l':
            self.txtpath = os.path.join(data_path, 'labeled.txt')
        elif mode == 'val':
            self.txtpath = os.path.join(data_path, 'labeled.txt')
        else:
            self.txtpath = os.path.join(data_path, 'unlabeled.txt')
        f = open(self.txtpath, 'r')
        data = f.readlines()
        imgs = []
        masks = []
        labels = []
        for line in data:
            if mode == 'val' or mode == 'train_l':
                line = line.rstrip()
                word = line.split(' ')   # other datasets use '-' , BUSI dataset uses ' '
                imgs.append(os.path.join(data_path,'image',word[0]))
                masks.append(os.path.join(data_path,'mask',word[1]))
                labels.append(word[2])
            else:
                line = line.rstrip()
                imgs.append(os.path.join(data_path, 'image', line))

        self.img =imgs
        self.mask =masks
        self.label =labels
        if len(self.img) == 0:
            raise RuntimeError('Found 0 images, please check the data set')

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        img_path = self.img[index]
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)

        if self.mode == 'train_l' or self.mode == 'val':
            mask_path = self.mask[index]
            label = self.label[index]
            mask = Image.open(mask_path).convert('L')
            mask = np.array(mask)
            mask =  mask / 255
            # mask = Image.fromarray(mask)
            label = np.array(label).astype(np.int64)  # The type of label must be torch.int64
            label = torch.from_numpy(label)
        if self.mode == 'val':
            x, y, _ = img.shape
            img = zoom(img, (self.size / x, self.size / y, 1), order=1)
            mask = zoom(mask, (self.size / x, self.size / y), order=0)
            return transforms.ToTensor()(img), torch.from_numpy(np.array(mask)), label

        # img, mask = _transforms.FreeScale((256,256))(img,mask)
        if random.random() > 0.5:
            if self.mode == 'train_l' or self.mode == 'val':
                img, mask = random_rot_flip(img, mask)
            else:
                img = random_rot_flip(img)
        elif random.random() > 0.5:
            if self.mode == 'train_l' or self.mode == 'val':
                img, mask = random_rotate(img, mask)
            else:
                img = random_rotate(img)
        x, y, _ = img.shape
        del _
        img = zoom(img, (self.size / x, self.size / y, 1), order=1)

        if self.mode == 'train_l' or self.mode == 'val':
            mask = zoom(mask, (self.size / x, self.size / y), order=0)

        if self.mode == 'train_l' :
            return transforms.ToTensor()(img), torch.from_numpy(np.array(mask)), label

        img = Image.fromarray(img)
        img_s1 = deepcopy(img)
        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1= blur(img_s1, p=0.5)
        img = transforms.ToTensor()(img)
        img_s1 = transforms.ToTensor()(img_s1)
        return img, img_s1






