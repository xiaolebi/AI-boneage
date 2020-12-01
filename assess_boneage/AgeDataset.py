from __future__ import print_function,division
import random
import os
import torch
import numpy as np
import csv
from torch.utils.data import Dataset,DataLoader
import cv2
from torchvision import transforms

class AgeDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=None,rgb=True):
        self.reader = csv.reader(open(csv_file))
        self.landmarks_frame = [[row[0],np.float64(row[1]),row[2]] for i,row in enumerate(self.reader) if i > 0]
        self.root_dir = root_dir
        self.transform = transform
        self.rgb = rgb

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.landmarks_frame[idx][0] + '.png')
        image = cv2.imread(img_name,cv2.COLOR_BGR2BGRA)
        # if self.rgb:
        #     image = np.dot(image[...,:3],[0.299,0.587,0.114])
        # image = normalize(image,True,0.05)
        # image[image > 4.5] = 4.5
        image = np.repeat(image[:,:,np.newaxis],3,axis=2)
        landmarks = self.landmarks_frame[idx][1]
        gender_s = self.landmarks_frame[idx][2]
        gender = np.array([1,])
        if gender_s == 'False':
            gender = np.array([0,])
        sample = {'image':image,'landmarks':landmarks}
        if self.transform:
            sample = self.transform(sample)
        gender = torch.from_numpy(gender).float()
        sample['gender'] = gender
        return sample

def normalize(input,crop=False,crop_val=0.5):
    if crop:
        b,t = np.percentile(input,(crop_val,100-crop_val))
        slice = np.clip(input,b,t)
        if np.std(slice) == 0:
            return slice
        else:
            return (slice-np.mean(slice)) / np.std(slice)
    return (input-np.mean(input)) / np.std(input)

def pad(img,new_size):
    old_size = img.shape[:1]
    scale = (np.array(new_size) - np.array(old_size)) / 2
    top,bottom,left,right = int(scale[0]),int(scale[0]),int(scale[1]),int(scale[1])
    new_img = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_REPLICATE)
    return new_img

class Rescale(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image,landmarks = sample['image'],sample['landmarks']
        h,w = image.shape[:2]
        if isinstance(self.output_size,int):
            if h>w:
                new_h,new_w = self.output_size * h / w,self.output_size
            else:
                new_h,new_w = self.output_size,self.output_size * w / h
        else:
            new_h,new_w = self.output_size
        new_h,new_w = int(new_h),int(new_w)
        img = pad(image,(new_w,new_h))
        return {'image':img,'landmarks':landmarks}

class RandomCrop(object):
    def __init__(self,output_size):
        assert isinstance(output_size,(int,tuple))
        if isinstance(output_size,int):
            self.output_size = (output_size,output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image,landmarks = sample['image'],sample['landmarks']
        h,w = image.shape[:2]
        new_h,new_w = self.output_size
        top = np.random.randint(0,h-new_h)
        left = np.random.randint(0,w-new_w)
        image = image[top:top+new_h,left:left+new_w]
        return {'image':image,'landmarks':landmarks}

class ToTensor(object):
    def __init__(self,image_size):
        self.image_size = image_size

    def __call__(self, sample):
        image,landmarks = sample['image'],sample['landmarks']
        image = image.transpose((2,0,1))
        landmarks = np.array([landmarks,])
        return {'image':torch.from_numpy(image).float(),'landmarks':torch.from_numpy(landmarks).float()}

class RandomFlip(object):
    def __call__(self,sample):
        image,landmarks = sample['image'],sample['landmarks']
        if random.random()<0.5:
            image = cv2.flip(image,1)
        return {'image':image,'landmarks':landmarks}

if __name__ == '__main__':
    trainset = AgeDataset(csv_file='../RSNA_boneage_dataset/assess_dataset/train.csv', transform=None,root_dir='../RSNA_boneage_dataset/assess_dataset/train')
    for i in range(10):
        sample = trainset.__getitem__(i)
        img = sample['image']
        landmarks = sample['landmarks']
        print(img.shape,landmarks)
        cv2.imshow('',img)
        cv2.waitKey(0)