import os
import torch
import cv2
import numpy as np
import torch.utils.data as DATA
from PIL import Image

root = '../RSNA_boneage_dataset/boneage-training-dataset/'
train = 'boneage-training-dataset'
mask = 'mask'
landmark_path = ''
test = 'test'
test_mask = 'test_mask'
num_class = 2
class_weight = [6,4]

class HandDataSet(DATA.Dataset):
    def __init__(self,root,train,mask=None,transform=True,trainable=True,train_image_list=None):
        self.trainable = trainable
        self.train_path = os.path.join(root,train)
        if self.trainable:
            self.mask_path = os.path.join(root,mask)
        self.image_list = train_image_list
        self.transform = transform

    def __getitem__(self, index):
        if self.trainable:
            img_path = os.path.join(self.train_path,self.image_list[index])
            mask_name = self.image_list[index]
            mask_path = os.path.join(self.mask_path,mask_name)
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            if self.transform:
                rotate = np.random.random()
                if rotate<0.25:
                    img = img.transpose(Image.ROTATE_90)
                    mask = mask.transpose(Image.ROTATE_90)
                elif rotate<0.5:
                    img = img.transpose(Image.ROTATE_180)
                    mask = mask.transpose(Image.ROTATE_180)
                elif rotate<0.75:
                    img = img.transpose(Image.ROTATE_270)
                    mask = mask.transpose(Image.ROTATE_270)
                mask = np.array(mask).astype(np.uint8)
                mask = cv2.resize(mask,(512,512))
                mask = np.array(mask).astype(np.int64)
                img = np.array(img.convert('RGB')).astype(np.float32)
                img = img/255.0
                if np.random.random()<0.5:
                    img = np.fliplr(img)
                    mask = np.fliplr(mask)
                mask = mask[...,np.newaxis]
                img = cv2.resize(img,(512,512))
            return np.transpose(img,[2,0,1]).copy(),mask.copy()
        else:
            img_path = os.path.join(self.train_path,self.image_list[index])
            img = np.array(Image.open(img_path).convert('RGB')).astype(np.float32)
            img /= 255.0
            img = np.resize(img, (512, 512))
            return np.transpose(img,[2,0,1]).copy(),self.image_list[index]
    def __len__(self):
        return  len(self.image_list)

def augement_train_valid_split(dataset,test_size=0.15,shuffle=False,random_seed=0):
    length = len(dataset)
    indices = list(range(1,length))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    if type(test_size) is float and test_size<1.0:
        split = int(test_size*length)
    elif type(test_size) is int:
        split = test_size
    else:
        raise ValueError('% should be an int or a float' % str)
    all_dataset = np.array(dataset)
    train_indices = indices[split:]
    eval_indices = indices[:split]
    train_list = list(all_dataset[train_indices])
    eval_list = list(all_dataset[eval_indices])
    return train_list,eval_list

if __name__ == '__main__':
    train_image_list = os.listdir(os.path.join(root, mask))
    train_filename_list, eval_filename_list = augement_train_valid_split(train_image_list,0.15, shuffle=True)
    print(len(train_filename_list),len(eval_filename_list))
    data = HandDataSet(root,train,mask, transform=True, trainable=True,train_image_list=train_filename_list)
    x,y = data.__getitem__(7)
    x = np.transpose(x,[1,2,0])
    print(x.shape)
    cv2.imshow('',y)
    cv2.waitKey(0)
    cv2.imshow('',x)
    cv2.waitKey(0)