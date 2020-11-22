from __future__ import  division,print_function
import random
import os
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import cv2

class HandLandmarksDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=None,rgb=True):
        self.landmarks_frame = self.get_points(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.rgb = rgb

    def __len__(self):
        return len(self.landmarks_frame)

    def get_points(self,source_file_path):
        points_list = list()
        if not os.path.exists(source_file_path):
            raise Exception("Error, the file:{} not exists!".format(source_file_path))
        with open(source_file_path,'r') as fp:
            for line in fp.readlines():
                if len(line)==1:
                    continue
                try:
                    imgname,mx,my,tx,ty,cx,cy = line.strip('\n\t\r').split(',')
                    imgname = imgname.strip('\n\t\r').split('/')[-1]
                    mx = float(mx)
                    my = float(my)
                    tx = float(tx)
                    ty = float(ty)
                    cx = float(cx)
                    cy = float(cy)
                    points_list.append([imgname,mx,my,tx,ty,cx,cy])
                except:
                    continue
            print('the number of reading data:{}'.format(len(points_list)))
        return points_list

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,self.landmarks_frame[idx][0])
        image = cv2.imread(img_name)
        if self.rgb:
            image = image[...,::-1]
        landmarks = np.array(self.landmarks_frame[idx][1:]).astype('float')
        landmarks = landmarks.reshape(-1,2)
        sample = {'image':image,'landmarks':landmarks}
        if self.transform:
            sample = self.transform(sample)
        return sample

def Rotate(image,degree,landmarks):
    wc,hc = int(image.shape[0]/2),int(image.shape[1]/2)
    R = cv2.getRotationMatrix2D((hc,wc),degree,1)
    rotate = cv2.warpAffine(image,R,(wc*2,hc*2))
    new_landmarks = landmarks*[1,-1]+[-wc,hc]
    rotate_landmarks = np.dot(new_landmarks,np.array([R[0,0],R[0,1],-R[0,1],R[0,0]]).reshape(2,2))
    final_landmarks = rotate_landmarks*[1,-1]+[wc,hc]
    return rotate,final_landmarks

class RotateRandom(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image,landmarks = sample['image'],sample['landmarks']
        rotate = np.random.random()
        img = image
        if rotate<0.5:
            angle = np.random.random()*90-45
            img,landmarks = Rotate(img,angle,landmarks)
        return {'image':img,'landmarks':landmarks}

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
        img = cv2.resize(image,(new_w,new_h))
        landmarks = landmarks * [new_w / w,new_h / h]
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
        landmarks = landmarks - [left,top]
        return {'image':image,'landmarks':landmarks}

class ToTensor(object):
    def __init__(self,image_size):
        self.image_size = image_size

    def __call__(self, sample):
        image,landmarks = sample['image'],sample['landmarks']
        image = image.transpose((2,0,1))
        landmarks = landmarks.reshape(-1,1)
        return {'image':torch.from_numpy(image).float().div(255),'landmarks':torch.from_numpy(landmarks).float()}

class RandomFlip(object):
    def __call__(self,sample):
        image,landmarks = sample['image'],sample['landmarks']
        if random.random()<0.5:
            image = cv2.flip(image,1)
            landmarks[:,0] = image.shape[1] - landmarks[:,0]
        return {'image':image,'landmarks':landmarks}

class Normalize(object):
    def __init__(self,mean,std):
        self.mean = mean
        self.std = std

    def __call__(self,sample):
        image = sample['image']
        for t,m,s in zip(image,self.mean,self.std):
            t = (t - m) / s
        sample['image'] = image
        return sample

class SwapChannels(object):
    '''
    Transforms a tensorized image by swapping the channels in the order specified in the swap tuple
    Args: swaps(int tuple): final order of channel
    '''
    def __init__(self,swaps):
        self.swaps = swaps

    def __call__(self,image):
        image = image[:,:,self.swaps]
        return image

class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0,1,2),(0,2,1),
                      (1,0,2),(1,2,0),
                      (2,0,1),(2,1,0))

    def __call__(self,sample):
        image = sample['image']
        if random.randint(0,2):
            swap = self.perms[random.randint(0,len(self.perms)-1)]
            shuffle = SwapChannels(swap)
            image = shuffle(image)
            sample['image'] = image
        return sample

class RandomContrast(object):
    def __init__(self,lower=0.5,upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower
        assert self.lower >= 0

    def __call__(self, sample):
        if random.randint(0,2):
            image = sample['image']
            alpha = random.uniform(self.lower,self.upper)
            image *= alpha
            sample['image'] = image
        return sample

class RandomBrightness(object):
    def __init__(self,delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self,sample):
        image = sample['image']
        if random.randint(0,2):
            delta = random.uniform(-self.delta,self.delta)
            np.add(image,delta,out=image,casting='safe')
            sample['image'] = image
        return sample

if __name__ == '__main__':
    csv_path = '../RSNA_boneage_dataset/landmarks_detection/train.csv'
    root_dir = '../RSNA_boneage_dataset/landmarks_detection/train'
    transform_train = transforms.Compose([
        Rescale((520, 520)),
        RandomCrop((512, 512)),
        RandomFlip(),
        RotateRandom(),
        # RandomBrightness(),
        # ToTensor(512),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    data = HandLandmarksDataset(csv_path,root_dir,transform=transform_train,rgb=True)
    # trainloader = data.DataLoader(data, batch_size=7, shuffle=True)
    for i in range(10):
        sample = data.__getitem__(i)
        img = sample['image']
        points = sample['landmarks']
        x1, y1 = points[0][0], points[0][1]
        x2, y2 = points[1][0], points[1][1]
        x3, y3 = points[2][0], points[2][1]
        img = cv2.circle(img, (int(x1), int(y1)), 1, (255, 0, 0), -1)
        img = cv2.circle(img, (int(x2), int(y2)), 1, (0, 255, 0), -1)
        img = cv2.circle(img, (int(x3), int(y3)), 1, (0, 0, 255), -1)
        cv2.imshow('', img)
        cv2.waitKey(0)