from __future__ import division
import argparse
import torch
import os
import cv2
import numpy as np
from torch.autograd import Variable
from models.basenet import Res152
from collections import OrderedDict
import csv
import time

parser = argparse.ArgumentParser(description='Pytorch hand landmark')
parser.add_argument('--img','--image',default='hand3',type=str)
parser.add_argument('--j','--workers',default=8,type=int,metavar='N',help='number of data loading workers (default:4)')
parser.add_argument('--gpu_id',default='0',type=str,help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--c','--checkpoint',default='',type=str,metavar='PATH',help='path to save checkpoint (default:checkpoint)')

args = parser.parse_args()
mean = np.asarray([0.4465,0.4822,0.4914])
std = np.asarray([0.1994,0.1994,0.2023])

def load_model():
    model = Res152(6)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint  = torch.load(args.checkpoint)
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict
    for k,v in state_dict.items():
        name = k[0:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model

def get_landmarks_point(model,seg_img,out_size):
    img = cv2.resize(seg_img,(out_size,out_size))
    img = img/255.0
    img = (img-mean)/std
    img = img.transpose((2,0,1))
    img = img[np.newaxis,...]
    input = torch.from_numpy(img).float()
    with torch.no_grad():
        input = Variable(input)
    out = model(input).cpu().data.numpy()
    return [out[0,0],out[0,1],out[0,2],out[0,3],out[0,4],out[0,5]]



if __name__ == '__main__':
    out_size = 512
    model = load_model()
    model = model.eval()
    img_root = ''
    with open('.csv','wb') as f:
        f_csv = csv.writer(f)
        img_save_path = ''
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)
        for img_path in os.listdir(img_root):
            start_time = time.time()
            img = cv2.imread(os.path.join(img_root, img_path))
            img = cv2.resize(img,(out_size,out_size))
            raw_img = img
            img = img/255.0
            img = (img-mean)/std
            img = img.transpose((2,0,1))
            img = img.reshape((1,) + img.shape)
            input = torch.from_numpy(img).float()
            input = torch.autograd.Variable(input)
            out = model(input).cpu().data.numpy()
            f_csv.writerow([img_path,out[0,0],out[0,1],out[0,2],out[0,3],out[0,4],out[0,5]])
            out = out.reshape(-1,2)
            raw_img = cv2.resize(raw_img,(out_size,out_size))
            for i in range(3):
                cv2.circle(raw_img,(int(out[i][0]),int(out[i][1])),10,(255,0,0),-1)
            cv2.imwrite(os.path.join(img_save_path,img_path),raw_img)
            end_time = time.time()
            print(img_path + 'time is ' + str(end_time - start_time))
