import torch
import cv2
import numpy as np
from AgeDataset import *
from collections import OrderedDict
import csv
from models.BoneAgeNet import BoneAge
import pandas as pd
mean = np.asarray([0.4465,0.4822,0.4914])
std = np.asarray([0.1994,0.1994,0.2023])

def load_model(checkpoint_path):
    model = BoneAge(1)
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[0:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model.eval()

def pre_bone_age(model,img,gender,out_size):
    img = cv2.resize(img,(out_size,out_size))
    img = np.dot(img[...,:3],[0.299,0.587,0.114])
    img = normalize(img,True,0.05)
    img[img>4.5] = 4.5
    img = np.repeat(img[:,:,np.newaxis],3,axis=2)
    img = img.transpose((2,0,1))
    img = img[np.newaxis,...]
    gender = gender[np.newaxis,...]
    input = torch.from_numpy(img).float()
    input = torch.autograd.Variable(input)
    gender = torch.from_numpy(gender).float()
    gender = torch.autograd.Variable(gender)
    output = model(input,gender).cpu().data.numpy()
    return output[0,0]

if __name__ == '__main__':
    checkpoint = ''
    model = load_model(checkpoint)