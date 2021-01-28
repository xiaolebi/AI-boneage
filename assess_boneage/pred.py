import torch
import cv2
import numpy as np
from AgeDataset import *
from collections import OrderedDict
import csv
from models.BoneAgeNet import BoneAge
import pandas as pd
import os
mean = np.asarray([0.4465,0.4822,0.4914])
std = np.asarray([0.1994,0.1994,0.2023])
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def load_model(checkpoint_path):
    model = BoneAge(1)
    print('========loading model==========')
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = k[0:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    print('========model loaded==========')
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
    checkpoint = '/content/checkpoints/model_best.pth.tar'
    test_path = '/content/dataset/valid'
    test_csv = '/content/dataset/valid.csv'
    model = load_model(checkpoint)
    reader = csv.reader(open(test_csv))
    landmarks_frame = [[row[0],np.float64(row[1]),row[2]] for i,row in enumerate(reader) if i > 0]
    data_num = len(landmarks_frame)
    boneage = []
    df_data = pd.read_csv(test_csv)
    num = 0
    num_6 = 0
    num_12 = 0
    for j in range(data_num):
        img_path = os.path.join(test_path,landmarks_frame[j][0]+'.png')
        r_age = landmarks_frame[j][1]
        gender_s = landmarks_frame[j][2]
        gender = np.array([1,])
        if gender_s == 'False':
            gender = np.array([0,])
        img = cv2.imread(img_path)
        age = pre_bone_age(model,img,gender,512)
        df_data.loc[df_data.id == int(landmarks_frame[j][0]),'AI boneage'] = age
        if abs(float(r_age)-round(age)) > 20 :
            print(abs(float(r_age) - round(age)), 'real age', r_age, '||', 'pred age', age)
            num += 1
        elif abs(float(r_age)-round(age)) > 12 :
            print(abs(float(r_age) - round(age)), 'real age', r_age, '||', 'pred age', age)
            num_12 += 1
        elif abs(float(r_age)-round(age)) > 6 :
            print(abs(float(r_age) - round(age)), 'real age', r_age, '||', 'pred age', age)
            num_6 += 1
        boneage.append(abs(float(r_age)-round(age)))
    print('MAE',np.mean(boneage),len(boneage))
    print('Difference greater than 20',num)
    print('Difference greater than 12', num_12)
    print('Difference greater than 6', num_6)
    df_data.to_csv('/content/dataset/valid_result_0127.csv',index=False)
