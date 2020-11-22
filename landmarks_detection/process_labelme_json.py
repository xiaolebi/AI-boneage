import os
import json
import cv2
import numpy as np
import csv

def checkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def get_landmarks(json_path,save_root,img_path):
    csvFile = open(save_root,'w',newline='')
    writer = csv.writer(csvFile)
    files = os.listdir(img_path)
    writer.writerow(('imagename','x1','y1','x2','y2','x3','y3'))
    for root,subfloders,filenames in os.walk(json_path):
        for filename in filenames:
            if filename[-5:] == '.json':
                file_path = os.path.join(root,filename)
                with open(file_path,'r',encoding='utf-8') as fp:
                    json_data = json.load(fp)
                    points = json_data['shapes'][0]['points']
                    x1, y1 = points[0][0], points[0][1]
                    x2, y2 = points[1][0], points[1][1]
                    x3, y3 = points[2][0], points[2][1]
                    filename = filename[:-5]+'.png'
                    if filename in files:
                        writer.writerow((filename,x1,y1,x2,y2,x3,y3))
                    # img = cv2.imread(img_path+filename)
                    # img = cv2.circle(img, (int(x1), int(y1)), 1, (255, 0, 0), -1)
                    # img = cv2.circle(img, (int(x2), int(y2)), 1, (0, 255, 0), -1)
                    # img = cv2.circle(img, (int(x3), int(y3)), 1, (0, 0, 255), -1)
                    # cv2.imshow('',img)
                    # cv2.waitKey(0)
                    print('Get landmarks from {}.json'.format(filename[:-5]))

if __name__ =="__main__":
    img_path = '../RSNA_boneage_dataset/landmarks_detection/test/'
    json_dir = '../RSNA_boneage_dataset/landmarks_detection/json/'
    save_root = '../RSNA_boneage_dataset/landmarks_detection/test.csv'
    get_landmarks(json_dir,save_root,img_path)