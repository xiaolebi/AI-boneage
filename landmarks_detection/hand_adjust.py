import numpy as np
import cv2

''' This script is used to register the images based on the hand landmarks and do lightness normalization'''

def Rotate(image,degree,landmarks):
    wc,hc = int(image.shape[0]/2),int(image.shape[1]/2)
    R = cv2.getRotationMatrix2D((hc,wc),degree,1)
    rotate_image = cv2.warpAffine(image,R,(wc*2,hc*2),flags=cv2.INTER_CUBIC)
    try:
        assert isinstance(landmarks,list)
        landmarks = np.array(landmarks)

    except:
        pass

    finally:
        landmarks = landmarks.reshape(-1,2)
        new_landmarks = landmarks * [1,-1] + [-wc,hc]
        rotate_landmarks = np.dot(new_landmarks,np.array([R[0,0],R[0,1],-R[0,1],R[0,1]]).reshape(2,2))
        final_landmarks = rotate_landmarks * [1,-1] + [wc,hc]
        final_landmarks = final_landmarks.reshape(-1).tolist()
        return rotate_image,final_landmarks

def drawPoints(image,points):
    img = image
    points = np.array(points,dtype=np.int32)
    points = points.reshape(-1,2)
    cv2.circle(img, (points[0][0], points[0][1]), 3, (255, 0, 0), 5)
    cv2.circle(img, (points[1][0], points[1][1]), 3, (0, 255, 0), 5)
    cv2.circle(img, (points[2][0], points[2][1]), 3, (0, 0, 255), 5)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, '1', (points[0][0], points[0][1]), font, 1, (255, 0, 0), 2)
    cv2.putText(img, '2', (points[1][0], points[1][1]), font, 1, (0, 255, 0), 2)
    cv2.putText(img, '3', (points[2][0], points[2][1]), font, 1, (0, 0, 255), 2)
    return img

def pad_square(img):
    old_size = img.shape
    new_size = tuple([np.max(img.shape),np.max(img.shape),img.shape[2]])#
    scale = (np.array(new_size)-np.array(old_size))/2
    top,bottom,left,right = int(scale[0]),int(new_size[0])-int(old_size[0])-int(scale[0]),int(scale[1]),int(new_size[1])-int(old_size[1])-int(scale[1])
    new_im = cv2.copyMakeBorder(img,top,bottom,left,right,cv2.BORDER_CONSTANT,value=[0,0,0])
    return new_im

def norm_lightness(img,mean_std):
    mean = np.sum(img)/np.sum(img!=0)
    scale = mean_std/mean
    img = img * scale
    return img

def flip(image,points):
    image = cv2.flip(image,1)
    landmarks = np.array(points).reshape(-1,2)
    landmarks[:,0] = image.shape[1] - landmarks[:,0]
    landmarks = landmarks.reshape(-1).tolist()
    return image,landmarks

def hand_adjust(hand_img,points):
    expected_mean = 120
    (x1,y1,x2,y2,x3,y3) = points
    #Rotate the image
    delta_vec = np.array([(x1-x3),(y1-y3)])
    y_vec = np.array([0,-1])
    angle_pi = np.sum(delta_vec*y_vec)/(np.linalg.norm(delta_vec)*np.linalg.norm(y_vec))
    angle = np.arccos(angle_pi)/3.1415926*180
    if x1 <= x3:
        angle *= -1
    rotate_image,rotate_points = Rotate(hand_img,angle,points)
    x1,x2,x3 = rotate_points[0],rotate_image[2],rotate_points[4]
    if (x2 < x1) and (x2 < x3):
        rotate_image,rotate_points = flip(rotate_image,rotate_points)
    rotate_points = np.array(rotate_points).reshape(-1,2)
    x11,x21,x31 = rotate_points[0,0],rotate_points[1,0],rotate_points[2,0]
    y11,y21,y31 = rotate_points[0,1],rotate_points[1,1],rotate_points[2,1]
    y_start = np.max([int(y11-30),0])
    y_end = np.min([512,int(y11 + (y31 - y11)*512.0/400)])
    ma = rotate_image[y_start:y_end,:,:]
    gray = cv2.cvtColor(ma,cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    contours = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cont = contours[np.argmax([len(con) for con in contours])]
    x_min = np.min(cont[:,0,0])
    x_max = np.max(cont[:,0,0])
    x_start = np.max([x_min-20,0])
    x_end = np.max([np.min([x_max+20,512]),int(x21+20)])
    x_end = np.min([x_end,512])
    rotate_points = rotate_points - [x_start,y_start]
    pad_shape = np.array([x_end - x_start, y_end - y_start])
    pad = np.array([np.max(pad_shape),np.max(pad_shape)]) - pad_shape
    pad_points = rotate_points + (pad/2.0).astype(np.int32)
    pad_image = pad_square(rotate_image[y_start:y_end,x_start:x_end,:])
    resize_image = cv2.resize(pad_image,(512,512),interpolation=cv2.INTER_LANCZOS4)
    final_points = (pad_points*1.0/pad_image.shape[0]*512).astype(np.int32).reshape(-1).tolist
    # final_image = norm_lightness(resize_image,expected_mean)
    final_image = resize_image
    return final_image,final_points

if __name__ == '__main__':
    #x1,y1 Middle x2,y2 Thumb x3,y3 carpal bones
    delta_vec = np.array([100,-100])
