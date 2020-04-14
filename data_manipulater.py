#!/usr/bin/python3.6
#library which contains function to manipulated training data

#torch libaries
from torchvision import transforms,datasets
import torch, torchvision, torch.optim as optim, torch.nn as nn, torch.nn.functional as F
#general libaries
from tqdm  import tqdm
import matplotlib.pyplot as plt, cv2
import numpy as np, sys, os, cv2, random, math, time
#own libaries
import net_handler as nh

#nomalize the output of an ANN to a range of 0 to 1
def pytorch_normalize(img):
    img = img-np.min(img)
    if np.max(img) != 0:
        img = img/np.max(img)#.astype(np.uint8)
    return img
  

#nomalize the output of an ANN to a range of 0 to 1 by clipping values
def pytorch_normalize_2(img):
    img = np.clip(img,0,1)
    return img

#change the format of batch of images from the pytorch standard to the opencv standard
def pytorch_to_opencv(batch):
    batch = batch.astype(np.float64)
    batch = np.clip(batch,0,1)*255.0
    if batch.shape[1]==1:
        batch = np.squeeze(batch,axis=1)
    else:
        batch = np.swapaxes(batch,2,3)
        batch = np.swapaxes(batch,1,3)
    batch = batch.astype(np.uint8)
    return batch

#change the format of batch of images from the opencv standard to the pytorch standard
def opencv_to_pytorch(batch):
    batch = batch.astype(np.float64)
    batch = batch/255.0
    if len(batch.shape) == 4:
        batch = np.swapaxes(batch,1,3)
        batch = np.swapaxes(batch,2,3)
    else:
        batch = np.expand_dims(batch,axis=1)
    return batch

#applz a histgram nornalization to a multi-channel image
def histogram_normalization(img):
    if len(img.shape) == 2:
        return cv2.equalizeHist(img)
    else:
        layer = list()
        for i in range(img.shape[2]):
            layer.append(cv2.equalizeHist(img[:,:,i]))
        return np.stack(layer,axis=2)

#generates an image containing all edges found in the given image "img"
def generate_edges(img,val_1=15,val_2=30):
    edges = np.stack((cv2.Canny(img,val_1,val_2),cv2.Canny(img,val_1*2,val_2*2),cv2.Canny(img,val_1*4,val_2*4)),axis=2)
    return np.dstack((img,edges))

#generates an image containing all edges found in the given image "img"
def generate_edges_2(img,weights=[(15,30),(30,60),(60,120)],stack=True,kernel_size=3):
    kernel = np.ones((kernel_size,kernel_size),dtype=np.uint8)
    edges = np.stack([cv2.morphologyEx(cv2.Canny(img,val_1,val_2), cv2.MORPH_DILATE,kernel) for val_1,val_2 in weights],axis=2)
    if stack:
        return np.dstack((img,edges))
    else:
        return edges

#generates an image containing all edges found in the given image "img"
def generate_edges_3(img,weights=[(15,30),(30,60),(60,120)],stack=True,kernel_size=[1]):
    edges = np.zeros(img.shape[:2],dtype=np.float64)
    for val_1,val_2 in weights:
      for ks in kernel_size:
        kernel = np.ones((ks,ks),dtype=np.uint8)
        tmp = cv2.morphologyEx(cv2.Canny(img,val_1,val_2), cv2.MORPH_DILATE,kernel).astype(np.float64)
        edges += tmp/len(weights)/len(kernel_size)
    #out /= len(weights)
    edges = edges.astype(np.uint8)
    if stack:
        return np.dstack((img,edges))
    else:
        return edges

#modifies an image by adding additional channels to it which contain detected edges
def big_img_manipulation(img,mask=np.array([])):
    out = dm.histogram_normalization(img)
    out = (255-out)
    edges = generate_edges_3(out,stack=False,kernel_size=[1],weights=[(x,8*x) for x in [5,20,80,160]])
    edges = 1-edges/255.0
    out = out.astype(np.float64) * np.stack((edges,edges,edges),axis=2)
    out = out.astype(np.uint8)
    if mask.shape[0]>0:
        out = cv2.bitwise_and(out,out,mask=mask)
    return out

#creats a binary image with a given threshold 
def binary_it(img,t,mask=False):
    tmp = (np.clip(img,t,t+1)-t)*255
    if mask:
        tmp = cv2.bitwise_and(img,img,mask=tmp)
    return tmp

#dream beyond
def lucidDreaming(wait=True):
    while 1:
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            print('keyboard interuption: its getting dark')
            return True
        if k == 32:
            return False
        if not wait:
            return False

#displays a given image by using functions from opencv
def display_img(img,name='default',resize=np.array([])):
    imgs = split_img(img,resize)
    for i in range(len(imgs)):
        cv2.imshow(name+'_'+str(i),imgs[i])

#displays a given image by using functions from opencv
def lucid_img(img,name='default',wait=0):
    while len(img.shape) < 3:
        img = np.expand_dims(img,len(img.shape))
    if img.shape[2]%3==2:
        black = np.zeros(list(img.shape[:2])+[1],np.uint8)
        img = np.dstack((img,black))
    for j in range(0,img.shape[2],3):
        tmp = img[:,:,j:j+3]
        cv2.imshow(name+str(int(j/3)),tmp)
    return lucidDreaming(wait)

#split an image into single channels
def split_img(img,resize=np.array([])):
    if len(img.shape) < 3:
        tmp = img
        if resize.shape[0]>0:
            tmp = cv2.resize(tmp, resize, interpolation = cv2.INTER_AREA)
        return [tmp]
    
    out = list()
    i = -3
    for i in range(0,img.shape[2]-2,3):
        tmp = img[:,:,i:i+3]
        if resize.shape[0]>0:
            tmp = cv2.resize(tmp, resize, interpolation = cv2.INTER_AREA)
        out.append(tmp)
    for j in range(i+3,img.shape[2]):
        tmp = img[:,:,j]
        if resize.shape[0]>0:
            tmp = cv2.resize(tmp, resize, interpolation = cv2.INTER_AREA)
        out.append(tmp)
    return out

#flip and append a data set to itself
def flip_data(data,mirror=2,rotate=4):
    out = list()
    for entry in data:
        for a in range(mirror):
            for b in range(rotate):
                out.append(entry)
                
                entry = np.flip(entry,0)
                entry = np.swapaxes(entry,0,1)
            entry = np.flip(entry,0)
    return np.array(out)

#inverse an stack of labels
def inverse_labels(labels):
    out = list()
    for label in labels:
        out.append(255-label)
    return np.array(out)