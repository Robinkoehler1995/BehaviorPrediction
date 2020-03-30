#torch libaries
from torchvision import transforms,datasets
import torch, torchvision, torch.optim as optim, torch.nn as nn, torch.nn.functional as F
#general libaries
from tqdm  import tqdm
import matplotlib.pyplot as plt, cv2
import numpy as np, sys, os, cv2, random, math, time
#own libaries
import data_manipulater as dm, net_handler as nh

def padded_bounding_box(img,cnt,padding=30):
    rect = cv2.boundingRect(cnt)
    boundray_first = max(rect[1]-padding,0)
    boundray_second = min(rect[1]+rect[3]+padding,img.shape[0])
    boundray_thrid = max(rect[0]-padding,0)
    boundray_fourth = min(rect[0]+rect[2]+padding,img.shape[1])
    return [boundray_first,boundray_second,boundray_thrid,boundray_fourth]

def center_of_contours(cnts):
    centers = list()
    for cnt in cnts:
        moments = cv2.moments(cnt)
        x = int(moments['m10']/moments['m00'])
        y = int(moments['m01']/moments['m00'])
        centers.append((x,y))
    return centers

def filtered_contours(binary,threshold = 200000):
    cnts,_ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filtered = list()
    for cnt in cnts:
        tmp = np.zeros(binary.shape, np.uint8)
        tmp = cv2.drawContours(tmp, [cnt], -1, 255, -1)
        tmp = cv2.bitwise_and(binary,binary,mask=tmp)
        
        if np.sum(tmp) > 200000:
            filtered.append(cnt)
    return filtered

def remove_filtered_contours(binary,threshold = 1000000):
    cnts = filtered_contours(binary,threshold)
    mask = np.zeros(binary.shape,dtype=np.uint8)
    mask = cv2.drawContours(mask, cnts, -1, 255, -1)
    binary = cv2.bitwise_and(binary,binary,mask=mask)
    return binary

def get_fragment(img,cnt,padding=30,mask_it=False):
    bb = padded_bounding_box(img,cnt,padding)
    if mask_it:
        mask = np.zeros(img.shape[:2],np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        img = cv2.bitwise_and(img,img,mask=mask)
    fragment_img = img[bb[0]:bb[1],bb[2]:bb[3]]
    return fragment_img, bb

def get_max_contour(binary):
    cnts,_ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnt_max = []
    cnt_max_area = 0
    for cnt in cnts:
        tmp = np.zeros(binary.shape, np.uint8)
        tmp = cv2.drawContours(tmp, [cnt], -1, 255, -1)
        tmp = cv2.bitwise_and(binary,binary,mask=tmp)
        
        if np.sum(tmp) > cnt_max_area:
            cnt_max = cnt
            cnt_max_area = np.sum(tmp)
    return cnt_max

def get_kernel(k):
    return np.ones((k,k),dtype=np.uint8)

