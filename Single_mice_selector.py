#!/usr/bin/python3.6
#this script generates list of sequences of single mice
#these sequences are needed to to train a ANN which is able to separated overlapping mice
#by using a video of two mice it is easy to determine if mice are separated
#just by checking of two shapes are predicted by the ANN "whole" and "fragment"
#if two shape are predicted the mice are separated and they are added to the output

#torch libaries
from torchvision import transforms,datasets
import torch, torchvision, torch.optim as optim, torch.nn as nn, torch.nn.functional as F
#general libaries
from tqdm  import tqdm
import matplotlib.pyplot as plt, cv2
import numpy as np, sys, os, cv2, random, math, time
#own libaries
import net_handler as nh, data_manipulater as dm, image_analyser as ia

#Check if cuda is available
print('Is cuda available:\t'+ str(torch.cuda.is_available()))
print('Available devices:\t'+str(torch.cuda.device_count()))
print('Device 0 name:\t'+ torch.cuda.get_device_name(0))
torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:\t'+ str(device))
if device.type == 'cuda':
    print('Memory Usage:')
    print('\tAllocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('\tCached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

#used to proper dislpay an image with opencv
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

#calculates the distance between two points
def get_distance(p1,p2):
    return round(math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2),2)

#reutrn the contour of the biggest object in a binary image
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

#return a stack of images "img_stacked" containing only a specific region "cnt" of another stack of images "imgs"
def stack_different_imgs(imgs,cnt):
    stack = list()
    for img in imgs:
        img_tmp, bb = ia.get_fragment(img,cnt)
        stack.append(img_tmp)
    img_stacked = np.dstack(stack)
    return img_stacked, bb

#combines the negative and positive prediction of two ANN
def combine_pos_neg(pos,neg,k=100):
    pos = dm.binary_it(pos,100,mask=True)
    neg = dm.binary_it(255-neg,100,mask=True)
    return cv2.bitwise_and(pos,neg)

#remove elements of a given lists to always ensure the same length of the list
def keep_list_at_size(lists,size):
    for l in lists:
        while(len(l) > size):
            del(l[0])

#load trained ANN        
kernel_size_conv = 9
net_whole = nh.create_net_3conv(kernel_size_conv,device=device)
model_path = 'whole'
net_whole.load_state_dict(torch.load(model_path))
net_whole.eval()
dim_whole = (128,128)

kernel_size_conv = 9
net_fragment = nh.create_net_3conv(kernel_size_conv,device=device)
model_path = 'fragment'
net_fragment.load_state_dict(torch.load(model_path))
net_fragment.eval()
dim_fragment = (128,128)

kernel_size_conv = 9
net_fragment_negative = nh.create_net_3conv(kernel_size_conv,device=device)
model_path = 'fragment_negative'
net_fragment_negative.load_state_dict(torch.load(model_path))
net_fragment_negative.eval()

#set up video path
base = 'test.avi'
path_vid = base

#init video reader
cap = cv2.VideoCapture(path_vid)

#init variables
centers_last = ((0,0),(100,100))
list_size = 100
X_pre_seperator = list()
y_pre_seperator = list()
index_pre_seperator = list()
sequences = [[[],[]],[[],[]]]
new_sequence = 0
#until the loop breaks
while 1:
    #read current frame
    ret,img = cap.read()
    #if current frame does not exist break
    if not ret:
        break
    
    #make a first prediction for the whole current frame
    img_whole = nh.evaluate_plus_resizing(net_whole,img,dim_whole,device=device,normalize=nh.pytorch_normalize_3) 
        
    #find contour of the shape of the first prediction
    cnts = ia.filtered_contours(img_whole)
    #make a second more accurate prediciton for each object of the first prediction
    #and combine them into a new image "img_whole_plus_fragment"
    img_whole_plus_fragment = np.zeros(img_whole.shape,dtype=np.uint8)
    for cnt in cnts:
        img_tmp, bb = ia.get_fragment(img,cnt)
        pos = nh.evaluate_plus_resizing(net_fragment,img_tmp,dim_fragment,device=device,normalize=nh.pytorch_normalize_3)
        neg = nh.evaluate_plus_resizing(net_fragment_negative,img_tmp,dim_fragment,device=device,normalize=nh.pytorch_normalize_3)
        img_fragment = combine_pos_neg(pos,neg)
        img_fragment = cv2.erode(img_fragment,ia.get_kernel(8))
        img_fragment = ia.remove_filtered_contours(img_fragment)
        img_whole_plus_fragment[bb[0]:bb[1],bb[2]:bb[3]] += img_fragment
    
    #find all the object from the second prediction
    cnts = ia.filtered_contours(img_whole_plus_fragment)
    
    #if mice are seperated
    if len(cnts) == 2:
        #determine which mice is which
        centers = ia.center_of_contours(cnts)
        if get_distance(centers[0],centers_last[0]) + get_distance(centers[1],centers_last[1]) >= get_distance(centers[0],centers_last[1]) + get_distance(centers[1],centers_last[0]):
            cnts = [cnts[1],cnts[0]]
            centers = [centers[1],centers[0]]
        centers_last = centers
        #append each mice to sequences
        for index,cnt in zip([0,1],cnts):
            single_pred = np.zeros(img_whole_plus_fragment.shape, np.uint8)
            single_pred = cv2.drawContours(single_pred, [cnt], -1, 255, -1)
            single_mice = cv2.bitwise_and(img,img,mask=single_pred)
            sequences[index][0].append(single_mice)
            sequences[index][1].append(single_pred)

        #display sequence
        dm.lucid_img(img,'frame')
        dm.lucid_img(img_whole_plus_fragment,'binary')
    #if mice are overlapping
    else:
        #if the sequences of seperated mice are longer than 100 frames add them to the output "X_pre_seperator" and "X_pre_seperator" plus the start and end position of these sequences to "index_pre_seperator" 
        if len(sequences[0][0]) > 99:
            for sequence in sequences:
                X, y = sequence
                start = len(X_pre_seperator)
                X_pre_seperator += X
                y_pre_seperator += y
                end = len(X_pre_seperator)
                index_pre_seperator.append(['filler',start,end])
                sequences = [[[],[]],[[],[]]]

#save list of seperated mice
print('size:',np.array(X_pre_seperator).shape)
np.save('X_pre_sep.npy', np.array(X_pre_seperator))
np.save('y_pre_sep.npy', np.array(y_pre_seperator))
np.save('index_pre_seperator.npy',np.array(index_pre_seperator))
#close all open windows of opencv
cv2.destroyAllWindows()