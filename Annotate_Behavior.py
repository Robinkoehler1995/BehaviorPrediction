#!/usr/bin/python3.6
#this script creates traninig data to create an ANN which classifies behavior
#this is done by creating images of cropped out mice and annotating it depending on the behavior which is displayed
#0 exploring
#1 grooming
#2 rearing
#3 keep down
#3 close following
#4 sniffing


#torch libaries
from torchvision import transforms,datasets
import torch, torchvision, torch.optim as optim, torch.nn as nn, torch.nn.functional as F
#general libaries
from tqdm  import tqdm
import matplotlib.pyplot as plt, cv2
import numpy as np, sys, os, cv2, random, math, time
#own libaries
import net_handler as nh, data_manipulater as dm, image_analyser as ia

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

#init video reader
base = 'test.avi'
path_vid = base
cap = cv2.VideoCapture(path_vid)

#storage variables which will be annotated
class_this_list_1 = list()
class_this_list_2 = list()
class_this_list = list()
class_this_double_list = list()
behavior_list_1 = list()
behavior_list_2 = list()
centers_last = ((0,0),(100,100))
last = list()

print("start collecting data")
while 1:
    #read current frame
    ret,img = cap.read()
    #if video ended break while loop
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
        img_fragment = ia.remove_filtered_contours(img_fragment)
        
        
        img_whole_plus_fragment[bb[0]:bb[1],bb[2]:bb[3]] += img_fragment
     
    #find all the object from the second prediction
    cnts = ia.filtered_contours(img_whole_plus_fragment)
    
    #overlapping mice add to social interaction
    if len(cnts) == 1:
        class_this,_ = ia.get_fragment(img,cnts[0])
        class_this_mask,_ = ia.get_fragment(img,cnts[0],mask_it=True)
        class_this_double_list.append((class_this,class_this_mask))
        
    #single mice add to single behavior
    if len(cnts) == 2:
        #determine which mice is which
        centers = ia.center_of_contours(cnts)
        if get_distance(centers[0],centers_last[0]) + get_distance(centers[1],centers_last[1]) >= get_distance(centers[0],centers_last[1]) + get_distance(centers[1],centers_last[0]):
            cnts = [cnts[1],cnts[0]]
            centers = [centers[1],centers[0]]
        #save data which will be annotated 
        centers_last = centers
        for cnt in cnts:
            class_this,_ = ia.get_fragment(img,cnt)
            class_this_mask,_ = ia.get_fragment(img,cnt,mask_it=True)
            
            if len(class_this_list_1) <= len(class_this_list_2):
                class_this_list_1.append((class_this,class_this_mask))
            else:
                class_this_list_2.append((class_this,class_this_mask))
        
    elif len(class_this_list_1) > 0:
        class_this_list.append(class_this_list_1)
        class_this_list.append(class_this_list_2)
        class_this_list_1 = list()
        class_this_list_2 = list()
        
    #break if 3000 elements were selected
    if len(class_this_list) == 3000:
        break
        

#shuffle single behavior
random.seed(42)
np.random.shuffle(class_this_list)
tmp = list()
for it in class_this_list:
    tmp+=it


behavior = list()
#-1 not determinable
#0 exploring
#1 grooming
#2 rearing
print('Q rearing, W grooming, E exploring, esc quit, space skip')
class Found(Exception): pass
#annotated single behavior by using the keybord
#an image is shown to you and by pressing Q W E or space the image is annotated
try:
    for list_tmp in class_this_list:
        print('next_list')
        for index,split_this in zip(range(len(list_tmp)),list_tmp):
            img,cropped = split_this
            cv2.imshow('img',img) 

            kk = 'filler'
            while(True):
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    print('its getting dark')
                    raise Found
                if k == 113:#Q
                    behavior.append(2)
                    print('added',index,'to rearing')
                    break
                if k == 119:#W
                    behavior.append(1)
                    print('added',index,'to grooming')
                    break
                if k == 101:#E
                    behavior.append(0)
                    print('added',index,'to exploring')
                    break
                if k == 32:
                    print('added',index,'to not determinable')
                    break

        #plt.imshow(img_without_background)
except Found:
    pass
print(len(behavior))
cv2.destroyAllWindows()



behavior = list()
#-1 not determinable
#3 keep down
#4 close following
#5 sniffing
#0 exploring
print('Q sniffing, W close following, E keeping down, R exploring, esc quit, space skip')
#annotated social behavior by using the keybord
#an image is shown to you and by pressing Q W E R or space the image is annotated
try:
    for index,split_this in zip(range(len(class_this_double_list)),class_this_double_list):
        img,cropped = split_this
        cv2.imshow('img',img) 

        kk = 'filler'
        while(True):
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                print('its getting dark')
                raise Found
            if k == 113:#Q
                behavior.append((cropped,5))
                print('added',index,'to sniffing')
                break
            if k == 119:#W
                behavior.append((cropped,4))
                print('added',index,'to close following')
                break
            if k == 101:#E
                behavior.append((cropped,3))
                print('added',index,'to keeping down')
                break 
            if k == 114:#R
                behavior.append((cropped,0))
                print('add, ',index,'to Nothing')
                break
            if k == 32:#space
                print('added',index,'to not determinable')
                break

except Found:
    pass
print(len(behavior))
cv2.destroyAllWindows()


X = list()
y = list()
#formate annotated behavior and save it 
for cropped, behav in behavior:
    cropped = cv2.resize(cropped, (128,128), interpolation = cv2.INTER_AREA)
    X.append(cropped)
    
    ohv = [0]*(6)
    ohv[behav] = 1
    y.append(ohv)
X = np.array(X)
y = np.array(y)
print(X.shape,y.shape)
np.save('X_class.npy',X)
np.save('y_class.npy',y)