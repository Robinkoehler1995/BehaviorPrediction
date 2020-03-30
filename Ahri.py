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

def get_distance(p1,p2):
    return round(math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2),2)

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

def stack_different_imgs(imgs,cnt):
    stack = list()
    for img in imgs:
        img_tmp, bb = ia.get_fragment(img,cnt)
        stack.append(img_tmp)
    img_stacked = np.dstack(stack)
    return img_stacked, bb

def process_seperation(sep,mask):
    k = 5
    kernel = np.ones((k,k),dtype=np.uint8)
    mask = cv2.erode(mask,kernel)
    modifier = np.less(sep[:,:,1],sep[:,:,0])
    sep[:,:,0] = sep[:,:,0]*modifier.astype(np.uint8)
    sep[:,:,1] = sep[:,:,1]*(1-modifier).astype(np.uint8)
    
    sep = cv2.bitwise_and(sep,sep,mask=mask)
    sep = dm.binary_it(sep,30)
    return sep

def combine_pos_neg(pos,neg,k=100):
    pos = dm.binary_it(pos,100,mask=True)
    neg = dm.binary_it(255-neg,100,mask=True)
    return cv2.bitwise_and(pos,neg)

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

kernel_size_conv = 9
net_rcdnn = nh.create_rcdnn_3conv_hidden(kernel_size_conv,1,1,4096,start_dim=1,end_dim=2)
hidden = net_rcdnn.init_hidden()
model_path = 'rcdnn'
net_rcdnn.load_state_dict(torch.load(model_path))
net_rcdnn.eval()
dim_rcdnn = (128,128)

kernel_size_conv = 15
net_class = nh.create_net_classifier(kernel_size_conv,device=device)
model_path = 'class'
net_class.load_state_dict(torch.load(model_path))
net_class.eval()
dim_class = (128,128)

kernel_size_conv = 15
net_class_d = nh.create_net_classifier(kernel_size_conv,device=device,classes=4)
model_path = 'class_double'
net_class_d.load_state_dict(torch.load(model_path))
net_class_d.eval()
dim_class_d = (128,128)
print('initilazied nets')

#init video reader
cap = cv2.VideoCapture('test.avi')

#init storage varialbes
centers_last = ((0,0),(100,100))
behavior_list_1 = list()
behavior_list_2 = list()
last = list()

#start analyzing
while 1:
    ret,img = cap.read()
    if not ret:
        break
    #dm.lucid_img(img,'img')
    img_whole = nh.evaluate_plus_resizing(net_whole,img,dim_whole,device=device,normalize=nh.pytorch_normalize_3) 
        
        
        
    cnts = ia.filtered_contours(img_whole)
    img_whole_plus_fragment = np.zeros(img_whole.shape,dtype=np.uint8)
    for cnt in cnts:
        img_tmp, bb = ia.get_fragment(img,cnt)
        pos = nh.evaluate_plus_resizing(net_fragment,img_tmp,dim_fragment,device=device,normalize=nh.pytorch_normalize_3)
        neg = nh.evaluate_plus_resizing(net_fragment_negative,img_tmp,dim_fragment,device=device,normalize=nh.pytorch_normalize_3)
        img_fragment = combine_pos_neg(pos,neg)
        img_fragment = ia.remove_filtered_contours(img_fragment)
        
        
        img_whole_plus_fragment[bb[0]:bb[1],bb[2]:bb[3]] += img_fragment
     
    cnts = ia.filtered_contours(img_whole_plus_fragment)
    
    sm = img.copy()
    if len(cnts) == 1:
        class_this,_ = ia.get_fragment(img,cnts[0])
        class_this_mask,_ = ia.get_fragment(img,cnts[0],mask_it=True)
        
        class_this,_ = ia.get_fragment(img,cnts[0],mask_it=True)
        out_class = nh.classify(net_class_d,class_this,dim_class,device=device,round_after=2)
        out_class = list(out_class)
        
        class_max = np.argmax(out_class)
        if class_max == 3 or class_max == 2:
            behavior_list_1.append([out_class[-1]]+[0,0]+out_class[0:3])
            behavior_list_2.append([out_class[-1]]+[0,0]+out_class[0:3])
        else:
            behavior_list_1.append([out_class[-1]]+[0,0]+out_class[0:3])
            behavior_list_2.append([0,0,0,0,0,0])
    if len(cnts) == 2:
        
        
        centers = ia.center_of_contours(cnts)
        if get_distance(centers[0],centers_last[0]) + get_distance(centers[1],centers_last[1]) >= get_distance(centers[0],centers_last[1]) + get_distance(centers[1],centers_last[0]):
            cnts = [cnts[1],cnts[0]]
            centers = [centers[1],centers[0]]
        
        centers_last = centers 
        test = False
        for cnt in cnts:
                
                
            class_this,_ = ia.get_fragment(img,cnt,mask_it=True)
            out_class = nh.classify(net_class,class_this,dim_class,device=device,round_after=2)


            spacing_amount = 20
            class_max = np.argmax(out_class)
            out_class = list(out_class)
            if len(behavior_list_1) <= len(behavior_list_2):
                behavior_list_1.append(out_class+[0,0,0])
            else:
                behavior_list_2.append(out_class+[0,0,0])
        #if test:    
        #    dm.lucid_img(sm,'a',1)
        
        
    #if lucidDreaming(True):
    #    break
    if len(behavior_list_1) == 3000:
        break
        
cv2.destroyAllWindows()


writer = open('behavior_list_1','w+')
for i in behavior_list_1:
    for j in i:
        writer.write(str(j)+'\t')
    writer.write('\n')
writer.flush()
writer.close()

writer = open('behavior_list_2','w+')
for i in behavior_list_2:
    for j in i:
        writer.write(str(j)+'\t')
    writer.write('\n')
writer.flush()
writer.close()