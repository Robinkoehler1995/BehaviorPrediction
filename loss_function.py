#torch libaries
from torchvision import transforms,datasets
import torch, torchvision, torch.optim as optim, torch.nn as nn, torch.nn.functional as F
#general libaries
from tqdm  import tqdm
import matplotlib.pyplot as plt, cv2
import numpy as np, sys, os, cv2, random, math, time
#own libaries
import net_handler as nh, data_manipulater as dm

def dice(pred, target):
    smooth = 1.
    pred = torch.clamp(pred,0,1)
    target = torch.clamp(target,0,1)
    
    intersection = (target * pred)
    return 1-(2*intersection.sum()+smooth)/(pred.sum()+target.sum()+smooth)

def cross_entropy(pred,target):
    pred = torch.clamp(pred,0.00001,0.99999)
    target = target
    ce1 = target*torch.log2(pred)
    ce2 = (1-target)*torch.log2(1-pred)
    return -(ce1+ce2).sum()/pred.reshape(-1).size(0)

def mse(pred,target):
    mean_squared_error = ((pred-target)**2).sum()/pred.reshape(-1).size(0)
    return mean_squared_error