#torch libaries
from torchvision import transforms,datasets
import torch, torchvision, torch.optim as optim, torch.nn as nn, torch.nn.functional as F
#general libaries
from tqdm  import tqdm
import matplotlib.pyplot as plt, cv2
import numpy as np, sys, os, cv2, random, math, time
#own libaries
import net_handler as nh, data_manipulater as dm, loss_function as lf

def custom_loss(pred,target):
    #target = target[:,0:1]
    pred = torch.clamp(pred,0,1)
    return (lf.dice(pred,target)+lf.mse(pred,target)) / 2

def custom_loss_sep_min(pred,target):
    out = 0
    for t1, t2 in zip(pred,target):
        #print(t1.shape,t2.shape)
        a = custom_loss(t1,t2)
        b = custom_loss(torch.stack((t1[:,1],t1[:,0]),1),t2)
        out += torch.min(a,b)
    return out/pred.size(0)
    
def train(net):
    loss_train_over_time = list()
    BATCH_SIZE = 30
    epochs = 1000
    for epoch in range(epochs):
        print('current epoch:',str(epoch+1)+'/'+str(epochs))
        loss_this_epoch = 0
        for i in range(0,train_X.shape[0],BATCH_SIZE):
            X_batch = torch.Tensor(train_X[i:i+BATCH_SIZE])
            y_batch = torch.Tensor(train_y[i:i+BATCH_SIZE])
            
            #X_batch = X_batch.to(device)

            net.zero_grad()
            output, hidden_out = net(X_batch, hidden)
            loss = loss_function(output, y_batch)
            print('this loss:',loss.item())
            loss_this_epoch = (loss_this_epoch + loss.item())/2
            loss.backward()
            optimizer.step()
            
            model_path = '../CheckPoint/tmp'
            torch.save(net.state_dict(), model_path)
            print('model saved')
        
        del(X_batch)    
        del(output)
        torch.cuda.empty_cache()
        
        if epoch % 100 == 0:
            model_path = '../CheckPoint/tmp'+str(epoch)
            torch.save(net.state_dict(), model_path)
        
        print('epoch',epoch,'loss:',loss_this_epoch)
        loss_train_over_time.append(loss.item())

    return loss_train_over_time

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

X = np.load('../Data/TrainingData/Numpy/X_sep_rcdnn.npy',allow_pickle=True)
y = np.load('../Data/TrainingData/Numpy/y_sep_rcdnn.npy',allow_pickle=True)
train_X = X[0:330]
train_y = y[0:330]
print('Training_set_shape',X.shape,y.shape)

kernel_size_conv = 9
net = nh.create_rcdnn_3conv_hidden(kernel_size_conv,30,100,4096,start_dim=1,end_dim=2)
hidden = net.init_hidden()
print(net)
print('net intilazied')

#init optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = custom_loss_sep_min

#training
loss_train_over_time = train(net)

#draw loss plot
loss = loss_train_over_time
fig, ax = plt.subplots()
ax.set_title('Overlapping Mice')
color_codex = [(170,178,32),(0,215,255),(0,255,127)]
behavior_codex = ['Exploring','Rearing','Grooming']
ax.plot(range(len(loss)), loss,label='Training',color=(32/255,178/255,170/255))
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
leg = ax.legend()
fig.savefig('loss_over_time_rcdnn.png')

#save rcdnnn
model_path = '../CheckPoint/rcdnn'
torch.save(net.state_dict(), model_path)







