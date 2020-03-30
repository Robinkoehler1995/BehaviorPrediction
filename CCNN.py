#torch libaries
from torchvision import transforms,datasets
import torch, torchvision, torch.optim as optim, torch.nn as nn, torch.nn.functional as F
#general libaries
from tqdm  import tqdm
import matplotlib.pyplot as plt, cv2
import numpy as np, sys, os, cv2, random, math, time
#own libaries
import net_handler as nh, data_manipulater as dm, loss_function as lf

    
def train(net):
    loss_train_over_time = list()
    loss_test_over_time = list()
    BATCH_SIZE = 80
    epochs = 10
    
    for epoch in range(epochs):
        for i in tqdm(range(0,len(X_train),BATCH_SIZE)):
            X_batch = X_train[i:i+BATCH_SIZE]
            y_batch = y_train[i:i+BATCH_SIZE]
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            net.zero_grad()
            output = net(X_batch)
            loss = loss_function(output, y_batch)
            loss.backward()
            optimizer.step()
            
        loss_test = loss_function(evaluate(net,X_test[0:BATCH_SIZE].to(device)),y_test[0:BATCH_SIZE].to(device)).item()
        loss_train_over_time.append(loss.item())
        loss_test_over_time.append(loss_test)
        
        print('current epoch:',str(epoch+1)+'/'+str(epochs))
        print('current loss:',loss.item())
        print('test set loss:',loss_test)
        time.sleep(0.5)
    return loss_train_over_time, loss_test_over_time

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


X = np.load('X_class.npy',allow_pickle = True)
y = np.load('y_class.npy',allow_pickle = True)
X = np.vstack((X,np.flip(X,0)))
y = np.vstack((y,y))

X_t = dm.opencv_to_pytorch(X)
y_t = y

test_size = 50
X_test = torch.Tensor(X_t[:test_size])
y_test = torch.Tensor(y_t[:test_size])

np.random.seed(1)
np.random.shuffle(X_t)
np.random.seed(1)
np.random.shuffle(y_t)

X_train = torch.Tensor(X_t[test_size:])
y_train = torch.Tensor(y_t[test_size:])

print('X:',X_train.shape,y_train.shape)
print('y:',X_test.shape,y_test.shape)

kernel_size_conv = 15
net = nh.create_net_classifier(kernel_size_conv,classes=4,start_dim=3,device=device)
print(net)
print('net intilazied')

#initlize optimizer
optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = lf.mse

#train
loss_train_over_time, loss_test_over_time = train(net)

#draw plot
fig, ax = plt.subplots()
ax.set_title('Behavior')
ax.plot(range(len(loss_train_over_time)), loss_train_over_time,label='Training',color=(32/255,178/255,170/255))
ax.plot(range(len(loss_test_over_time)), loss_test_over_time,label='Test',color=(127/255,1,0))
ax.set_xlabel('Epochs')
ax.set_ylabel('Loss')
leg = ax.legend()
fig.savefig('loss_over_time_behavior.png')

#save net
model_path = '../CheckPoint/class'
torch.save(net.state_dict(), model_path)