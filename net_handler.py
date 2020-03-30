#torch libaries
from torchvision import transforms,datasets
import torch, torchvision, torch.optim as optim, torch.nn as nn, torch.nn.functional as F
#general libaries
from tqdm  import tqdm
import matplotlib.pyplot as plt, cv2
import numpy as np, sys, os, cv2, random, math, time
#own libaries
import data_manipulater as dm

def create_net_3conv(kernel_size,device=None,start_dim=3,end_dim=1):
    #kernel_size = 11 #its needs to be uneven, at least I think so
    padding = int(kernel_size/2)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            pm = 'same'
            self.conv1 = nn.Conv2d(start_dim, 16, kernel_size, padding=padding, padding_mode=pm)
            self.conv2 = nn.Conv2d(16, 64, kernel_size, padding=padding, padding_mode=pm)
            self.conv3 = nn.Conv2d(64, 256, kernel_size, padding=padding, padding_mode=pm)

            self.dconv1 = nn.ConvTranspose2d(256,64,kernel_size, padding=padding)
            self.dconv2 = nn.ConvTranspose2d(64*2,16,kernel_size, padding=padding)
            self.dconv3 = nn.ConvTranspose2d(16*2,end_dim,kernel_size, padding=padding)

            self.ident = 1
            self.ident_pad = 0
            self.iconv1 = nn.Conv2d(16,16,self.ident,padding=self.ident_pad, padding_mode=pm)
            self.iconv2 = nn.Conv2d(64,64,self.ident,padding=self.ident_pad, padding_mode=pm)

            self.cbn1 = nn.BatchNorm2d(16)
            self.cbn2 = nn.BatchNorm2d(64)
            self.cbn3 = nn.BatchNorm2d(256)

            self.dbn1 = nn.BatchNorm2d(64)
            self.dbn2 = nn.BatchNorm2d(16)
            self.dbn3 = nn.BatchNorm2d(end_dim)
            
            self.ibn1 = nn.BatchNorm2d(16)
            self.ibn2 = nn.BatchNorm2d(64)
            
            
        def conv2d_plus_pool(self, in_it_goes, conv, bn, pool=(2,2)):
            out = F.relu(conv(in_it_goes))
            out = F.max_pool2d(out,pool)
            out = bn(out)
            return out
        
        def conv2d_without_pool(self, in_it_goes, conv, bn):
            out = F.relu(conv(in_it_goes))
            out = bn(out)
            return out
        
        def conv3d_plus_stack(self, in_it_goes,conv,bn): #u could add bn at this point
            out = torch.stack(in_it_goes,axis=1)
            out = conv(out)
            out = torch.squeeze(out,axis=1)
            out = bn(out)
            return out
        
        def dconv2d_plus_pool(self, in_it_goes, dconv, bn, pool=(2,2)):
            out = F.interpolate(in_it_goes,scale_factor=2)
            out = dconv(out)
            out = F.relu(out)
            out = bn(out)
            return out
        
        def convs(self, x):
            #convs
            x = self.conv2d_plus_pool(x,self.conv1,self.cbn1)
            i1 = self.conv2d_without_pool(x,self.iconv1,self.ibn1)
            
            x = self.conv2d_plus_pool(x,self.conv2,self.cbn2)
            i2 = self.conv2d_without_pool(x,self.iconv2,self.ibn2)
            
            x = self.conv2d_plus_pool(x,self.conv3,self.cbn3)
            
            #dcons
            x = self.dconv2d_plus_pool(x,self.dconv1,self.dbn1)
            x = torch.cat((x, i2),1)
            
            x = self.dconv2d_plus_pool(x,self.dconv2,self.dbn2)
            x = torch.cat((x, i1),1)
            
            x = self.dconv2d_plus_pool(x,self.dconv3,self.dbn3)
            
            return x
        
        def forward(self, x):
            
            x = self.convs(x)
            return x
    
    if device == None:        
        net = Net()
    else:
        net = Net().to(device)
    return net

def create_net_3conv_without_skip(kernel_size,device=None,start_dim=3,end_dim=1):
    #kernel_size = 11 #its needs to be uneven, at least I think so
    padding = int(kernel_size/2)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            pm = 'same'
            self.conv1 = nn.Conv2d(start_dim, 16, kernel_size, padding=padding, padding_mode=pm)
            self.conv2 = nn.Conv2d(16, 64, kernel_size, padding=padding, padding_mode=pm)
            self.conv3 = nn.Conv2d(64, 256, kernel_size, padding=padding, padding_mode=pm)

            self.dconv1 = nn.ConvTranspose2d(256,64,kernel_size, padding=padding)
            self.dconv2 = nn.ConvTranspose2d(64,16,kernel_size, padding=padding)
            self.dconv3 = nn.ConvTranspose2d(16,end_dim,kernel_size, padding=padding)

            self.cbn1 = nn.BatchNorm2d(16)
            self.cbn2 = nn.BatchNorm2d(64)
            self.cbn3 = nn.BatchNorm2d(256)

            self.dbn1 = nn.BatchNorm2d(64)
            self.dbn2 = nn.BatchNorm2d(16)
            self.dbn3 = nn.BatchNorm2d(end_dim)
            
            
            
        def conv2d_plus_pool(self, in_it_goes, conv, bn, pool=(2,2)):
            out = F.relu(conv(in_it_goes))
            out = F.max_pool2d(out,pool)
            out = bn(out)
            return out
        
        def conv2d_without_pool(self, in_it_goes, conv, bn):
            out = F.relu(conv(in_it_goes))
            out = bn(out)
            return out
        
        def conv3d_plus_stack(self, in_it_goes,conv,bn): #u could add bn at this point
            out = torch.stack(in_it_goes,axis=1)
            out = conv(out)
            out = torch.squeeze(out,axis=1)
            out = bn(out)
            return out
        
        def dconv2d_plus_pool(self, in_it_goes, dconv, bn, pool=(2,2)):
            out = F.interpolate(in_it_goes,scale_factor=2)
            out = dconv(out)
            out = F.relu(out)
            out = bn(out)
            return out
        
        def convs(self, x):
            #convs
            x = self.conv2d_plus_pool(x,self.conv1,self.cbn1)
            
            x = self.conv2d_plus_pool(x,self.conv2,self.cbn2)
            
            x = self.conv2d_plus_pool(x,self.conv3,self.cbn3)
            
            #dcons
            x = self.dconv2d_plus_pool(x,self.dconv1,self.dbn1)
            
            x = self.dconv2d_plus_pool(x,self.dconv2,self.dbn2)
            
            x = self.dconv2d_plus_pool(x,self.dconv3,self.dbn3)
            
            return x
        
        def forward(self, x):
            
            x = self.convs(x)
            return x
    
    if device == None:        
        net = Net()
    else:
        net = Net().to(device)
    return net

def create_net_5conv(kernel_size,arcitecture = [3,16,32,64,128,256,1],device=None):
    #kernel_size = 11 #its needs to be uneven, at least I think so
    padding = int(kernel_size/2)
    padding_matrix = (padding,padding,padding,padding)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(arcitecture[0], arcitecture[1], kernel_size)
            self.conv2 = nn.Conv2d(arcitecture[1], arcitecture[2], kernel_size)
            self.conv3 = nn.Conv2d(arcitecture[2], arcitecture[3], kernel_size)
            self.conv4 = nn.Conv2d(arcitecture[3], arcitecture[4], kernel_size)
            self.conv5 = nn.Conv2d(arcitecture[4], arcitecture[5], kernel_size)

            self.dconv1 = nn.ConvTranspose2d(arcitecture[5],arcitecture[4],kernel_size,padding=padding)
            self.dconv2 = nn.ConvTranspose2d(arcitecture[4]*2,arcitecture[3],kernel_size,padding=padding)
            self.dconv3 = nn.ConvTranspose2d(arcitecture[3]*2,arcitecture[2],kernel_size,padding=padding)
            self.dconv4 = nn.ConvTranspose2d(arcitecture[2]*2,arcitecture[1],kernel_size,padding=padding)
            self.dconv5 = nn.ConvTranspose2d(arcitecture[1]*2,arcitecture[-1],kernel_size,padding=padding)

            self.ident = 1
            self.iconv1 = nn.Conv2d(arcitecture[1],arcitecture[1],self.ident)
            self.iconv2 = nn.Conv2d(arcitecture[2],arcitecture[2],self.ident)
            self.iconv3 = nn.Conv2d(arcitecture[3],arcitecture[3],self.ident)
            self.iconv4 = nn.Conv2d(arcitecture[4],arcitecture[4],self.ident)
            
            #self.econv1 = nn.Conv2d(arcitecture[-1], arcitecture[-1], 1)
            #self.econv2 = nn.Conv2d(arcitecture[-1], arcitecture[-1], 1)

            self.cbn1 = nn.BatchNorm2d(arcitecture[1])
            self.cbn2 = nn.BatchNorm2d(arcitecture[2])
            self.cbn3 = nn.BatchNorm2d(arcitecture[3])
            self.cbn4 = nn.BatchNorm2d(arcitecture[4])
            self.cbn5 = nn.BatchNorm2d(arcitecture[5])

            self.dbn1 = nn.BatchNorm2d(arcitecture[4])
            self.dbn2 = nn.BatchNorm2d(arcitecture[3])
            self.dbn3 = nn.BatchNorm2d(arcitecture[2])
            self.dbn4 = nn.BatchNorm2d(arcitecture[1])
            self.dbn5 = nn.BatchNorm2d(arcitecture[-1])
            
            self.ibn1 = nn.BatchNorm2d(arcitecture[1])
            self.ibn2 = nn.BatchNorm2d(arcitecture[2])
            self.ibn3 = nn.BatchNorm2d(arcitecture[3])
            self.ibn4 = nn.BatchNorm2d(arcitecture[4])
        
        def conv2d_plus_pool(self, in_it_goes, conv, bn, pool=(2,2)):
            out = F.pad(in_it_goes, padding_matrix, 'replicate')
            out = F.relu(conv(out))
            out = F.max_pool2d(out,pool)
            out = bn(out)
            return out
        
        def conv2d_without_pool(self, in_it_goes, conv, bn):
            out = F.relu(conv(in_it_goes))
            out = bn(out)
            return out
        
        def dconv2d_plus_pool(self, in_it_goes, dconv, bn, pool=(2,2)):
            out = F.interpolate(in_it_goes,scale_factor=2)
            out = dconv(out)
            out = F.relu(out)
            out = bn(out)
            return out
            
        def convs(self, x):
            #convs
            x = self.conv2d_plus_pool(x,self.conv1,self.cbn1)
            i1 = self.conv2d_without_pool(x,self.iconv1,self.ibn1)
            
            x = self.conv2d_plus_pool(x,self.conv2,self.cbn2)
            i2 = self.conv2d_without_pool(x,self.iconv2,self.ibn2)
            
            x = self.conv2d_plus_pool(x,self.conv3,self.cbn3)
            i3 = self.conv2d_without_pool(x,self.iconv3,self.ibn3)
            
            x = self.conv2d_plus_pool(x,self.conv4,self.cbn4)
            i4 = self.conv2d_without_pool(x,self.iconv4,self.ibn4)
            
            x = self.conv2d_plus_pool(x,self.conv5,self.cbn5)
            #deconvs
            
            
            x = self.dconv2d_plus_pool(x,self.dconv1,self.dbn1)
            x = torch.cat((x, i4),1)
            
            x = self.dconv2d_plus_pool(x,self.dconv2,self.dbn2)
            x = torch.cat((x, i3),1)
            
            x = self.dconv2d_plus_pool(x,self.dconv3,self.dbn3)
            x = torch.cat((x, i2),1)
            
            x = self.dconv2d_plus_pool(x,self.dconv4,self.dbn4)
            x = torch.cat((x, i1),1)
            
            x = self.dconv2d_plus_pool(x,self.dconv5,self.dbn5)
            
            return x
        
        def forward(self, x):
            
            x = self.convs(x)
            return x
            
    if device == None:        
        net = Net()
    else:
        net = Net().to(device)
    return net

def create_rcdnn_3conv(kernel_size,batch_size,sequence_size,hidden_dim,device=None,start_dim=3,end_dim=1):
    #kernel_size = 11 #its needs to be uneven, at least I think so
    padding = int(kernel_size/2)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(start_dim, 16, kernel_size, padding=padding)
            self.conv2 = nn.Conv2d(16, 64, kernel_size, padding=padding)
            self.conv3 = nn.Conv2d(64, 256, kernel_size, padding=padding)

            self.dconv1 = nn.ConvTranspose2d(256*2,64,kernel_size, padding=padding)
            self.dconv2 = nn.ConvTranspose2d(64*2,16,kernel_size, padding=padding)
            self.dconv3 = nn.ConvTranspose2d(16*2,end_dim,kernel_size, padding=padding)

            self.ident = 1
            self.ident_pad = 0
            self.iconv1 = nn.Conv2d(16,16,self.ident,padding=self.ident_pad)
            self.iconv2 = nn.Conv2d(64,64,self.ident,padding=self.ident_pad)

            self.cbn1 = nn.BatchNorm2d(16)
            self.cbn2 = nn.BatchNorm2d(64)
            self.cbn3 = nn.BatchNorm2d(256)

            self.dbn1 = nn.BatchNorm2d(64)
            self.dbn2 = nn.BatchNorm2d(16)
            self.dbn3 = nn.BatchNorm2d(end_dim)
            
            self.ibn1 = nn.BatchNorm2d(16)
            self.ibn2 = nn.BatchNorm2d(64)
            
            self.rnn = nn.RNN(hidden_dim, hidden_dim, sequence_size, batch_first=True)   

            
        def conv2d_plus_pool(self, in_it_goes, conv, bn, pool=(2,2)):
            out = F.relu(conv(in_it_goes))
            out = F.max_pool2d(out,pool)
            out = bn(out)
            return out
        
        def conv2d_without_pool(self, in_it_goes, conv, bn):
            out = F.relu(conv(in_it_goes))
            out = bn(out)
            return out
        
        def conv3d_plus_stack(self, in_it_goes,conv,bn): #u could add bn at this point
            out = torch.stack(in_it_goes,axis=1)
            out = conv(out)
            out = torch.squeeze(out,axis=1)
            out = bn(out)
            return out
        
        def dconv2d_plus_pool(self, in_it_goes, dconv, bn, pool=(2,2)):
            out = F.interpolate(in_it_goes,scale_factor=2)
            out = dconv(out)
            out = F.relu(out)
            out = bn(out)
            return out
        
        def init_hidden(self):
            hidden = torch.zeros(sequence_size, batch_size, hidden_dim)
            return hidden
        
        def convs(self, x):
            x = x.view([batch_size*sequence_size]+list(x.shape)[2:])
            
            #convs
            
            x = self.conv2d_plus_pool(x,self.conv1,self.cbn1)
            i1 = self.conv2d_without_pool(x,self.iconv1,self.ibn1)
            
            x = self.conv2d_plus_pool(x,self.conv2,self.cbn2)
            i2 = self.conv2d_without_pool(x,self.iconv2,self.ibn2)
            
            x = self.conv2d_plus_pool(x,self.conv3,self.cbn3)
            
            #rnn
            hidden_in = x.view([batch_size,sequence_size]+list(x.shape)[-3:])
            hidden_in = torch.flatten(hidden_in,start_dim=2)
            
            hidden = self.init_hidden()
            out, hidden = self.rnn(hidden_in, hidden)
            out = out.contiguous().view([batch_size*sequence_size]+list(x.shape)[1:])
            
            #dcons
            x = torch.cat((x, out),1)
            x = self.dconv2d_plus_pool(x,self.dconv1,self.dbn1)
            x = torch.cat((x, i2),1)
            
            x = self.dconv2d_plus_pool(x,self.dconv2,self.dbn2)
            x = torch.cat((x, i1),1)
            
            x = self.dconv2d_plus_pool(x,self.dconv3,self.dbn3)
            
            x = x.view([batch_size,sequence_size]+list(x.shape)[1:])
            return x
        
        def forward(self, x):
            
            x = self.convs(x)
            return x
            
    if device == None:        
        net = Net()
    else:
        net = Net().to(device)
    return net

def create_rcdnn_3conv_hidden(kernel_size,batch_size,sequence_size,hidden_dim,device=None,start_dim=3,end_dim=1):
    #kernel_size = 11 #its needs to be uneven, at least I think so
    padding = int(kernel_size/2)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(start_dim, 16, kernel_size, padding=padding)
            self.conv2 = nn.Conv2d(16, 32, kernel_size, padding=padding)
            self.conv3 = nn.Conv2d(32, 64, kernel_size, padding=padding)

            self.dconv1 = nn.ConvTranspose2d(64*2,64,kernel_size, padding=padding)
            self.dconv2 = nn.ConvTranspose2d(32*3,16,kernel_size, padding=padding)
            self.dconv3 = nn.ConvTranspose2d(16*2,end_dim,kernel_size, padding=padding)

            self.ident = 1
            self.ident_pad = 0
            self.iconv1 = nn.Conv2d(16,16,self.ident,padding=self.ident_pad)
            self.iconv2 = nn.Conv2d(32,32,self.ident,padding=self.ident_pad)
            
            self.fc1 = nn.Linear(16384,8192)
            self.fc2 = nn.Linear(8192,4096)
            self.fc3 = nn.Linear(4096,8192)
            self.fc4 = nn.Linear(8192,16384)
            
            self.cbn1 = nn.BatchNorm2d(16)
            self.cbn2 = nn.BatchNorm2d(32)
            self.cbn3 = nn.BatchNorm2d(64)

            self.dbn1 = nn.BatchNorm2d(64)
            self.dbn2 = nn.BatchNorm2d(16)
            self.dbn3 = nn.BatchNorm2d(end_dim)
            
            self.ibn1 = nn.BatchNorm2d(16)
            self.ibn2 = nn.BatchNorm2d(32)
            
            self.fbn1 = nn.BatchNorm1d(8192)
            self.fbn2 = nn.BatchNorm1d(4096)
            self.fbn3 = nn.BatchNorm1d(8192)
            self.fbn4 = nn.BatchNorm1d(16384)
            
            
            self.rnn = nn.RNN(hidden_dim, hidden_dim, 1, batch_first=True)#, sequence_size, batch_first=True)   

            
        def conv2d_plus_pool(self, in_it_goes, conv, bn, pool=(2,2)):
            out = F.relu(conv(in_it_goes))
            out = F.max_pool2d(out,pool)
            out = bn(out)
            return out
        
        def conv2d_without_pool(self, in_it_goes, conv, bn):
            out = F.relu(conv(in_it_goes))
            out = bn(out)
            return out
        
        def conv3d_plus_stack(self, in_it_goes,conv,bn): #u could add bn at this point
            out = torch.stack(in_it_goes,axis=1)
            out = conv(out)
            out = torch.squeeze(out,axis=1)
            out = bn(out)
            return out
        
        def dconv2d_plus_pool(self, in_it_goes, dconv, bn, pool=(2,2)):
            out = F.interpolate(in_it_goes,scale_factor=2)
            out = dconv(out)
            out = F.relu(out)
            out = bn(out)
            return out
        
        def init_hidden(self):
            hidden = torch.zeros(1, batch_size, hidden_dim)
            return hidden
        
        def convs(self, x, h):
            x = x.view([batch_size*sequence_size]+list(x.shape)[2:])
            
            #convs
            
            x = self.conv2d_plus_pool(x,self.conv1,self.cbn1)
            i1 = self.conv2d_without_pool(x,self.iconv1,self.ibn1)
            x = self.conv2d_plus_pool(x,self.conv2,self.cbn2)
            i2 = self.conv2d_without_pool(x,self.iconv2,self.ibn2)
            x = self.conv2d_plus_pool(x,self.conv3,self.cbn3)
            
            #rnn
            
            hi = torch.flatten(x,start_dim=1)
            hi = self.fbn1(F.relu(self.fc1(hi)))
            hi = self.fbn2(F.relu(self.fc2(hi)))
            
            hi = hi.view([batch_size,sequence_size,hidden_dim])
            #h = self.init_hidden()
            ho, h = self.rnn(hi, h)
            ho = ho.contiguous().view([batch_size*sequence_size,hidden_dim])
            
            ho = self.fbn3(F.relu(self.fc3(ho)))
            ho = self.fbn4(F.relu(self.fc4(ho)))
            ho = ho.view(x.shape)
            
            #dcons
            x = torch.cat((x, ho),1)
            #x = ho
            x = self.dconv2d_plus_pool(x,self.dconv1,self.dbn1)
            x = torch.cat((x, i2),1)
            x = self.dconv2d_plus_pool(x,self.dconv2,self.dbn2)
            x = torch.cat((x, i1),1)
            x = self.dconv2d_plus_pool(x,self.dconv3,self.dbn3)
            
            x = x.view([batch_size,sequence_size]+list(x.shape)[1:])
            return x, h
        
        def forward(self, x, h):
            
            x, h = self.convs(x, h)
            return x, h
            
    if device == None:        
        net = Net()
    else:
        net = Net().to(device)
    return net



def create_rcdnn_3conv_hidden_in(kernel_size,batch_size,sequence_size,hidden_dim,device=None,start_dim=3,end_dim=1):
    #kernel_size = 11 #its needs to be uneven, at least I think so
    padding = int(kernel_size/2)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(start_dim, 16, kernel_size, padding=padding)
            self.conv2 = nn.Conv2d(16, 64, kernel_size, padding=padding)
            self.conv3 = nn.Conv2d(64, 256, kernel_size, padding=padding)

            self.dconv1 = nn.ConvTranspose2d(256*2,64,kernel_size, padding=padding)
            self.dconv2 = nn.ConvTranspose2d(64*2,16,kernel_size, padding=padding)
            self.dconv3 = nn.ConvTranspose2d(16*2,end_dim,kernel_size, padding=padding)

            self.ident = 1
            self.ident_pad = 0
            self.iconv1 = nn.Conv2d(16,16,self.ident,padding=self.ident_pad)
            self.iconv2 = nn.Conv2d(64,64,self.ident,padding=self.ident_pad)

            self.cbn1 = nn.BatchNorm2d(16)
            self.cbn2 = nn.BatchNorm2d(64)
            self.cbn3 = nn.BatchNorm2d(256)

            self.dbn1 = nn.BatchNorm2d(64)
            self.dbn2 = nn.BatchNorm2d(16)
            self.dbn3 = nn.BatchNorm2d(end_dim)
            
            self.ibn1 = nn.BatchNorm2d(16)
            self.ibn2 = nn.BatchNorm2d(64)
            
            self.rnn = nn.RNN(hidden_dim, hidden_dim, sequence_size, batch_first=True)   

            
        def conv2d_plus_pool(self, in_it_goes, conv, bn, pool=(2,2)):
            out = F.relu(conv(in_it_goes))
            out = F.max_pool2d(out,pool)
            out = bn(out)
            return out
        
        def conv2d_without_pool(self, in_it_goes, conv, bn):
            out = F.relu(conv(in_it_goes))
            out = bn(out)
            return out
        
        def conv3d_plus_stack(self, in_it_goes,conv,bn): #u could add bn at this point
            out = torch.stack(in_it_goes,axis=1)
            out = conv(out)
            out = torch.squeeze(out,axis=1)
            out = bn(out)
            return out
        
        def dconv2d_plus_pool(self, in_it_goes, dconv, bn, pool=(2,2)):
            out = F.interpolate(in_it_goes,scale_factor=2)
            out = dconv(out)
            out = F.relu(out)
            out = bn(out)
            return out
        
        def init_hidden(self):
            hidden = torch.zeros(sequence_size, batch_size, hidden_dim)
            return hidden
        
        def convs(self, x, hidden_in):
            x = x.view([batch_size*sequence_size]+list(x.shape)[2:])
            
            #convs
            
            x = self.conv2d_plus_pool(x,self.conv1,self.cbn1)
            i1 = self.conv2d_without_pool(x,self.iconv1,self.ibn1)
            
            x = self.conv2d_plus_pool(x,self.conv2,self.cbn2)
            i2 = self.conv2d_without_pool(x,self.iconv2,self.ibn2)
            
            x = self.conv2d_plus_pool(x,self.conv3,self.cbn3)
            
            #rnn
            hidden_in = x.view([batch_size,sequence_size]+list(x.shape)[-3:])
            hidden_in = torch.flatten(hidden_in,start_dim=2)
            
            hidden = self.init_hidden()
            out, hidden_out = self.rnn(hidden_in, hidden)
            out = out.contiguous().view([batch_size*sequence_size]+list(x.shape)[1:])
            
            #dcons
            x = torch.cat((x, out),1)
            x = self.dconv2d_plus_pool(x,self.dconv1,self.dbn1)
            x = torch.cat((x, i2),1)
            
            x = self.dconv2d_plus_pool(x,self.dconv2,self.dbn2)
            x = torch.cat((x, i1),1)
            
            x = self.dconv2d_plus_pool(x,self.dconv3,self.dbn3)
            
            x = x.view([batch_size,sequence_size]+list(x.shape)[1:])
            return x, hidden_out
        
        def forward(self, x, hidden_in):
            
            x, hidden_out = self.convs(x, hidden_in)
            return x, hidden_out
            
    if device == None:        
        net = Net()
    else:
        net = Net().to(device)
    return net






def create_rcdnn_3conv_2(kernel_size,batch_size,sequence_size,hidden_dim,device=None,start_dim=3,end_dim=1):
    #kernel_size = 11 #its needs to be uneven, at least I think so
    padding = int(kernel_size/2)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(start_dim, 16, kernel_size, padding=padding)
            self.conv2 = nn.Conv2d(16, 32, kernel_size, padding=padding)
            self.conv3 = nn.Conv2d(32, 64, kernel_size, padding=padding)

            self.dconv1 = nn.ConvTranspose2d(64*2,64,kernel_size, padding=padding)
            self.dconv2 = nn.ConvTranspose2d(32*2,16,kernel_size, padding=padding)
            self.dconv3 = nn.ConvTranspose2d(16*1,end_dim,kernel_size, padding=padding)

            self.fc1 = nn.Linear(16384,8192)
            self.fc2 = nn.Linear(8192,4096)
            self.fc3 = nn.Linear(4096,8192)
            self.fc4 = nn.Linear(8192,16384)
            
            self.cbn1 = nn.BatchNorm2d(16)
            self.cbn2 = nn.BatchNorm2d(32)
            self.cbn3 = nn.BatchNorm2d(64)

            self.dbn1 = nn.BatchNorm2d(64)
            self.dbn2 = nn.BatchNorm2d(16)
            self.dbn3 = nn.BatchNorm2d(end_dim)
            
            self.fbn1 = nn.BatchNorm1d(8192)
            self.fbn2 = nn.BatchNorm1d(4096)
            self.fbn3 = nn.BatchNorm1d(8192)
            self.fbn4 = nn.BatchNorm1d(16384)
            
            
            self.rnn = nn.RNN(hidden_dim, hidden_dim, 1, batch_first=True)#, sequence_size, batch_first=True)   

            
        def conv2d_plus_pool(self, in_it_goes, conv, bn, pool=(2,2)):
            out = F.relu(conv(in_it_goes))
            out = F.max_pool2d(out,pool)
            out = bn(out)
            return out
        
        def conv2d_without_pool(self, in_it_goes, conv, bn):
            out = F.relu(conv(in_it_goes))
            out = bn(out)
            return out
        
        def conv3d_plus_stack(self, in_it_goes,conv,bn): #u could add bn at this point
            out = torch.stack(in_it_goes,axis=1)
            out = conv(out)
            out = torch.squeeze(out,axis=1)
            out = bn(out)
            return out
        
        def dconv2d_plus_pool(self, in_it_goes, dconv, bn, pool=(2,2)):
            out = F.interpolate(in_it_goes,scale_factor=2)
            out = dconv(out)
            out = F.relu(out)
            out = bn(out)
            return out
        
        def init_hidden(self):
            hidden = torch.zeros(1, batch_size, hidden_dim)
            return hidden
        
        def convs(self, x):
            x = x.view([batch_size*sequence_size]+list(x.shape)[2:])
            
            #convs
            
            x = self.conv2d_plus_pool(x,self.conv1,self.cbn1)
            x = self.conv2d_plus_pool(x,self.conv2,self.cbn2)
            x = self.conv2d_plus_pool(x,self.conv3,self.cbn3)
            
            #rnn
            
            hi = torch.flatten(x,start_dim=1)
            hi = self.fbn1(F.relu(self.fc1(hi)))
            hi = self.fbn2(F.relu(self.fc2(hi)))
            
            hi = hi.view([batch_size,sequence_size,hidden_dim])
            h = self.init_hidden()
            ho, h = self.rnn(hi, h)
            ho = ho.contiguous().view([batch_size*sequence_size,hidden_dim])
            
            ho = self.fbn3(F.relu(self.fc3(ho)))
            ho = self.fbn4(F.relu(self.fc4(ho)))
            ho = ho.view(x.shape)
            
            #dcons
            x = torch.cat((x, ho),1)
            #x = ho
            x = self.dconv2d_plus_pool(x,self.dconv1,self.dbn1)
            x = self.dconv2d_plus_pool(x,self.dconv2,self.dbn2)
            x = self.dconv2d_plus_pool(x,self.dconv3,self.dbn3)
            
            x = x.view([batch_size,sequence_size]+list(x.shape)[1:])
            return x
        
        def forward(self, x):
            
            x = self.convs(x)
            return x
            
    if device == None:        
        net = Net()
    else:
        net = Net().to(device)
    return net


def create_rcdnn_5conv(kernel_size,batch_size,sequence_size,hidden_dim,arcitecture = [3,16,32,64,128,256,1],device=None):
    #kernel_size = 11 #its needs to be uneven, at least I think so
    padding = int(kernel_size/2)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            pm = 'same'
            self.conv1 = nn.Conv2d(arcitecture[0], arcitecture[1], kernel_size, padding=padding, padding_mode = pm)
            self.conv2 = nn.Conv2d(arcitecture[1], arcitecture[2], kernel_size, padding=padding)
            self.conv3 = nn.Conv2d(arcitecture[2], arcitecture[3], kernel_size, padding=padding)
            self.conv4 = nn.Conv2d(arcitecture[3], arcitecture[4], kernel_size, padding=padding)
            self.conv5 = nn.Conv2d(arcitecture[4], arcitecture[5], kernel_size, padding=padding)

            self.dconv1 = nn.ConvTranspose2d(arcitecture[5]*2,arcitecture[4],kernel_size, padding=padding)
            self.dconv2 = nn.ConvTranspose2d(arcitecture[4]*2,arcitecture[3],kernel_size, padding=padding)
            self.dconv3 = nn.ConvTranspose2d(arcitecture[3]*2,arcitecture[2],kernel_size, padding=padding)
            self.dconv4 = nn.ConvTranspose2d(arcitecture[2]*2,arcitecture[1],kernel_size, padding=padding)
            self.dconv5 = nn.ConvTranspose2d(arcitecture[1]*2,arcitecture[6],kernel_size, padding=padding)

            self.ident = 1
            self.ident_pad = 0
            self.iconv1 = nn.Conv2d(arcitecture[1],arcitecture[1],self.ident,padding=self.ident_pad)
            self.iconv2 = nn.Conv2d(arcitecture[2],arcitecture[2],self.ident,padding=self.ident_pad)
            self.iconv3 = nn.Conv2d(arcitecture[3],arcitecture[3],self.ident,padding=self.ident_pad)
            self.iconv4 = nn.Conv2d(arcitecture[4],arcitecture[4],self.ident,padding=self.ident_pad)

            #self.econv1 = nn.Conv2d(arcitecture[-1], arcitecture[-1], 1)
            #self.econv2 = nn.Conv2d(arcitecture[-1], arcitecture[-1], 1)
            
            self.cbn1 = nn.BatchNorm2d(arcitecture[1])
            self.cbn2 = nn.BatchNorm2d(arcitecture[2])
            self.cbn3 = nn.BatchNorm2d(arcitecture[3])
            self.cbn4 = nn.BatchNorm2d(arcitecture[4])
            self.cbn5 = nn.BatchNorm2d(arcitecture[5])

            self.dbn1 = nn.BatchNorm2d(arcitecture[4])
            self.dbn2 = nn.BatchNorm2d(arcitecture[3])
            self.dbn3 = nn.BatchNorm2d(arcitecture[2])
            self.dbn4 = nn.BatchNorm2d(arcitecture[1])
            self.dbn5 = nn.BatchNorm2d(arcitecture[6])
            
            self.ibn1 = nn.BatchNorm2d(arcitecture[1])
            self.ibn2 = nn.BatchNorm2d(arcitecture[2])
            self.ibn3 = nn.BatchNorm2d(arcitecture[3])
            self.ibn4 = nn.BatchNorm2d(arcitecture[4])
            
            self.rnn = nn.RNN(hidden_dim, hidden_dim, sequence_size, batch_first=True) 
        
        def init_hidden(self):
            hidden = torch.zeros(sequence_size, batch_size, hidden_dim)
            return hidden
            
        def convs(self, x):
            x = x.view([batch_size*sequence_size]+list(x.shape)[2:])
            
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x,(2,2))
            x = self.cbn1(x)
            
            i1 = F.relu(self.iconv1(x))
            i1 = self.ibn1(i1)
            
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x,(2,2))
            x = self.cbn2(x)
            
            i2 = F.relu(self.iconv2(x))
            i2 = self.ibn2(i2)
            
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x,(2,2))
            x = self.cbn3(x)

            i3 = F.relu(self.iconv3(x))
            i3 = self.ibn3(i3)
            
            x = F.relu(self.conv4(x))
            x = F.max_pool2d(x,(2,2))
            x = self.cbn4(x)

            i4 = F.relu(self.iconv4(x))
            i4 = self.ibn4(i4)
            
            x = F.relu(self.conv5(x))
            x = F.max_pool2d(x,(2,2))
            x = self.cbn5(x)
            
            #rnn
            hidden_in = x.view([batch_size,sequence_size]+list(x.shape)[-3:])
            hidden_in = torch.flatten(hidden_in,start_dim=2)
            
            hidden = self.init_hidden()
            out, hidden = self.rnn(hidden_in, hidden)
            out = out.contiguous().view([batch_size*sequence_size]+list(x.shape)[1:])
            
            #deconvs
            x = torch.cat((x, out),1)
            x = F.interpolate(x,scale_factor=2)
            x = self.dconv1(x)
            x = F.relu(x)
            x = self.dbn1(x)
            x = torch.cat((x, i4),1)
            
            x = F.interpolate(x,scale_factor=2)
            x = self.dconv2(x)
            x = F.relu(x)
            x = self.dbn2(x)
            x = torch.cat((x, i3),1)
            
            x = F.interpolate(x,scale_factor=2)
            x = self.dconv3(x)
            x = F.relu(x)
            x = self.dbn3(x)
            x = torch.cat((x, i2),1)
                
            x = F.interpolate(x,scale_factor=2)
            x = self.dconv4(x)
            x = F.relu(x)
            x = self.dbn4(x)
            x = torch.cat((x, i1),1)
                
            x = F.interpolate(x,scale_factor=2)
            x = self.dconv5(x)
            x = F.relu(x)
            x = self.dbn5(x)

            x = x.view([batch_size,sequence_size]+list(x.shape)[1:])
            return x
        
        def forward(self, x):
            
            x = self.convs(x)
            return x
            
    
    if device == None:        
        net = Net()
    else:
        net = Net().to(device)
    return net

def create_rcdnn_5conv_2(kernel_size,batch_size,sequence_size,hidden_dim,arcitecture = [3,16,32,64,128,256,1],device=None):
    #kernel_size = 11 #its needs to be uneven, at least I think so
    padding = int(kernel_size/2)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            pm = 'same'
            self.conv1 = nn.Conv2d(arcitecture[0], arcitecture[1], kernel_size, padding=padding, padding_mode = pm)
            self.conv2 = nn.Conv2d(arcitecture[1], arcitecture[2], kernel_size, padding=padding, padding_mode = pm)
            self.conv3 = nn.Conv2d(arcitecture[2], arcitecture[3], kernel_size, padding=padding, padding_mode = pm)
            self.conv4 = nn.Conv2d(arcitecture[3], arcitecture[4], kernel_size, padding=padding, padding_mode = pm)
            self.conv5 = nn.Conv2d(arcitecture[4], arcitecture[5], kernel_size, padding=padding, padding_mode = pm)

            self.dconv1 = nn.ConvTranspose2d(arcitecture[5]*2,arcitecture[4],kernel_size, padding=padding)
            self.dconv2 = nn.ConvTranspose2d(arcitecture[4]*2,arcitecture[3],kernel_size, padding=padding)
            self.dconv3 = nn.ConvTranspose2d(arcitecture[3]*2,arcitecture[2],kernel_size, padding=padding)
            self.dconv4 = nn.ConvTranspose2d(arcitecture[2]*2,arcitecture[1],kernel_size, padding=padding)
            self.dconv5 = nn.ConvTranspose2d(arcitecture[1]*2,arcitecture[6],kernel_size, padding=padding)

            self.ident = 1
            self.ident_pad = 0
            self.iconv1 = nn.Conv2d(arcitecture[1],arcitecture[1],self.ident,padding=self.ident_pad)
            self.iconv2 = nn.Conv2d(arcitecture[2],arcitecture[2],self.ident,padding=self.ident_pad)
            self.iconv3 = nn.Conv2d(arcitecture[3],arcitecture[3],self.ident,padding=self.ident_pad)
            self.iconv4 = nn.Conv2d(arcitecture[4],arcitecture[4],self.ident,padding=self.ident_pad)

            self.econv1 = nn.Conv2d(arcitecture[-1], arcitecture[-1], 1)
            self.econv2 = nn.Conv2d(arcitecture[-1], arcitecture[-1], 1)
            
            self.cbn1 = nn.BatchNorm2d(arcitecture[1])
            self.cbn2 = nn.BatchNorm2d(arcitecture[2])
            self.cbn3 = nn.BatchNorm2d(arcitecture[3])
            self.cbn4 = nn.BatchNorm2d(arcitecture[4])
            self.cbn5 = nn.BatchNorm2d(arcitecture[5])

            self.dbn1 = nn.BatchNorm2d(arcitecture[4])
            self.dbn2 = nn.BatchNorm2d(arcitecture[3])
            self.dbn3 = nn.BatchNorm2d(arcitecture[2])
            self.dbn4 = nn.BatchNorm2d(arcitecture[1])
            self.dbn5 = nn.BatchNorm2d(arcitecture[6])
            
            self.ibn1 = nn.BatchNorm2d(arcitecture[1])
            self.ibn2 = nn.BatchNorm2d(arcitecture[2])
            self.ibn3 = nn.BatchNorm2d(arcitecture[3])
            self.ibn4 = nn.BatchNorm2d(arcitecture[4])
            
            self.ebn = nn.BatchNorm2d(arcitecture[-1])

            self.rnn = nn.RNN(hidden_dim, hidden_dim, sequence_size, batch_first=True) 
        
        def init_hidden(self):
            hidden = torch.zeros(sequence_size, batch_size, hidden_dim)
            return hidden
            
        def convs(self, x):
            x = x.view([batch_size*sequence_size]+list(x.shape)[2:])
            
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x,(2,2))
            x = self.cbn1(x)
            
            i1 = F.relu(self.iconv1(x))
            i1 = self.ibn1(i1)
            
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x,(2,2))
            x = self.cbn2(x)
            
            i2 = F.relu(self.iconv2(x))
            i2 = self.ibn2(i2)
            
            x = F.relu(self.conv3(x))
            x = F.max_pool2d(x,(2,2))
            x = self.cbn3(x)

            i3 = F.relu(self.iconv3(x))
            i3 = self.ibn3(i3)
            
            x = F.relu(self.conv4(x))
            x = F.max_pool2d(x,(2,2))
            x = self.cbn4(x)

            i4 = F.relu(self.iconv4(x))
            i4 = self.ibn4(i4)
            
            x = F.relu(self.conv5(x))
            x = F.max_pool2d(x,(2,2))
            x = self.cbn5(x)
            
            #rnn
            hidden_in = x.view([batch_size,sequence_size]+list(x.shape)[-3:])
            hidden_in = torch.flatten(hidden_in,start_dim=2)
            
            hidden = self.init_hidden()
            out, hidden = self.rnn(hidden_in, hidden)
            out = out.contiguous().view([batch_size*sequence_size]+list(x.shape)[1:])
            
            #deconvs
            x = torch.cat((x, out),1)
            x = F.interpolate(x,scale_factor=2)
            x = self.dconv1(x)
            x = F.relu(x)
            x = self.dbn1(x)
            x = torch.cat((x, i4),1)
            
            x = F.interpolate(x,scale_factor=2)
            x = self.dconv2(x)
            x = F.relu(x)
            x = self.dbn2(x)
            x = torch.cat((x, i3),1)
            
            x = F.interpolate(x,scale_factor=2)
            x = self.dconv3(x)
            x = F.relu(x)
            x = self.dbn3(x)
            x = torch.cat((x, i2),1)
                
            x = F.interpolate(x,scale_factor=2)
            x = self.dconv4(x)
            x = F.relu(x)
            x = self.dbn4(x)
            x = torch.cat((x, i1),1)
                
            x = F.interpolate(x,scale_factor=2)
            x = self.dconv5(x)
            x = F.relu(x)
            x = self.dbn5(x)

            x = F.relu(self.econv1(x))
            x = F.relu(self.econv2(x))
            x = self.ebn(x)

            x = x.view([batch_size,sequence_size]+list(x.shape)[1:])
            return x
        
        def forward(self, x):
            
            x = self.convs(x)
            return x
            
    
    if device == None:        
        net = Net()
    else:
        net = Net().to(device)
    return net

def create_net_classifier(kernel_size,device=None,start_shape=[128,128],start_dim=3,classes = 3):
    #kernel_size = 11 #its needs to be uneven, at least I think so
    padding = int(kernel_size/2)
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(start_dim, 16, kernel_size, padding=padding)
            self.conv2 = nn.Conv2d(16, 32, kernel_size, padding=padding)
            self.conv3 = nn.Conv2d(32, 64, kernel_size, padding=padding)
            self.conv4 = nn.Conv2d(64, 128, kernel_size, padding=padding)
            self.conv5 = nn.Conv2d(128, 256, kernel_size, padding=padding)
            
            self.cbn1 = nn.BatchNorm2d(16)
            self.cbn2 = nn.BatchNorm2d(32)
            self.cbn3 = nn.BatchNorm2d(64)
            self.cbn4 = nn.BatchNorm2d(128)
            self.cbn5 = nn.BatchNorm2d(256)

            self.fully_connected = int(start_shape[0]*start_shape[1]*256/32/32)
            self.fc1 = nn.Linear(self.fully_connected,1000)
            self.fc2 = nn.Linear(1000,100)
            self.fc3 = nn.Linear(100,classes)
        
        def conv2d_plus_pool(self, in_it_goes, conv, bn, pool=(2,2)):
            out = F.relu(conv(in_it_goes))
            out = F.max_pool2d(out,pool)
            out = bn(out)
            return out 
        
        def convs(self, x):
            x = self.conv2d_plus_pool(x,self.conv1,self.cbn1)
            x = self.conv2d_plus_pool(x,self.conv2,self.cbn2)
            x = self.conv2d_plus_pool(x,self.conv3,self.cbn3)
            x = self.conv2d_plus_pool(x,self.conv4,self.cbn4)
            x = self.conv2d_plus_pool(x,self.conv5,self.cbn5)
            x = torch.flatten(x,start_dim=1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.softmax(self.fc3(x),dim=1)#,axis=1)
            #x = self.fc3(x)#,axis=1)
            
            return x
      
        def forward(self, x):

            x = self.convs(x)
            return x


    if device == None:        
        net = Net()
    else:
        net = Net().to(device)
    return net

def create_cdnn3d_3conv(kernel_size,device=None,start_dim=1,end_dim=1):
    #kernel_size = 11 #its needs to be uneven, at least I think so
    padding = int(kernel_size/2)
    class Net(nn.Module):
        def __init__(self):
            
            super().__init__()
            
            self.conv1 = nn.Conv3d(start_dim, 16, kernel_size, padding=padding)
            self.conv2 = nn.Conv3d(16, 32, kernel_size, padding=padding)
            self.conv3 = nn.Conv3d(32, 64, kernel_size, padding=padding)

            self.dconv1 = nn.ConvTranspose3d(64,32,kernel_size, padding=padding)
            self.dconv2 = nn.ConvTranspose3d(32*2,16,kernel_size, padding=padding)
            self.dconv3 = nn.ConvTranspose3d(16*2,end_dim,kernel_size, padding=padding)

            self.iconv1 = nn.Conv3d(16,16,1)
            self.iconv2 = nn.Conv3d(32,32,1)
            
            self.cbn1 = nn.BatchNorm3d(16)
            self.cbn2 = nn.BatchNorm3d(32)
            self.cbn3 = nn.BatchNorm3d(64)

            self.dbn1 = nn.BatchNorm3d(32)
            self.dbn2 = nn.BatchNorm3d(16)
            self.dbn3 = nn.BatchNorm3d(end_dim)
            
            self.ibn1 = nn.BatchNorm3d(16)
            self.ibn2 = nn.BatchNorm3d(32)

            
        def conv3d_plus_pool(self, in_it_goes, conv, bn, pool=(1,2,2)):
            out = F.relu(conv(in_it_goes))
            out = F.max_pool3d(out,pool)
            out = bn(out)
            return out
        
        def conv3d_without_pool(self, in_it_goes, conv, bn):
            out = F.relu(conv(in_it_goes))
            out = bn(out)
            return out
        
        def dconv3d_plus_up(self, in_it_goes, dconv, bn, up=(1,2,2)):
            out = F.interpolate(in_it_goes,scale_factor=up)
            out = dconv(out)
            out = F.relu(out)
            out = bn(out)
            return out
        
        def convs(self, x):
            #convs
            
            x = self.conv3d_plus_pool(x,self.conv1,self.cbn1)
            i1 = self.conv3d_without_pool(x,self.iconv1,self.ibn1)
            
            x = self.conv3d_plus_pool(x,self.conv2,self.cbn2)
            i2 = self.conv3d_without_pool(x,self.iconv2,self.ibn2)
            
            x = self.conv3d_plus_pool(x,self.conv3,self.cbn3)
            
            #dcons
            x = self.dconv3d_plus_up(x,self.dconv1,self.dbn1)
            x = torch.cat((x, i2),1)
            
            x = self.dconv3d_plus_up(x,self.dconv2,self.dbn2)
            x = torch.cat((x, i1),1)
            
            x = self.dconv3d_plus_up(x,self.dconv3,self.dbn3)
            
            return x
        
        def forward(self, x):
            
            x = self.convs(x)
            return x
            
    if device == None:        
        net = Net()
    else:
        net = Net().to(device)
    return net

def onExit():
    print('its getting dark')
    cv2.destroyAllWindows()
    sys.exit()


def pytorch_normalize(img):
    img = img-np.min(img)
    if np.max(img) != 0:
        img = img/np.max(img)#.astype(np.uint8)
    return img
  
def pytorch_normalize_2(img):
    img = img-np.mean(img)
    img = np.clip(img,0,1)
    if np.max(img) != 0:
        img = img/np.max(img)#.astype(np.uint8)
    return img

def pytorch_normalize_3(img,p=0.5):
    img = np.clip(img-p,0,1)
    mask = img > 0
    if np.max(img) != 0:
        img = img/np.max(img)*(1-p)
    img += p
    img = img * mask
    return img

def use_net(net,data,device=None):
  if device == None:
    with torch.no_grad():
      out = net(torch.Tensor(data))
    out = out.data.numpy()
  else:
    with torch.no_grad():
      out = net(torch.Tensor(data).to(device))
    out = out.cpu().data.numpy()
  return out

def classify(net,img,dim,device=None,round_after=3):
    tmp = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    tmp = dm.opencv_to_pytorch(np.array([tmp]))
    out = use_net(net,tmp,device)[0]
    out = np.around(out,decimals=round_after)
    return out

def evaluate(net,img,normalize = pytorch_normalize,device=None):
  tmp = dm.opencv_to_pytorch(np.array([img]))
  out = use_net(net,tmp,device)
  out = normalize(out)
  out = dm.pytorch_to_opencv(out)[0]
  return out

def evaluate_plus_resizing(net,img,dim,normalize = pytorch_normalize,device=None):
  shape = (img.shape[1],img.shape[0])
  '''
  if len(img.shape) == 3 and img.shape[2] == 6:
    tmp1 = img[:,:,0:3]
    tmp2 = img[:,:,3:6]
    tmp1 = cv2.resize(tmp1, dim, interpolation = cv2.INTER_AREA)
    tmp2 = cv2.resize(tmp2, dim, interpolation = cv2.INTER_AREA)
    tmp = np.dstack((tmp1,tmp2))
  else:
    tmp = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  '''
  tmp = resize_to_avoid_bug(img,dim,cv2.INTER_AREA)

  out = evaluate(net,tmp,normalize=normalize,device=device)

  out = cv2.resize(out, shape, interpolation = cv2.INTER_AREA)
  return out

def resize_to_avoid_bug(img,dim,interpolation):
    if len(img.shape)>2:
        layers = list()
        for i in range(img.shape[2]):
            layers.append(cv2.resize(img[:,:,i], dim, interpolation = interpolation))
        return np.dstack(layers)
    return cv2.resize(img, dim, interpolation = interpolation)