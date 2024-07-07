import os
import numpy as np
import matplotlib.pyplot as plt
import copy as copy
import matplotlib as mpl
import netCDF4 as ncd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as Data
#import torchvision
from torch import nn, optim
import matplotlib.cm as cm

import copy as copy

np.random.seed(10)
#torch.set_default_tensor_type(torch.DoubleTensor)  # sets float 64 as default

class learnKappa_layers1(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):

        super(learnKappa_layers1, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid) 
        self.linear2 = nn.Linear(Hid, Out_nodes)
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):

            x2=self.linear1(x)
            h1 = torch.relu(x2)
            h1=self.dropout(h1)
            y_pred = self.linear2(h1)
            return y_pred

    
class learnKappa_layers2(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):

        super(learnKappa_layers2, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid) 
        self.linear2 = nn.Linear(Hid, Hid) 
        self.linear3 = nn.Linear(Hid, Out_nodes)
        
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):

            x2 = self.linear1(x)
            h1 = torch.relu(x2)
    
            h1=self.dropout(h1) # dropout for hid layer 1 
            
            h2 = self.linear2(h1)
            h3 = torch.relu(h2)
   
            h3=self.dropout(h3) # dropout for hid layer 1 
            y_pred = self.linear3(h3)       
            return y_pred
        
class learnKappa_layers3(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):

        super(learnKappa_layers3, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid) 
        self.linear2 = nn.Linear(Hid, Hid) 
        self.linear3 = nn.Linear(Hid, Hid) 
        self.linear4 = nn.Linear(Hid, Out_nodes)
        
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):

            x2 = self.linear1(x)
            h1 = torch.relu(x2)
            h1=self.dropout(h1) # dropout for hid layer 1 
            
            h2 = self.linear2(h1)
            h3 = torch.relu(h2)
            h2=self.dropout(h2)
            
            h4 = self.linear3(h3)
            h5 = torch.relu(h4)
            h5=self.dropout(h5)
            
            y_pred = self.linear4(h5)       
            return y_pred

class learnKappa_layers4(nn.Module):
    def __init__(self, In_nodes, Hid, Out_nodes):

        super(learnKappa_layers4, self).__init__()
        self.linear1 = nn.Linear(In_nodes, Hid) 
        self.linear2 = nn.Linear(Hid, Hid) 
        self.linear3 = nn.Linear(Hid, Hid) 
        self.linear4 = nn.Linear(Hid, Hid)
        self.linear5 = nn.Linear(Hid, Out_nodes)
        
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):

            x2 = self.linear1(x)
            h1 = torch.relu(x2)
            h1=self.dropout(h1) # dropout for hid layer 1 
            
            h2 = self.linear2(h1)
            h3 = torch.relu(h2)
            h2=self.dropout(h2)
            
            h4 = self.linear3(h3)
            h5 = torch.relu(h4)
            h5=self.dropout(h5)
            
            h6 = self.linear4(h5)
            h7 = torch.relu(h6)
            h7=self.dropout(h7)
            
            
            y_pred = self.linear5(h7)       
            return y_pred


# pre-processing data

def preprocess_train_data(data_load,trainpts,log_y_n,kpoints):


    ind=np.arange(0,len(data_load),1)    # creates a list of indices to shuffle
    ind_shuffle=copy.deepcopy(ind)  # deep copies the indices
    np.random.shuffle(ind_shuffle)  # shuffles the array
    
    l_mean=np.mean(data_load[:,0]); 
    l_std=np.std(data_load[:,0]); 
    data_load[:,0]=(data_load[:,0]-l_mean)/l_std    #l
    
    h_mean=np.mean(data_load[:,1]); 
    h_std=np.std(data_load[:,1]); 
    data_load[:,1]=(data_load[:,1]-h_mean)/h_std    #b0
    
    t_mean=np.mean(data_load[:,2]); 
    t_std=np.std(data_load[:,2]); 
    data_load[:,2]=(data_load[:,2]-t_mean)/t_std    #u*
    
    hb_mean= np.mean(data_load[:,3]); 
    hb_std=np.std(data_load[:,3]); 
    data_load[:,3]=(data_load[:,3]-hb_mean)/(hb_std)  #w*

        
    data=data_load
        
        
    stats=np.array([l_mean, l_std, h_mean, h_std, t_mean, t_std, hb_mean, hb_std])
 
    for j in range(len(data[:,0])):
        data[j,4:]=np.log(data[j,4:]/np.max(data[j,4:]))

    k_mean=np.mean(data[:,4:],axis=0)
    k_std= np.std(data[:,4:],axis=0)

    k_points=16 #np.shape(data[:,4:])[1]

    for k in range(k_points):
        data[:,k+4]=(data[:,k+4]-k_mean[k])/k_std[k]
        
    x=data[ind_shuffle,0:4]
    y=data[ind_shuffle,4:]
    
    return data, x,y, stats, k_mean, k_std

def preprocessdata(data_load,trainpts,log_y_n, kpoints):
    # function process the data
    # l b u w hbl
    # 0 1 2 3 4

    ind=np.arange(0,len(data_load),1)    # creates a list of indices to shuffle
    ind_shuffle=copy.deepcopy(ind)  # deep copies the indices
    np.random.shuffle(ind_shuffle)  # shuffles the array
    
    l_mean=np.mean(data_load[:,0]); 
    l_std=np.std(data_load[:,0]); 
    data_load[:,0]=(data_load[:,0]-l_mean)/l_std    #l
    
    h_mean=np.mean(data_load[:,1]); 
    h_std=np.std(data_load[:,1]); 
    data_load[:,1]=(data_load[:,1]-h_mean)/h_std    #b0
    
    t_mean=np.mean(data_load[:,2]); 
    t_std=np.std(data_load[:,2]); 
    data_load[:,2]=(data_load[:,2]-t_mean)/t_std    #u*
    
    hb_mean= np.mean(data_load[:,3]); 
    hb_std=np.std(data_load[:,3]); 
    data_load[:,3]=(data_load[:,3]-hb_mean)/(hb_std)  #w*

    if trainpts > len(data_load):
        print('Error, training points exceeds availability')
    else:
        data=data_load[ind_shuffle[0:trainpts],:]
        data_test=data_load[ind_shuffle[trainpts:],:]
        
        
    stats=np.array([l_mean, l_std, h_mean, h_std, t_mean, t_std, hb_mean, hb_std])#, u_mean, u_std])

    for j in range(len(data[:,0])):
        if log_y_n==True:
            data[j,4:]=np.log(data[j,4:]/np.max(data[j,4:]))
        elif log_y_n==False:
            data[j,4:]=data[j,4:]/np.max(data[j,4:])
    for j in range(len(data_test[:,0])):
        if log_y_n==True:
            data_test[j,4:]=np.log(data_test[j,4:]/np.max(data_test[j,4:]))
        elif log_y_n==False:
            data_test[j,4:]=data_test[j,4:]/np.max(data_test[j,4:])
            

    k_mean=np.mean(data[:,4:],axis=0)
    k_std= np.std(data[:,4:],axis=0)

    k_points=np.shape(data[:,4:])[1]

    for k in range(k_points):
        data[:,k+4]=(data[:,k+4]-k_mean[k])/k_std[k]
        data_test[:,k+4]=(data_test[:,k+4]-k_mean[k])/k_std[k]

    x=data[:,0:4]
    y=data[:,4:]

    test_x=data_test[:,0:4]
    test_y=data_test[:,4:]
    
    return data, data_test, x,y, test_x, test_y, ind_shuffle, stats, k_mean, k_std

def modeltrain(In_nodes, Hid, Out_nodes, lr, epochs, x, y, test_x, test_y,model):

    optimizer = torch.optim.Adam(model.parameters(), lr)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    loss_array=torch.zeros([epochs,3])

    for k in range(epochs):
        optimizer.zero_grad()  
        y_pred=model(x)
        
        loss=loss_fn(y_pred,y)

        loss_valid=loss_fn(model(test_x),test_y)
        loss.backward()
        optimizer.step()

        t = torch.cuda.get_device_properties(0).total_memory/(1024*1024*1024)
        r = torch.cuda.memory_reserved(0)/(1024*1024*1024)
        a = torch.cuda.memory_allocated(0)/(1024*1024*1024)
        f = (r-a)

        loss_array[k,0]=k
        loss_array[k,1]=loss.item()
        loss_array[k,2]=loss_valid.item()

        del loss, y_pred

    return model, loss_array
