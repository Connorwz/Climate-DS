import numpy as np
import copy as copy
import netCDF4 as ncd
def corio(lat):
    return  2*(2*np.pi/(24*60*60)) * np.sin(lat*(np.pi/180))
import cuml
from cuml.ensemble import RandomForestRegressor as RFR
from sklearn.model_selection import GridSearchCV
import gc
import torch
np.random.seed(100)
print("Libs are read")

d=ncd.Dataset("/user/wx2309/code_and_data/Data/training_data_for_SF_hbl_gaps_filled.nc").variables

l0=corio(d['l'][:])
b00=d['b0'][:]
ustar0=d['ustar'][:]
h0=d['h'][:]
lat0=d['lat'][:]
heat0=d['heat'][:]
tx0=d['tx'][:] 
tx0=np.round(tx0,2)
SF0=d['SF'][:] 

ind1=np.where(np.abs(heat0)<601)[0]
ind2=np.where(tx0<1.2)[0]
ind3=np.where(h0>29)[0]
ind4=np.where(h0<301)[0]

ind5=np.intersect1d(ind1,ind2)
ind6=np.intersect1d(ind3,ind5)
ind7=np.intersect1d(ind4,ind6)

def preprocess_train_data(data_load):


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

    stats=np.array([l_mean, l_std, h_mean, h_std, t_mean, t_std, hb_mean, hb_std])
    tr_x=data_load[ind_shuffle,0:4]

    log_gsigma = data_load[:,4:]

    # Take the logarithm of normalized Kappa(sigma)
    for j in range(log_gsigma.shape[0]):
        log_gsigma[j,4:] = np.log(log_gsigma[j,4:]/np.max(log_gsigma[j,4:]))

    log_gsigma_mean=np.mean(log_gsigma,axis=0)
    log_gsigma_std= np.std(log_gsigma,axis=0)

    k_points=16 #np.shape(data[:,4:])[1]

    # Demean logsigma and normalize it with standard deviation
    for k in range(k_points):
        log_gsigma[:,k]=(log_gsigma[:,k]-log_gsigma_mean[k])/log_gsigma_std[k]

    tr_y=log_gsigma
    
    return tr_x,tr_y, stats, log_gsigma_mean, log_gsigma_std

mm1=0; mm2=16  #0; 16
data_load_main=np.zeros([len(h0[ind7]),4+mm2-mm1])
data_load_main[:,0]=l0[ind7]
data_load_main[:,1]=b00[ind7]
data_load_main[:,2]=ustar0[ind7]
data_load_main[:,3]=h0[ind7]
data_load_main[:,4:(mm2-mm1+4)]=SF0[ind7,mm1:mm2]

data_load3=copy.deepcopy(data_load_main)

tr_x,tr_y, stats, log_gsigma_mean, log_gsigma_std=preprocess_train_data(data_load3)

valid_data=np.loadtxt('/user/wx2309/code_and_data/Data/data_testing_4_paper.txt')[:,3:]

ind3=np.where(valid_data[:,3]>29)[0]
ind4=np.where(valid_data[:,3]<301)[0]
ind=np.intersect1d(ind3,ind4)

valid_x=valid_data[ind,0:4]

valid_x[:,0]=(valid_x[:,0]-stats[0])/stats[1]
valid_x[:,1]=(valid_x[:,1]-stats[2])/stats[3]
valid_x[:,2]=(valid_x[:,2]-stats[4])/stats[5]
valid_x[:,3]=(valid_x[:,3]-stats[6])/stats[7]
valid_y=valid_data[ind,5:]

for i in range(len(valid_y)):
    valid_y[i,:]=np.log(valid_y[i,:]/np.max(valid_y[i,:]))

for i in range(16):
    valid_y[:,i]=(valid_y[:,i]-log_gsigma_mean[i])/log_gsigma_mean[i]

# Randomly select 10% of data for hyper-parameter tuning
ind_tune = np.arange(0,tr_x.shape[0],1)
np.random.shuffle(ind_tune) 
num_selected = round(len(ind_tune)*0.1)
tr_x_tune = tr_x[:num_selected,:]
tr_y_tune = tr_y[:num_selected,:]

best_param_dict = dict()
scores_list = []

for i in range(16):
    torch.cuda.empty_cache()
    gc.collect()

    rfr = RFR()
    param_grid = {"n_estimators": list(range(10,110,10)), 
                "max_depth": list(range(10,20)),
                "max_features": list(range(1,5))}
    GSCV = GridSearchCV(rfr,param_grid,scoring = "neg_mean_absolute_error")
    GSCV.fit(tr_x_tune,tr_y_tune[:,i])
    best_param_dict[f"{i+1} column"] = GSCV.best_params_
    scores_list.append(-GSCV.best_score_)
    
    del rfr,GSCV

print(f"The best parameters are {best_param_dict}")
print(f"The list of scores (mean absolute error) are {scores_list}")
print(f"The average performance is {np.mean(scores_list)}")