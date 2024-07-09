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

mm1=0; mm2=16  #0; 16
data_load_main=np.zeros([len(h0[ind7]),4+mm2-mm1])
data_load_main[:,0]=l0[ind7]
data_load_main[:,1]=b00[ind7]
data_load_main[:,2]=ustar0[ind7]
data_load_main[:,3]=h0[ind7]
data_load_main[:,4:(mm2-mm1+4)]=SF0[ind7,mm1:mm2]

data_load3=copy.deepcopy(data_load_main)
print("data loaded")

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
        log_gsigma[j,:] = np.log(log_gsigma[j,:]/np.max(log_gsigma[j,:]))

    log_gsigma_mean=np.mean(log_gsigma,axis=0)
    log_gsigma_std= np.std(log_gsigma,axis=0)

    k_points=16 #np.shape(data[:,4:])[1]

    # Demean logsigma and normalize it with standard deviation
    for k in range(k_points):
        log_gsigma[:,k]=(log_gsigma[:,k]-log_gsigma_mean[k])/log_gsigma_std[k]

    tr_y=log_gsigma
    
    return tr_x,tr_y, stats, log_gsigma_mean, log_gsigma_std

tr_x,tr_y, stats, log_gsigma_mean, log_gsigma_std=preprocess_train_data(data_load3)
print("data preprocessed")

# Randomly select 10% of data for hyper-parameter tuning
ind_tune = np.arange(0,tr_x.shape[0],1)
np.random.shuffle(ind_tune) 
num_selected = round(len(ind_tune)*0.1)
tr_x_tune = tr_x[ind_tune,:][:num_selected,:]
tr_y_tune = tr_y[ind_tune,:][:num_selected,:]

best_param_dict = dict()
scores_list = []

for i in range(16):
    torch.cuda.empty_cache()
    gc.collect()

    rfr = RFR(random_state = 66)
    param_grid = {"n_estimators": list(range(10,160,10)), 
                "max_depth": list(range(1,11)),
                "max_features": list(range(1,5))}
    GSCV = GridSearchCV(rfr,param_grid,scoring = "neg_mean_squared_error")
    GSCV.fit(tr_x_tune,tr_y_tune[:,i])
    best_param_dict[f"{i+1} column"] = GSCV.best_params_
    scores_list.append(-GSCV.best_score_)
    
    del rfr,GSCV

print(f"The best parameters are {best_param_dict}")
print(f"The list of scores (mean squared error) are {scores_list}")
print(f"The average performance is {np.mean(scores_list)}")

# MAE as metric
# The best parameters are
                        #  {'1 column': {'max_depth': 5, 'max_features': 2, 'n_estimators': 50}, 
                        #   '2 column': {'max_depth': 5, 'max_features': 1, 'n_estimators': 80}, 
                        #   '3 column': {'max_depth': 5, 'max_features': 1, 'n_estimators': 140}, 
                        #   '4 column': {'max_depth': 5, 'max_features': 1, 'n_estimators': 20}, 
                        #   '5 column': {'max_depth': 8, 'max_features': 2, 'n_estimators': 30}, 
                        #   '6 column': {'max_depth': 6, 'max_features': 3, 'n_estimators': 90}, 
                        #   '7 column': {'max_depth': 5, 'max_features': 2, 'n_estimators': 90}, 
                        #   '8 column': {'max_depth': 5, 'max_features': 1, 'n_estimators': 90}, 
                        #   '9 column': {'max_depth': 5, 'max_features': 1, 'n_estimators': 90}, 
                        #   '10 column': {'max_depth': 5, 'max_features': 1, 'n_estimators': 10}, 
                        #   '11 column': {'max_depth': 5, 'max_features': 1, 'n_estimators': 10}, 
                        #   '12 column': {'max_depth': 5, 'max_features': 1, 'n_estimators': 20}, 
                        #   '13 column': {'max_depth': 5, 'max_features': 1, 'n_estimators': 30}, 
                        #   '14 column': {'max_depth': 5, 'max_features': 1, 'n_estimators': 120}, 
                        #   '15 column': {'max_depth': 5, 'max_features': 1, 'n_estimators': 120}, 
                        #   '16 column': {'max_depth': 5, 'max_features': 1, 'n_estimators': 140}}
# The list of scores (mean absolute error) are
                    # [0.8261802564920877, 0.7558332389563769, 0.641316259121996, 0.6665849246506932,
                    # 0.900231269164701, 0.8273990549184675, 0.744042427685627, 0.6937356107115706, 
                    # 0.6758924643351577, 0.6256811290788289, 0.5977666525426375, 0.7530039657558006, 
                    # 0.7810863614024693, 0.7597234201281696, 0.7315923597523083, 0.6940271663365525]

# MSE as metric
# The best parameters are 
                        # {'1 column': {'max_depth': 1, 'max_features': 3, 'n_estimators': 130}, 
                        # '2 column': {'max_depth': 2, 'max_features': 1, 'n_estimators': 100}, 
                        # '3 column': {'max_depth': 1, 'max_features': 1, 'n_estimators': 90}, 
                        # '4 column': {'max_depth': 1, 'max_features': 4, 'n_estimators': 150}, 
                        # '5 column': {'max_depth': 1, 'max_features': 1, 'n_estimators': 60}, 
                        # '6 column': {'max_depth': 1, 'max_features': 1, 'n_estimators': 90}, 
                        # '7 column': {'max_depth': 1, 'max_features': 1, 'n_estimators': 70}, 
                        # '8 column': {'max_depth': 1, 'max_features': 1, 'n_estimators': 110}, 
                        # '9 column': {'max_depth': 2, 'max_features': 1, 'n_estimators': 10}, 
                        # '10 column': {'max_depth': 1, 'max_features': 1, 'n_estimators': 10}, 
                        # '11 column': {'max_depth': 1, 'max_features': 1, 'n_estimators': 40}, 
                        # '12 column': {'max_depth': 1, 'max_features': 1, 'n_estimators': 20}, 
                        # '13 column': {'max_depth': 1, 'max_features': 1, 'n_estimators': 70}, 
                        # '14 column': {'max_depth': 1, 'max_features': 1, 'n_estimators': 60}, 
                        # '15 column': {'max_depth': 1, 'max_features': 1, 'n_estimators': 150},
                        # '16 column': {'max_depth': 1, 'max_features': 1, 'n_estimators': 150}}
# The list of scores (mean squared error) are 
                        # [1.004898988970417, 1.005664965347105, 1.0078069400648908, 1.0107348054347667, 
                        #  1.0030113798044062, 0.9991510372487049, 0.9975427637758436, 0.9925515306370241,
                        #  0.9894305826256865, 0.9862748933787826, 0.983556185376802, 0.9963159572151905, 
                        #  1.0033396774325605, 1.0006888161916057, 0.9988668042053632, 0.9988153059455639]
