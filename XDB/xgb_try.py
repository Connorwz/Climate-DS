# -*- coding: utf-8 -*-
"""XGB_TRY.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1vwUPCjBe6y0VLO8mSp5jUezxArGuE5_D
"""

!pip install netCDF4
!pip install xgboost
#!pip install dask distributed dask-ml dask-xgboost

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
#from torch_lr_finder import LRFinder
import copy as copy
import multiprocessing as mp
from datetime import datetime
today = datetime.today()
# custom modules
np.random.seed(100)
import func_file as ff
from scipy import stats
import time as time
  # setting random seed
def corio(lat):
    return  2*(2*np.pi/(24*60*60)) * np.sin(lat*(np.pi/180))
import matplotlib.font_manager
#torch.set_default_tensor_type(torch.DoubleTensor)  # sets float 64 as default

torch.cuda.empty_cache()

#cwd=os.getcwd()
#cwd1=os.path.dirname(os.path.dirname(cwd))+'/Data/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import xgboost as xgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV

from google.colab import drive
drive.mount('/content/drive')

folder_path = '/content/drive/My Drive/Colab_Notebooks/code_and_data'
os.chdir(folder_path)
cwd=os.getcwd()
cwd

# Load and preprocess data (same as what paper did)

def load_and_preprocess_data():
    d = ncd.Dataset(cwd + '/Data/training_data_for_SF_hbl_gaps_filled.nc').variables

    l0 = corio(d['l'][:])
    b00 = d['b0'][:]
    ustar0 = d['ustar'][:]
    h0 = d['h'][:]
    lat0 = d['lat'][:]
    heat0 = d['heat'][:]
    tx0 = d['tx'][:]
    tx0 = np.round(tx0, 2)
    SF0 = d['SF'][:]

    ind101 = np.where(np.abs(heat0) < 601)[0]
    ind1 = ind101
    ind2 = np.where(tx0 < 1.2)[0]
    ind3 = np.where(h0 > 29)[0]
    ind4 = np.where(h0 < 301)[0]

    ind5 = np.intersect1d(ind1, ind2)
    ind6 = np.intersect1d(ind3, ind5)
    ind7 = np.intersect1d(ind4, ind6)
    mm1 = 0; mm2 = 16  # 0; 16
    data_load_main = np.zeros([len(h0[ind7]), 4 + mm2 - mm1])
    data_load_main[:, 0] = l0[ind7]
    data_load_main[:, 1] = b00[ind7]
    data_load_main[:, 2] = ustar0[ind7]
    data_load_main[:, 3] = h0[ind7]
    data_load_main[:, 4:(mm2 - mm1 + 4)] = SF0[ind7, mm1:mm2]

    data_forc = np.zeros([len(ind7), 3])
    data_forc[:, 0] = lat0[ind7]
    data_forc[:, 1] = heat0[ind7]
    data_forc[:, 2] = tx0[ind7]

    data_load3 = copy.deepcopy(data_load_main)

    print('started')

    data, x, y, stats, k_mean, k_std = ff.preprocess_train_data(data_load3, _, _, _)

    valid_data = np.loadtxt(cwd + '/Data/data_testing_4_paper.txt')[:, 3:]
    ind3 = np.where(valid_data[:, 3] > 29)[0]
    ind4 = np.where(valid_data[:, 3] < 301)[0]
    ind = np.intersect1d(ind3, ind4)

    valid_x = valid_data[ind, 0:4]

    valid_x[:, 0] = (valid_x[:, 0] - stats[0]) / stats[1]
    valid_x[:, 1] = (valid_x[:, 1] - stats[2]) / stats[3]
    valid_x[:, 2] = (valid_x[:, 2] - stats[4]) / stats[5]
    valid_x[:, 3] = (valid_x[:, 3] - stats[6]) / stats[7]
    k_mean_test = np.zeros(16)
    valid_y = valid_data[ind, 5:]

    for i in range(len(valid_y)):
        valid_y[i, :] = np.log(valid_y[i, :] / np.max(valid_y[i, :]))

    for i in range(16):
        valid_y[:, i] = (valid_y[:, i] - k_mean[i]) / k_std[i]

    return x, y, valid_x, valid_y, k_mean, k_std

def xgboost_train(x, y, valid_x, valid_y, lr, max_depth, epochs, k_std, k_mean):
    dtrain = xgb.DMatrix(x, label=y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    params = {
        'objective': 'reg:absoluteerror',
        'learning_rate': lr,
        'max_depth': max_depth,
        'eval_metric': 'mae',
        #'device': 'cuda'
    }

    evals = [(dtrain, 'train'), (dvalid, 'eval')]
    model = xgb.train(params, dtrain, num_boost_round=epochs, evals=evals, early_stopping_rounds=30)

    # Predict and calculate losses for each epoch
    loss_array = np.zeros((epochs, 3))
    for k in range(epochs):
        y_pred = model.predict(dtrain, iteration_range=(0, k+1))
        valid_pred = model.predict(dvalid, iteration_range=(0, k+1))

        loss_train = np.mean(np.abs(np.exp(y_pred * k_std + k_mean) - np.exp(y * k_std + k_mean)))
        loss_valid = np.mean(np.abs(np.exp(valid_pred * k_std + k_mean) - np.exp(valid_y * k_std + k_mean)))

        loss_array[k, 0] = k
        loss_array[k, 1] = loss_train
        loss_array[k, 2] = loss_valid

    return model, loss_array

if __name__ == "__main__":
    x, y, valid_x, valid_y, k_mean, k_std = load_and_preprocess_data()

    # Define the parameter grid
    learning_rates = [0.01, 0.05, 0.1]
    max_depths = [2, 3, 5]
    epochs = 3000

    best_mae = float("inf")
    best_params = None
    best_model = None
    best_loss_array = None

    for lr in learning_rates:
        for max_depth in max_depths:
            print(f"Training with learning_rate={lr}, max_depth={max_depth}")
            model, loss_array = xgboost_train(x, y, valid_x, valid_y, lr, max_depth, epochs, k_std, k_mean)

            # Evaluate the model
            y_pred = model.predict(xgb.DMatrix(valid_x))
            mae = mean_absolute_error(np.exp(y_pred * k_std + k_mean), np.exp(valid_y * k_std + k_mean))
            print(f"MAE for learning_rate={lr}, max_depth={max_depth}: {mae}")

            # Check if this is the best model
            if mae < best_mae:
                best_mae = mae
                best_params = {'learning_rate': lr, 'max_depth': max_depth}
                best_model = model
                best_loss_array = loss_array

    print(f"Best parameters found: {best_params}")
    print(f"Best MAE: {best_mae}")

    # Use best parameters to train the final model using the XGB function we defined
    best_lr = best_params['learning_rate']
    best_max_depth = best_params['max_depth']

    model, loss_array = xgboost_train(x, y, valid_x, valid_y, best_lr, best_max_depth, epochs, k_std, k_mean)

    # Evaluate on validation data
    valid_pred = model.predict(xgb.DMatrix(valid_x))
    valid_pred_original = np.exp(valid_pred * k_std + k_mean)
    valid_y_original = np.exp(valid_y * k_std + k_mean)

    # Plotting loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(loss_array[:, 0], loss_array[:, 1], label='Training Loss')
    plt.plot(loss_array[:, 0], loss_array[:, 2], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()

# Plot g(sigma) for a specific case
valid_pred = model.predict(xgb.DMatrix(valid_x))
valid_pred_original = np.exp(valid_pred * k_std + k_mean)
valid_y_original = np.exp(valid_y * k_std + k_mean)

sigma_values = np.linspace(0, 1, 16)  # sigma values from 0 to 1

plt.figure(figsize=(10, 6))
example_index = 1000  # Change this index to plot different example
plt.plot(sigma_values, valid_pred_original[example_index, :], label=f'Predicted {example_index+1}', alpha=0.7)
plt.plot(sigma_values, valid_y_original[example_index, :], label=f'Actual {example_index+1}', linestyle='dashed', alpha=0.7)

plt.xticks(sigma_values, [f'{i+1}/17' for i in range(16)])
plt.xlabel('Sigma')
plt.ylabel('g(sigma)')
plt.title('Shape Function g(sigma)')
plt.legend()
plt.show()









