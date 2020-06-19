# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:24:41 2020

@author: varis
"""

import numpy as np


def lin_f(pred,weights,var=1E-4):
    if pred.ndim == 1:
        return np.dot(pred,weights)+np.sqrt(var)*np.random.randn()
    else:
        return np.dot(pred,weights)+np.sqrt(var)*np.random.randn(pred.shape[0])

def rms_loss(weights,pred,data):
    '''
  
    Parameters
    ----------
    weights : size dim, float of  current linear model weights
    pred : dim x size data, float of current predictors per input point
    data : value of data at uncertain regression points

    Returns
    -------
    RMS : RMS loss function

    '''
   
    RMS = np.sum(lin_f(pred,weights)-data)**2
    return RMS

train = 5
dim = 10
true_pts = np.random.normal(scale = 4.0, size = (train,dim))

data = np.sum(true_pts,axis = 1)
print('true points',true_pts)
print('true data',data)
print('noisy map',lin_f(true_pts,np.ones(dim)))

print('RMS error',rms_loss(np.ones(dim),true_pts,data))

noisy_pred = true_pts + np.random.normal(scale = 1E-2, size = true_pts.shape)

init_weights = np.random.normal(size=dim)

print('initial weights',init_weights)


    