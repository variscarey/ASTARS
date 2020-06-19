# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:24:41 2020

@author: varis
"""

import numpy as np
from astars.stars_sim import Stars_sim


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
print('initial RMS loss',rms_loss(init_weights,noisy_pred,data))

def stars_wrapper(iterate,dim=10):
    weights = iterate[0:10]
    #print(weights)
    pred = iterate[10:].reshape(-1,dim)
    print('initial first component',noisy_pred[:,0])
    print('current first component',pred[:,0])
    return rms_loss(weights,pred,data)

#stars setup
maxit = 500
init_pt = np.hstack((init_weights,noisy_pred.flatten()))    
test = Stars_sim(stars_wrapper, init_pt, L1 = 400.0, var = 1E-4, verbose = True, maxit = maxit)
test.STARS_only = True
test.debug = False
test.get_mu_star()
test.get_h()
# do 100 steps
while test.iter < test.maxit:
   test.step()

import matplotlib.pyplot as plt

plt.semilogy(test.fhist)


    