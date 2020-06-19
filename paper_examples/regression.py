# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:24:41 2020

@author: varis
"""

import numpy as np
from astars.stars_sim import Stars_sim
import matplotlib.pyplot as plt


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

dim = 10
train = 5
true_pts = np.random.normal(scale = 4.0, size = (train,dim))

#data = np.sum(true_pts,axis = 1)
data = true_pts[:,0]
print('true points',true_pts)
print('true data',data)
map = np.zeros(dim)
map[0] = 1
print('noisy map',lin_f(true_pts,map))

print('RMS error',rms_loss(np.ones(dim),true_pts,data))

noisy_pred = true_pts + np.random.normal(scale = 1E-2, size = true_pts.shape)

init_weights = np.random.normal(size=dim)

print('initial weights',init_weights)
print('initial RMS loss',rms_loss(init_weights,noisy_pred,data))

def stars_wrapper(iterate,dim=10):
    weights = iterate[0:10]
    #print(weights)
    pred = iterate[10:].reshape(-1,dim)
    #print('initial first component',noisy_pred[:,0])
    #print('current first component',pred[:,0])
    return rms_loss(weights,pred,data)

#stars setup
maxit = 500
init_pt = np.hstack((init_weights,noisy_pred.flatten()))    
ntrials = 1
f_avr = np.zeros(maxit+1)  #set equal to number of iterations + 1

for trial in range(ntrials):
    #sim setup
    test = Stars_sim(stars_wrapper, init_pt, L1 = 400.0, var = 1E-4, verbose = False, maxit = maxit)
    test.STARS_only = True
    test.get_mu_star()
    test.get_h()
    # do 100 steps
    while test.iter < test.maxit:
        test.step()
    
    #update average of f
    f_avr += test.fhist  

print('STARS min',test.x[0:10])
f2_avr = np.zeros(maxit+1)

for trial in range(ntrials):
    #sim setup
    test = Stars_sim(stars_wrapper, init_pt, L1 = 400.0, var = 1E-4, verbose =False, maxit = maxit)
    #test.STARS_only = True
    test.get_mu_star()
    test.get_h()
    # adapt every 10 timesteps using quadratic(after inital burn)
    # test.train_method = 'GQ'
    test.adapt = 100 # Sets number of sub-cylcing steps
    
    # do 100 steps
    while test.iter < test.maxit:
        test.step()  
        if test.active is not None and test.iter % 10 == 0:
            print('Iteration', test.iter)
            print('Active dimension',test.active.shape[1])
            print('Active weights',test.wts[0:test.active.shape[1]+1])
            print('True active variable comps.',test.active[0:-1:10])
    f2_avr += test.fhist
    print('trial',trial,' minval',test.fhist[-1])
    print(test.x[0:10])

f_avr /= ntrials
f2_avr /= ntrials


plt.semilogy(f_avr,label='Stars')
plt.semilogy(f2_avr, label='Astars')
plt.legend()
plt.show()
