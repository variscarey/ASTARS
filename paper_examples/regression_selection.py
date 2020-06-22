# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:24:41 2020

@author: varis
"""

import numpy as np
from astars.stars_sim import Stars_sim
import matplotlib.pyplot as plt


def noisy_cubic(x,var=1E-4):
    return x**3-6*x + np.random.normal(scale=np.sqrt(var),size=x.shape)

def poly_mod(weights,pred):
    return np.dot(weights,pred)

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
   
    RMS = np.sum(poly_mod(pred,weights)-data)**2
    return RMS

num_features = 10
num_train = 20
true_pts = np.random.normal(scale = 4.0, size = num_train)

train_feat = np.vander(true_pts,N=num_features)[:,1:]

data = noisy_cubic(true_pts)
#print('true points',true_pts)
print('true data',data)
#initialize model weights
init_weights = np.random.normal(size=num_features)

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
