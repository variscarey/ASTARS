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

def poly_mod(pred,weights):
    return pred@weights

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
   
    RMS = np.sum((poly_mod(pred,weights)-data)**2)
    return RMS

num_features = 6
num_train = 20
true_pts = np.random.uniform(low=-1.0,high=1.0, size = num_train)
#build polynomial features
train_feat = np.vander(true_pts,N=num_features+1,increasing=True)[:,1:]

#build synthetic data
data = noisy_cubic(true_pts)
#print('true points',true_pts)
print('true data',data)
#initialize model weights
#init_weights = np.random.uniform(low=-1.0,high=1.0,size=num_features)
init_weights = np.zeros(num_features)
init_weights[0]=-6.0
init_weights[2]=1

print('initial weights',init_weights)
print('initial RMS loss',rms_loss(init_weights,train_feat,data))

def stars_wrapper(iterate,dim=10):
    return rms_loss(iterate,train_feat,data)
    #print('initial first component',noisy_pred[:,0])
    #print('current first component',pred[:,0])
#    return rms_loss(weights,pred,data)

#stars setup
maxit = 100
init_pt = np.copy(init_weights)    
ntrials = 1
f_avr = np.zeros(maxit+1)  #set equal to number of iterations + 1
myL1= 2.0*np.max(train_feat**2)
print('L1=',myL1)

for trial in range(ntrials):
    #sim setup
    test = Stars_sim(stars_wrapper, init_pt, L1 = myL1, var = 1E-4, verbose = False, maxit = maxit)
    test.STARS_only = True
    test.get_mu_star()
    test.get_h()
    # do 100 steps
    while test.iter < test.maxit:
        test.step()
    
    #update average of f
    f_avr += test.fhist  

print('STARS min',test.x)
s_min = test.x
f2_avr = np.zeros(maxit+1)

init_pt=np.copy(init_weights)
for trial in range(ntrials):
    #sim setup
    test = Stars_sim(stars_wrapper, init_pt, L1 = myL1, var = 1E-4, verbose =False, maxit = maxit)
    #test.STARS_only = True
    test.get_mu_star()
    test.get_h()
    # adapt every 10 timesteps using quadratic(after inital burn)
    test.train_method = 'GQ'
    test.adapt = 200 # Sets number of sub-cylcing steps
    
    # do 100 steps
    while test.iter < test.maxit:
        test.step()  
        if test.active is not None and test.iter % 10 == 0:
            print('Iteration', test.iter)
            print('Active dimension',test.active.shape[1])
            print('Active weights',test.wts)
            print('True active variable comps.',test.active)
    f2_avr += test.fhist
    print('trial',trial,' minval',test.fhist[-1])
    print(test.x)

f_avr /= ntrials
f2_avr /= ntrials

as_min = test.x

print('Stars_min',s_min)
print('Astars min',as_min)


plt.semilogy(f_avr,label='Stars')
plt.semilogy(f2_avr, label='Astars')
plt.legend()
plt.show()
