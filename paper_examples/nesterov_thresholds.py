#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 9/1/2020

@author: jordan/varis
"""

import numpy as np
import matplotlib.pyplot as plt
import timeit

import active_subspaces as ss   
from astars.stars_sim import Stars_sim
from astars.utils.misc import subspace_dist, find_active

class nesterov:
    
    def __init__(self, dim = 10, sig = 1E-16):
        self.dim = dim
        self.sig = sig
        self.L1 = 2**9
        self.var = self.sig**2
        self.name = 'Nesterov'
        self.fstar = 0
    
    def __call__(self,x):
        
        temp = np.arange(self.dim,dtype=float)
        weights = 2**((-1)**temp*temp)
        y = np.copy(x)
        y *= y
        ans = np.dot(weights,y) +self.sig*np.random.randn(1)

        return ans
        
f = nesterov()

this_init_pt = 0.1*np.random.randn(f.dim)

ntrials = 1
maxit = 35000

dim = f.dim

import active_subspaces as ss

sub_sp = ss.subspaces.Subspaces()
train_size = 500*(dim+2)*(dim+1)//2
print(train_size)

train_set = np.random.rand(train_size,dim)
for loop in range(2):
    if loop != 0: #append new data
        new_pts = np.random.rand(train_size,dim)
        train_set = np.vstack((train_set,new_pts))
        print('training data size',train_set.shape)
    #train active subspace
    f_data = f(train_set.T)
    print('data size', f_data.shape)
    #don't normalize 
    sub_sp.compute(X=train_set,f=f_data,sstype='QPHD')
    #usual threshold
    adim = find_active(sub_sp.eigenvals,sub_sp.eigenvecs)
    print(adim)
    print(sub_sp.eigenvals)
    #print('Subspace Distance',subspace_dist(true_as,sub_sp.eigenvecs[:,0:adim]))
    
thresh = sub_sp.eigenvals
tsum = np.sum(thresh)
thresholds = [1-thresh[9]/tsum,1-np.sum(thresh[8:9])/tsum, 1-np.sum(thresh[7:9])/tsum, 1-np.sum(thresh[6:9])/tsum, 1-np.sum(thresh[5:9])/tsum, 1-np.sum(thresh[4:9])/tsum, 1-np.sum(thresh[3:9])/tsum, 1-np.sum(thresh[2:9])/tsum]

f_avr = np.zeros(maxit+1)
f2_avr = np.zeros((maxit+1,np.size(thresholds)))

# Start the clock!
start = timeit.default_timer()

# STARS
for trial in range(ntrials):
    test = Stars_sim(f, this_init_pt, L1 = f.L1, var = f.var, verbose = False, maxit = maxit)
    test.STARS_only = True
    test.get_mu_star()
    test.get_h()
    # do stars steps
    while test.iter < test.maxit:
        test.step()
	    
    #update average of f
    f_avr += test.fhist
    print('STARS trial',trial,' minval',test.fhist[-1])
    
adim_sto = np.zeros(np.size(thresholds))

for i in range(np.size(thresholds)):
        
    for trial in range(ntrials):
        test = Stars_sim(f, this_init_pt, L1 = f.L1, var = f.var, verbose = False, maxit = maxit)
        test.get_mu_star()
        test.get_h()
        test.train_method = 'GQ'
        #test.adapt = 3.0*f.dim # Sets number of sub-cylcing steps
        test.adapt = 10
        test.regul = None #test.sigma
        test.threshold = thresholds[i]
	# do steps
        while test.iter < test.maxit:
            test.step()
        
        adim_sto[i] += test.adim  # store final adim

        f2_avr[:,i] += test.fhist
        print('ASTARS trial',trial,' minval',test.fhist[-1])
	    

# Stop the clock!
stop = timeit.default_timer()

# Difference stop-start tells us run time
time = stop - start

print('the time of this experiment was:    ', time/3600, 'hours')

f_avr /= ntrials
f2_avr /= ntrials
adim_sto /= ntrials

print('average active dimension found for each threshold',adim_sto)


print(f_avr)
print(f2_avr)
plt.semilogy(np.abs(f_avr-f.fstar),label='STARS')
for i in range(np.size(thresholds)):
    plt.semilogy(np.abs(f2_avr[:,i]-f.fstar), label='ASTARS, thresh='+str(thresholds[i])+', avg adim='+str(adim_sto[i]))

plt.title(f.name)
plt.legend()
plt.show()
