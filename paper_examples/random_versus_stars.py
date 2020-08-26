# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 11:26:28 2020

@author: varis
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:06:33 2020

@author: cmg
"""

import numpy as np
import matplotlib.pyplot as plt

#import active_subspaces as as   
from astars.stars_sim import Stars_sim
from astars.utils.misc import subspace_dist,find_active

mag = 1E4
dim = 20
#weights = np.random.randn(10)
weights=np.ones(dim)
true_as = weights / np.linalg.norm(weights)
print('True active subspaces',true_as)

def toy_f(x,var=1E-2):
    if x.ndim > 1:
        return np.sum(x,axis = 1)**2+ var*np.random.randn(1)
    else:
        return np.sum(x)**2 + var * np.random.randn(1)
    

init_pt = np.random.randn(dim)
import active_subspaces as ss

sub_sp = ss.subspaces.Subspaces()
train_size = (dim+2)*(dim+1)//2
print(train_size)

train_set = np.random.randn(train_size,dim)
for loop in range(7):
    if loop != 0: #append new data
        new_pts = np.random.randn(train_size,dim)
        train_set = np.vstack((train_set,new_pts))
        print('training data size',train_set.shape)
    #train active subspace
    f_data = toy_f(train_set)
    print('data size', f_data.shape)
    #don't normalize 
    sub_sp.compute(X=train_set,f=f_data,sstype='QPHD')
    #usual threshold
    adim = find_active(sub_sp.eigenvals,sub_sp.eigenvecs)
    print('Subspace Distance',subspace_dist(true_as,sub_sp.eigenvecs[:,0:adim]))
    
 
test = Stars_sim(toy_f, init_pt, L1 = 2.0, var = 1E-4, verbose = False, maxit = train_size*3)
test.STARS_only = True
test.get_mu_star()
test.get_h()
# do 100 steps
while test.iter < test.maxit:
    test.step()
    if test.iter > (dim+2)*(dim+1)//4:
        #compute active subspace
        train_x = np.hstack((test.xhist[:,0:test.iter+1],test.yhist[:,0:test.iter]))
        train_f = np.hstack((test.fhist[0:test.iter+1],test.ghist[0:test.iter]))
        train_x = train_x.T
        sub_sp.compute(X=train_x,f=train_f,sstype='QPHD')
        adim = find_active(sub_sp.eigenvals,sub_sp.eigenvecs)
        print('Subspace Distance',subspace_dist(true_as,sub_sp.eigenvecs[:,0:adim]))
        
 
    
