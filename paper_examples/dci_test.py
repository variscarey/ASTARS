#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:06:33 2020

@author: cmg
"""

import numpy as np
import matplotlib.pyplot as plt
import timeit
import active_subspaces as ss   
from astars.stars_sim import Stars_sim
from astars.utils.misc import subspace_dist, find_active
import pandas as pd

###############################################################################
############ Set user-desired file path for storing output data!!! ############
###############################################################################
user_file_path = '/home/ccm/Desktop/'
###############################################################################
###############################################################################
###############################################################################




class data_misfit:
    def __init__(self, dim = 20, weights= None, data = 20, sig = 1E-3):
        self.dim = dim
        self.L1 = self.dim*2.0
        self.sig = sig
        self.var = self.sig**2
        self.name = 'DCI Example 1: '
        self.nickname = 'DCI'
        self.fstar = 0
        self.var = self.sig**2
        self.data = data
        if weights is None:
            self.weights = np.ones(self.dim)
            self.active = self.weights / np.linalg.norm(self.weights)
            self.active = self.active.reshape(-1,1)
        else:
            self.weights = weights
                
        self.maxit = 10*dim**2
        self.ntrials = 1000
        self.adapt = 2*dim
        
        self.threshold = 0.99
        self.initscl = 1.0
        self.inputs = None
        self.outputs = None
                        
    def __call__(self, x):
        temp = x.flatten()
        return (self.qoi(temp)-self.data)**2 # + self.sig*np.random.randn(1)

    
    def qoi(self,x):
        ans = np.dot(self.weights,x)  + self.sig*np.random.randn(1)
        if self.inputs is None:
            self.inputs = np.array(x).reshape(1,x.size)
            self.outputs = np.array(ans).reshape(-1,1)
        else:
            self.inputs = np.vstack((self.inputs,x.reshape(1,x.size)))
            self.outputs = np.vstack((self.outputs,np.array(ans).reshape(-1,1)))
        
        return ans

    


#plotting parameters and definitions
wt = np.zeros(40)
wt[0] = 1
#dci = data_misfit(dim=10, weights = wt)
dci = data_misfit(dim=10)

params = {'legend.fontsize': 28,'legend.handlelength': 3}
plt.rcParams["figure.figsize"] = (60,40)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['font.size'] = 30
plt.rcParams['figure.titlesize'] = 'xx-large'
plt.rcParams.update(params)

stars_full, sf_ls = 'red', '--'
active_stars_learned, lr_ls = 'black', '-.'
active_stars_ref, rf_ls = 'blue', ':'


# Start the clock!
start = timeit.default_timer()



for f in {dci}:

    #np.random.seed(9)
    init_pt = np.zeros(f.dim) #prior mean
    #init_pt /= np.linalg.norm(init_pt)
    ntrials = 3
    maxit = 500


    f_avr = np.zeros(maxit+1)
    f2_avr = np.zeros(maxit+1)
    trial_final = np.zeros(f.dim)
    
    # STARS no sphere
    for trial in range(ntrials):
    #sim setup
        test = Stars_sim(f, init_pt, L1 = None, var = None, verbose = False, maxit = maxit)
        test.STARS_only = True
        test.get_mu_star()
        test.get_h()
        test.update_L1 = True
        test2 = Stars_sim(f, init_pt, L1 = None, var = None, verbose = True, maxit = maxit)
        test2.update_L1 = True
        #test.STARS_only = True
        test2.get_mu_star()
        test2.get_h()
        test2.train_method = 'GQ'
        test2.adapt = 2*f.dim
        test2.regul = test2.var
        test2.pad_train = 2.0
        test2.explore_weight = 2.0
        #test2.regul = None
    
        dist = None
        L1_hist = None
        while test.iter < test.maxit:
            test.step()
            test2.step()
            if test2.active is not None:
                temp = np.array(subspace_dist(test2.active,f.active))
                if dist is None:
                    dist = np.copy(temp)
                    L1_hist = np.copy(np.array(test2.L1))
                else:
                    dist = np.append(dist,temp)
                    L1_hist = np.append(L1_hist,test2.L1)
                     
    #update average of f
        f_avr += test.fhist
        f2_avr += test2.fhist
        trial_final += test2.active@test2.active.T@test2.x
        #final answer 
        #project test2 solution
        
        
        
        # data dump

        
        print('STARS trial',trial,' minval',test.fhist[-1])

   
        
    f_avr /= ntrials
    f2_avr /= ntrials
    trial_final /= ntrials
    
    

 
    plt.semilogy(np.abs(f_avr-f.fstar),lw = 5,label='DCI',color=stars_full, ls=sf_ls)
    plt.semilogy(np.abs(f2_avr-f.fstar),lw = 5,label='DCI ASTARS',color='blue', ls=sf_ls)
    for data in test2.as_comp_step:
        plt.axvline(data)
    plt.title(f.name)
    plt.xlabel('$k$, iteration count')
    plt.ylabel('$|f(\lambda^{(k)})-f^*|$')
    plt.legend()
plt.show()    

#fit active subspace using all data
sub = ss.subspaces.Subspaces()
g_surr = ss.utils.response_surfaces.PolynomialApproximation(N=2)
g_surr.train(dci.inputs, dci.outputs, regul = test2.regul) #regul = self.var)
mud_grad = g_surr.predict(trial_final.reshape(1,trial_final.size),compgrad = True)
