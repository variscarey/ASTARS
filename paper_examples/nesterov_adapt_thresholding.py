#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/3/2020 15:06:33 2020

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



        
class nesterov_2:
    
    def __init__(self, dim = 10, sig = 1E-3):
        self.dim = dim
        self.sig = sig
        self.L1 = 2**9
        self.var = self.sig**2
        self.nickname = 'nest_2'
        self.name = 'Example 3: STARS vs FAASTARS With Adaptive Thresholding'
        self.fstar = 0
        self.maxit = 20000
        self.ntrials = 1 #50
        self.adapt = 2*dim
        self.regul = None # maybe - self.sig**2
        self.threshold = .9    
        self.initscl = 1
    
    def __call__(self,x):
        
        temp = np.arange(self.dim,dtype=float)
        weights = 2**((-1)**temp*temp)
        y = np.copy(x)
        y *= y
        ans = np.dot(weights,y) +self.sig*np.random.randn(1)

        return ans
        
class test_weights:
    
    def __init__(self, dim = 10, sig = 1E-3):
        self.dim = dim
        self.sig = sig
        self.L1 = 200
        self.var = self.sig**2
        self.nickname = 'test_weights'
        self.name = 'Example 4: STARS vs FAASTARS With Adaptive Thresholding'
        self.fstar = 0
        self.maxit = 1000
        self.ntrials = 1 #50
        self.adapt = 2*dim
        self.regul = None # maybe - self.sig**2
        self.threshold = .9    
        self.initscl = 1
        
        
    
    def __call__(self,x):
        
        temp = np.arange(self.dim,dtype=float)
        weights = np.zeros(dim)
        weights[0], weights[1] = 100, 1
        y = np.copy(x)
        y *= y
        ans = np.dot(weights,y) +self.sig*np.random.randn(1)

        return ans        
        
#plotting parameters and definitions
nest = nesterov_2()
wt_fn = test_weights()

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

for f in {wt_fn}:

    dim = f.dim
    #np.random.seed(9)
    #init_pt = f.initscl*np.random.randn(dim)
    init_pt = np.ones(dim)
    ntrials = f.ntrials
    maxit = f.maxit

    f_avr = np.zeros(maxit+1)
    f2_avr = np.zeros(maxit+1)
    f3_avr = np.zeros(maxit+1)
    f4_avr = np.zeros(maxit+1)            
    
    # STARS
    for trial in range(ntrials):
    #sim setup
        test = Stars_sim(f, init_pt, L1 = f.L1, var = f.var, verbose = False, maxit = maxit)
        test.STARS_only = True
        test.get_mu_star()
        test.get_h()
        
        # STARS steps
        while test.iter < test.maxit:
            test.step()
    
    #update average of f
        f_avr += test.fhist


    # FAASTARS (3 scenarios: no extensions, adaptive thresholding, and active subcycling)
    for trial in range(ntrials):
        #sim setup
        test = Stars_sim(f, init_pt, L1 = f.L1, var = f.var, verbose = False, maxit = maxit, true_as = None)
        test2 = Stars_sim(f, init_pt, L1 = f.L1, var = f.var, verbose = False, maxit = maxit, true_as = None)
        test3 = Stars_sim(f, init_pt, L1 = f.L1, var = f.var, verbose = False, maxit = maxit, true_as = None)
        
        test.get_mu_star()
        test.get_h()
        test2.get_mu_star()
        test2.get_h()  
        test3.get_mu_star()
        test3.get_h()   
                     
        # adapt every f.adapt timesteps using quadratic(after inital burn)
        test.train_method = 'GQ'
        test2.train_method = 'GQ'
        test3.train_method = 'GQ'
        
        test.adapt = f.adapt # Sets retraining steps
        test2.adapt = f.adapt
        test3.adapt = f. adapt
        
        # Make test2 our adaptive thresholding trial, and test3 our subcycling trial
        test2.threshadapt = True
        test3.subcycle = True

        test.regul = f.regul
        test2.regul = f.regul
        test3.regul = f.regul
        
        test.threshold = f.threshold
        test2.threshold = f.threshold
        test3.threshold = f.threshold
        
        # do 100 steps
        while test.iter < test.maxit:
            test.step()
            test2.step()
            test3.step()
            
            print(test.iter,'Iteration')
            
            if test.iter % 200 == 0:
                print(test2.eigenvals)

        f2_avr += test.fhist
        f3_avr += test2.fhist
        f4_avr += test3.fhist
        


        
    f_avr /= ntrials
    f2_avr /= ntrials
    f3_avr /= ntrials
    f4_avr /= ntrials

    
   
        
    # Stop the clock!
    stop = timeit.default_timer()

    # Difference stop-start tells us run time
    time = stop - start
    print('the time of this experiment was:    ', time/3600, 'hours')
 
    plt.semilogy(np.abs(f_avr-f.fstar),lw = 5,label='STARS',color=stars_full, ls=sf_ls)
    plt.semilogy(np.abs(f2_avr-f.fstar), lw = 5, label='FAASTARS (No Extensions, $\\tau = 0.9$)',color=active_stars_learned ,ls=lr_ls)
    plt.semilogy(np.abs(f3_avr-f.fstar), lw = 5,label = 'FAASTARS (Adaptive Thresholding)',color=active_stars_ref ,ls=rf_ls)
    plt.semilogy(np.abs(f4_avr-f.fstar), lw = 5,label = 'FAASTARS (Active Subcycling)',color='orange' ,ls=rf_ls)    
    plt.title(f.name)
    plt.xlabel('$k$, iteration count')
    plt.ylabel('$|f(\lambda^{(k)})-f^*|$')
    plt.legend()
    plt.show()
    
    plt.plot(test.xhist[1,-1000:], label='FAASTARS (No Extensions, $\\tau = 0.9$)')
    plt.plot(test2.xhist[1,-1000:], label = 'FAASTARS (Adaptive Thresholding)')
    plt.plot(test3.xhist[1,-1000:], label = 'FAASTARS (Active Subcycling)') 
    plt.title(f.name)
    plt.xlabel('$k$, iteration count')
    plt.ylabel('$\lambda^{(k)}_2$') 
    plt.legend()  
    plt.show()


for fa in [f2_avr,f3_avr,f4_avr]:
    slopes = []
    for j in range(80,f2_avr.size):
        fsamp = fa[j-20:j]
        poly = np.polyfit(np.arange(20),fsamp,1)
        slopes.append(poly[0])
    slopes = np.array(slopes)
    plt.plot(slopes)
plt.axhline(-test.sigma*f.dim)
plt.show()