#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 9/1/2020

@author: jordan/varis
"""

import numpy as np
import matplotlib.pyplot as plt

import active_subspaces as ss   
from astars.stars_sim import Stars_sim
from astars.utils.misc import subspace_dist, find_active

class nesterov_2_f:
    
    def __init__(self, dim = 50, adim = 5, sig = 1E-4):
        self.dim = dim
        self.adim = adim
        self.sig = sig
        
        self.L1 = 4.0
        self.var = self.sig**2
        self.name = 'Nesterov 2'
        self.fstar = .5*(-1 + 1 / (self.adim + 1))
    
    def __call__(self,x):
        
        ans = 0.5*(x[0]**2 + x[self.adim-1]**2) - x[0]
        for i in range(self.adim-1):
            ans += .5*(x[i] - x[i+1])**2
            if ans.ndim == 1:
                ans += self.sig*np.random.randn(1)
            else:
                ans += self.sig*np.random.randn(ans.size)
        return ans
        
thresholds = [0.996,0.997,0.998]

nest = nesterov_2_f()
for this_thresh in thresholds:

    for f in {nest}:
	#for f in {toy2f,sph}:
        dim = f.dim
        init_pt = np.random.randn(dim)
        ntrials = 1
        maxit = 4*dim**2
	#maxit = 1000
        f2_avr = np.zeros(maxit+1)
        f_avr = np.zeros(maxit+1)
        for trial in range(ntrials):
	#sim setup
            test = Stars_sim(f, init_pt, L1 = f.L1, var = f.var, verbose = False, maxit = maxit)
            test.STARS_only = True
            test.get_mu_star()
            test.get_h()
	    # do stars steps
            while test.iter < test.maxit:
                test.step()
	    
	    #update average of f
            f_avr += test.fhist
            print('STARS trial',trial,' minval',test.fhist[-1])
        for trial in range(ntrials):
            #sim setup
            test = Stars_sim(f, init_pt, L1 = f.L1, var = f.var, verbose = False, maxit = maxit)
            test.get_mu_star()
            test.get_h()
            test.train_method = 'GQ'
            test.adapt = 2*dim # Sets number of sub-cylcing steps
            test.regul = None #test.sigma
            test.threshold = this_thresh
	    # do 100 steps
            while test.iter < test.maxit:
                test.step()  

        f2_avr += test.fhist
        print('ASTARS trial',trial,' minval',test.fhist[-1])
	    

        f_avr /= ntrials
        f2_avr /= ntrials
	 
        plt.semilogy(np.abs(f_avr-f.fstar),label='Stars')
        plt.semilogy(np.abs(f2_avr-f.fstar), label='Astars')
        plt.title(f.name)
        plt.legend()
        plt.show()
