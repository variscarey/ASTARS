#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:06:33 2020

@author: cmg
"""

import numpy as np
import scipy as sp
import scipy.optimize as opt
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

class toy2:
        def __init__(self, mag = 1.0, dim = 20, weights = None, sig = 1E-6):
                self.mag = mag
                self.dim = dim
                self.L1 = self.mag*self.dim*2.0
                self.sig = sig
                self.var = self.sig**2
                self.name = 'Example 1: STARS, FAASTARS, and ASTARS Convergence'
                self.nickname = 'toy_2'
                self.fstar = 0
                if weights is None:
                    self.weights = np.ones(self.dim)
                self.active = self.weights / np.linalg.norm(self.weights)
                self.active = self.active.reshape(-1,1)
                
                self.maxit = 700 # because FAASTARS converged in about 350 iterations...
                self.ntrials = 1000
                self.adapt = 2*dim
                self.regul = None
                self.threshold = 0.99
                self.initscl = 1.0
            
   
        def __call__(self, x):
            return self.mag*(np.dot(self.weights,x)**2) + self.sig*np.random.randn(1)


            
#sphere function, was toy 1
           
class sphere:
        def  __init__(self, mag = 1.0, dim = 20, adim = 10, sig = 1E-3):
              self.dim = dim
              self.adim = adim
              self.sig = sig
              self.mag = mag
              self.active = np.eye(dim,adim)
              self.L1 = 2.0*self.mag
              self.var = self.sig**2
              self.name = 'Example 2: STARS, FAASTARS, and ASTARS Convergence'
              self.nickname = 'sphere'
              self.fstar = 0
              
              self.maxit = 1400 # because FAASTARS converged in about 700 iterations
              self.ntrials = 100
              self.adapt = dim
              self.regul = self.sig**2
              self.threshold = 0.999
              self.initscl = 10.0
            
        def __call__(self,X):
            return self.mag*np.sum(X[0:self.adim]**2) + self.sig*np.random.randn(1)
 
class nesterov_2_f:
    
    def __init__(self, dim = 50, adim = 5, sig = 1E-4):
        self.dim = dim
        self.adim = adim
        self.sig = sig
        self.active = np.eye(dim,adim)
        self.L1 = 4.0
        self.var = self.sig**2
        self.name = 'Example 3: STARS, FAASTARS, and ASTARS Convergence'
        self.nickname = 'nesterov_2'
        self.fstar = .5*(-1 + 1 / (self.adim + 1))
        
        self.maxit = 12000 # because FAASTARS converges in about 6000 iterations
        self.ntrials = 50
        self.adapt = 2*dim
        self.regul = self.sig**2
        self.threshold = 0.9999
        self.initscl = 50.0
    
    def __call__(self,x):
        
        ans = 0.5*(x[0]**2 + x[self.adim-1]**2) - x[0]
        for i in range(self.adim-1):
            ans += .5*(x[i] - x[i+1])**2
            if ans.ndim == 1:
                ans += self.sig*np.random.randn(1)
            else:
                ans += self.sig*np.random.randn(ans.size)
        return ans
    


#plotting parameters and definitions
toy2f = toy2()
sph = sphere()
nest = nesterov_2_f()

for f in {toy2f, sph, nest}:

    # initiate storage
    n_samps = f.maxit
    x_sto = np.zeros((n_samps,f.dim))
    f_sto = np.zeros(n_samps)

   # perform random draws, store x and f values
    for i in range(n_samps):
        x = np.random.randn(f.dim) # already normalized to [-1,1]
        x_sto[i,:] = x.T
        f_sto[i] = f(x)
        
    
    # Make quadratic response surface
    subspace = ss.subspaces.Subspaces()
    subspace.compute(X = x_sto, f = f_sto, nboot = 0, sstype = 'QPHD')
    RS = ss.utils.response_surfaces.PolynomialApproximation(2)
    y = x_sto.dot(subspace.W1)
    RS.train(y, f_sto)
    
    print('Rsqr',RS.Rsqr,'using',n_samps,'samples for problem', f.nickname)
    
    avdom = ss.domains.UnboundedActiveVariableDomain(subspace)
    
    ystar, fstar = ss.optimizers.av_minimize(lambda x: RS.predict(np.array([x]))[0], avdom)

  
            
    print(ystar, fstar)



































