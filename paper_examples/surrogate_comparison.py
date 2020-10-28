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
                self.tr_start = 250 # could actually start at 231, but using "nice" number
            
   
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
              self.shift = 5*np.ones(adim)
              self.tr_start = 250 # could actually start at 231, but using "nice" number
            
        def __call__(self,X):
            return self.mag*np.sum((X[0:self.adim]-self.shift)**2) + self.sig*np.random.randn(1)
 
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
        self.tr_start = 1350 # could actually start at 1,326, but using "nice" number
    
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

params = {'legend.fontsize': 28,'legend.handlelength': 3}
plt.rcParams["figure.figsize"] = (60,40)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['font.size'] = 30
plt.rcParams['figure.titlesize'] = 'xx-large'
plt.rcParams.update(params)

toy2f = toy2()
sph = sphere()
nest = nesterov_2_f()



for f in {nest}:
    
    n_samps = f.tr_start
    while n_samps <= f.maxit:

        # initiate storage
        x_sto = np.zeros((n_samps, f.dim))
        f_sto = np.zeros((n_samps, 1))

       # perform random draws, store x and f values
        for i in range(n_samps):
            x = np.random.rand(f.dim)
            x_sto[i,:] = x.T
            f_sto[i] = f(x)
        
    
        # Make quadratic response surface using Paul's library -- no actual active subspace computation involved here!
        RS = ss.utils.response_surfaces.PolynomialApproximation(2)
        y = x_sto 
        RS.train(y, f_sto)
        #print(RS.poly_weights[0])
    
        print('Rsqr',RS.Rsqr,'using',n_samps,'samples for problem', f.nickname, '\n')
    
        def surrogate(x):
            return RS.predict(np.array([x]))[0]
         
        def grad_surrogate(x):
            return RS.predict(np.array([x]), compgrad=True)[1].flatten()
        
        print('check surrogate by evaluating at vector of all ones', surrogate(np.ones(f.dim)))
        print('check compgrad of surrogate by evaluating at vector of all ones',grad_surrogate(np.ones(f.dim)))
        # print('quadratic surrogate coefficients', RS.g, RS.H)
    
    
        x0 = 10 * np.random.rand(f.dim)
        bd = opt.Bounds(np.min(x_sto), np.max(x_sto)) # Make domain for optimization bounding observed random samples via hyperrectangle
    
        surrogate_min = opt.minimize(surrogate, x0, jac = grad_surrogate, bounds = bd)
    
        print('surrogate at x0:', surrogate(x0) , 'f at x0 is', f(x0), '\n')
    
        print('obtained min:',surrogate_min.x, '\n surrogate evaluated at min:' , surrogate(surrogate_min.x), '\n f evaluated at min:' , f(surrogate_min.x), '\n \n')
        
        n_samps += 50
    

