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

class weight_sphere:

    def __init__(self, mag = 1.0, dim = 10, adim = 3, weights = None, sig = 1E-3):
                self.mag = mag
                self.dim = dim
                self.L1 = self.mag*self.dim*2.0
                self.sig = sig
                self.var = self.sig**2
                self.name = 'Weeighted Sphere'
                self.nickname = 'wsph'
                self.fstar = 0
                self.adim = adim
                if weights is None:
                    self.weights = np.ones(self.dim)
                else:
                    self.weights = weights
                self.L1 = np.max(self.weights)*2.0*self.mag
                self.active = np.eye(dim,M=self.adim)
                #self.active = self.active.reshape(-1,1)
                
                self.maxit = 2*dim**2
                self.ntrials = 1000
                self.adapt = 2*dim
                self.regul = None
                self.threshold = 0.99
                self.initscl = 1.0
    def __call__(self, x):
            return self.mag*np.sum(self.weights*x[0:self.adim]*x[0:self.adim]) + self.sig*np.random.randn(1)

ws = weight_sphere(weights = np.array([100,10,1],dtype=float))                            
                             
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

#g = 2**np.arange(1,7)
g = np.arange(5,101,step=5)
conv = np.zeros(g.size)
ind = 0

#for f in {toy2f, sph, nest}:
f = ws
#prinp.random.seed(9)
init_pt = np.ones(f.dim)
init_pt /= np.linalg.norm(init_pt)
ntrials = 30
maxit = 900


f_avr = np.zeros(maxit+1)
f_av2 = np.copy(f_avr)   
   
    
    
    # STARS, no weights
for trial in range(ntrials):
    #sim setup
   test = Stars_sim(f, init_pt, L1 = f.L1, var = f.var, verbose = False, maxit = maxit)
   #test.STARS_only = True
   test.get_mu_star()
   test.get_h()
   test.train_method = 'GQ'
   test.threshold = .999
    # do 100 steps
   while test.iter < test.maxit:
       test.step()
    
    #update average of f
   f_avr += test.fhist
        
        # data dump

        
   print('STARS trial',trial,' minval',test.fhist[-1])

   
        
f_avr /= ntrials
 
 

 
 

for trial in range(ntrials):
   #sim setup
   test = Stars_sim(f, init_pt, L1 = f.L1, var = f.var, verbose = False, maxit = maxit)
   #test.STARS_only = True
   test.use_weights = True
   #test.wts = np.sqrt(np.array([1,10,100],dtype=float))
   test.get_mu_star()
   test.get_h()
   test.train_method = 'GQ'
   test.threshold = .999
   while test.iter < test.maxit:
       test.step()
    
    #update average of f
   f_av2 += test.fhist
        
        # data dump

        
   print('STARS trial',trial,' minval',test.fhist[-1])

   
        
f_av2 /= ntrials
 
plt.semilogy(np.abs(f_avr-f.fstar),lw = 5,label='STARS',color=stars_full, ls=sf_ls)
plt.semilogy(np.abs(f_av2-f.fstar),lw = 5,label='weighted STARS',color=active_stars_ref, ls=sf_ls)
 
plt.title(f.name)
plt.xlabel('$k$, iteration count')
plt.ylabel('$|f(\lambda^{(k)})-f^*|$')
plt.legend()
plt.show() 