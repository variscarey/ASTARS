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
from astars.utils.misc import subspace_dist


mag = 1
dim = 10
#weights = np.random.randn(10)
weights=np.ones(dim)
true_as = weights / np.linalg.norm(weights)

our_sig=1E-6
our_var=our_sig**2

def toy_f(x,sig=our_sig):
    return mag*(np.dot(weights,x))**2 + sig*np.random.randn(1)

our_L1 = 2.0*mag*dim



init_pt = np.random.randn(dim)
ntrials = 500
maxit = 200

f_avr = np.zeros(maxit+1)  #set equal to number of iterations + 1

for trial in range(ntrials):
    #sim setup
    test = Stars_sim(toy_f, init_pt, L1 = our_L1, var = our_var, verbose = False, maxit = maxit)

    test.STARS_only = True
    test.update_L1 = True
    test.get_mu_star()
    test.get_h()
    # do 100 steps
    while test.iter < test.maxit:
        test.step()
    
    #update average of f
    f_avr += test.fhist  
    print('STARS trial',trial,' minval',test.fhist[-1])
f2_avr = np.zeros(maxit+1)

for trial in range(ntrials):
    #sim setup

    test = Stars_sim(toy_f, init_pt, L1 = our_L1, var = our_var, verbose = False, maxit = maxit)

    #test.STARS_only = True
    test.update_L1 = True
    test.get_mu_star()
    test.get_h()
    # adapt every 10 timesteps using quadratic(after inital burn)
    test.train_method = 'GQ'
    test.adapt = 20 # Sets number of sub-cylcing steps
    
    # do 100 steps
    while test.iter < test.maxit:
        test.step()  
        if test.iter % 20 == 0 and test.active is not None:
            print('Step',test.iter,'Active dimension',test.active.shape[1])
            print('Subspace Distance',subspace_dist(true_as,test.active))
    f2_avr += test.fhist
    print('ASTARS trial',trial,' minval',test.fhist[-1])

f_avr /= ntrials
f2_avr /= ntrials
 
plt.semilogy(f_avr,label='Stars')
plt.semilogy(f2_avr, label='Astars')
plt.legend()
plt.show()
