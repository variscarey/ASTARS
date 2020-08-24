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

def toy_f(x,sig=1E-4):
    return x[0]**2 + sig*np.random.randn(1)
    
dim = 40
init_pt= np.random.randn(dim,1)
ntrials = 2
maxit = 1000
f_avr = np.zeros(maxit+1)  #set equal to number of iterations + 1

for trial in range(ntrials):
    #sim setup
    test = Stars_sim(toy_f, init_pt, L1 = 2.0, var = 1E-8, verbose = False, maxit = maxit)
    test.STARS_only = True
    test.get_mu_star()
    test.get_h()
    # do 100 steps
    while test.iter < test.maxit:
        test.step()
    
    #update average of f
    f_avr += test.fhist  
    
f2_avr = np.zeros(maxit+1)



for trial in range(ntrials):
    #sim setup
    test = Stars_sim(toy_f, init_pt, L1 = 2.0, var = 1E-8, verbose = True, maxit = maxit)
    #test.STARS_only = True
    test.get_mu_star()
    test.get_h()
    # adapt every 10 timesteps using quadratic(after inital burn)
    test.train_method = 'GQ'
    test.adapt = 10 # Sets number of sub-cylcing steps
    
    # do 100 steps
    while test.iter < test.maxit:
        test.step()    
    f2_avr += test.fhist
    print('trial',trial,' minval',test.fhist[-1])

f_avr /= ntrials
f2_avr /= ntrials
 
plt.semilogy(f_avr,label='Stars')
plt.semilogy(f2_avr, label='Astars')
plt.legend()
plt.show()
