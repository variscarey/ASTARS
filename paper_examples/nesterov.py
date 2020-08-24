#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6/10/2020

@author: Jordan Hall and Varis Carey
"""

import numpy as np
import matplotlib.pyplot as plt  
from astars.stars_sim import Stars_sim

def nesterov_f(x,var=1E-2):
    dim=x.shape[0]
    temp=np.arange(dim,dtype=float)
    weights=2**((-1)**temp*temp)
    y=np.copy(x)
    y*=y
    ans=np.dot(weights,y) +var*np.random.randn(1)
    #print(ans.shape)
    return ans
    

init_pt=5*np.random.randn(10)
ntrials = 2
maxit = 2000
f_avr = np.zeros(maxit+1)  #set equal to number of iterations + 1

for trial in range(ntrials):
    #sim setup
    test = Stars_sim(nesterov_f, init_pt, L1 = 2**11, var = 1E-4, verbose = False, maxit = maxit)
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
    test = Stars_sim(nesterov_f, init_pt, L1 = 2**11, var = 1E-4, verbose = False, maxit = maxit)
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
