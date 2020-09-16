#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 9/14/2020

@author: jordan/varis
"""

import numpy as np
import matplotlib.pyplot as plt
import timeit

import active_subspaces as ss   
from astars.stars_sim import Stars_sim
from astars.utils.misc import subspace_dist, find_active

class new_test:
    
    def __init__(self, dim = 10, sig = 1E-2):
        self.dim = dim
        self.sig = sig
        self.L1 = 2.0
        self.var = self.sig**2
        self.name = 'Sphere function'
        self.fstar = 0
       
    
    def __call__(self,x):
        
        temp = np.arange(self.dim,dtype=float)
        weights = np.ones(self.dim)
        y = np.copy(x)
        y *= y
        ans = np.dot(weights,y) +self.sig*np.random.randn(1)

        return ans
        
f = new_test()

K = [1,0.1,0.2,4]
this_init_pt = 10*np.random.randn(f.dim)

ntrials = 10
maxit = 200

f_avr = np.zeros((maxit+1,np.size(K)))


# Start the clock!
start = timeit.default_timer()



for i in range(np.size(K)):


    # STARS
    for trial in range(ntrials):
        test = Stars_sim(f, this_init_pt, L1 = f.L1*K[i], var = f.var, verbose = False, maxit = maxit)
        test.STARS_only = True
        test.get_mu_star()
        test.get_h()
        # do stars steps
        while test.iter < test.maxit:
            test.step()
	    
    #update average of f
    f_avr[:,i] += test.fhist
    print('STARS trial',trial,' minval',test.fhist[-1])
        

	    

# Stop the clock!
stop = timeit.default_timer()

# Difference stop-start tells us run time
time = stop - start

print('the time of this experiment was:    ', time/3600, 'hours')

f_avr /= ntrials


print(f_avr)



for i in range(np.size(K)):
    plt.semilogy(np.abs(f_avr[:,i]-f.fstar), label='STARS, L1 scaled by '+str(K[i]))



plt.title(f.name)
plt.legend()
plt.show()
