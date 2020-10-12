#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 9/1/2020

@author: jordan/varis
"""

import numpy as np
import matplotlib.pyplot as plt
import timeit

import active_subspaces as ss   
from astars.stars_sim import Stars_sim
from astars.utils.misc import subspace_dist, find_active

class nesterov:
    
    def __init__(self, dim = 10, sig = 1E-3):
        self.dim = dim
        self.sig = sig
        self.L1 = 2**9
        self.var = self.sig**2
        self.name = 'Nesterov'
        self.fstar = 0
    
    def __call__(self,x):
        
        temp = np.arange(self.dim,dtype=float)
        weights = 2**((-1)**temp*temp)
        y = np.copy(x)
        y *= y
        ans = np.dot(weights,y) +self.sig*np.random.randn(1)

        return ans
        
f = nesterov()

this_init_pt = np.random.randn(f.dim)

ntrials = 20
maxit = 1000

dim = f.dim


f_avr = np.zeros(maxit+1)


# Start the clock!
start = timeit.default_timer()

# STARS
for trial in range(ntrials):
    test = Stars_sim(f, this_init_pt, L1 = f.L1, var = f.var, verbose = False, maxit = maxit)
    test.STARS_only = True
    test.get_mu_star()
    test.get_h()
    # do stars steps
    while test.iter < test.maxit:
        test.step()
	    
    #update average of f
    f_avr += test.fhist
    print('STARS trial',trial,' minval',test.fhist[-1])
    
a_dims = [1,2,4,8]
n_a_dims = np.size(a_dims)
f2_avr = np.zeros((maxit+1,n_a_dims))
j=0

for i in a_dims:
    for trial in range(ntrials):
        test = Stars_sim(f, this_init_pt, L1 = f.L1, var = f.var, verbose = False, maxit = maxit)
        test.get_mu_star()
        test.get_h()        
        test.set_dim = True
        test.adim = i # hard-code adim
        test.train_method = 'GQ'
        #test.adapt = 3.0*f.dim # Sets number of sub-cylcing steps
        test.adapt = 25
        test.regul = None #test.sigma

	# do steps
        while test.iter < test.maxit:
            test.step()

        f2_avr[:,j] += test.fhist
        print('ASTARS trial',trial,' minval',test.fhist[-1])
    j += 1
	    

# Stop the clock!
stop = timeit.default_timer()

# Difference stop-start tells us run time
time = stop - start

print('the time of this experiment was:    ', time/3600, 'hours')

f_avr /= ntrials
f2_avr /= ntrials




print(f_avr)
print(f2_avr)
plt.semilogy(np.abs(f_avr-f.fstar),label='STARS')
for i in range(0,n_a_dims):
    plt.semilogy(np.abs(f2_avr[:,i]-f.fstar), label='ASTARS, fixed j='+str(a_dims[i]))

plt.title(f.name)
plt.legend()
plt.show()
