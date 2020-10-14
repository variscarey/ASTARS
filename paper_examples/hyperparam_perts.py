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
    
    def __init__(self, dim = 10, sig = 1E-5):
        self.dim = dim
        self.sig = sig
        self.L1 = 2.0
        self.var = self.sig**2
        self.name = 'Example 4: STARS Convergence with Various Scalings $c L_1$'
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

np.random.seed(9)
this_init_pt = 10*np.random.randn(f.dim)

ntrials = 100
maxit = 2000

f_avr = np.zeros((maxit+1,np.size(K)))


params = {'legend.fontsize': 24,'legend.handlelength': 3}
plt.rcParams["figure.figsize"] = (60,40)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['font.size'] = 30
plt.rcParams['figure.titlesize'] = 'xx-large'
plt.rcParams.update(params)


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

color_pal = ['blue', 'red', 'orange']
ls_pal = [':','--','-.']

for i in range(np.size(K)):
    if i == 0:
        plt.semilogy(np.abs(f_avr[:,i]-f.fstar), lw = 5,label='STARS, $c=$ '+str(K[i]), color='black')
    else:
        plt.semilogy(np.abs(f_avr[:,i]-f.fstar), lw = 5,label='STARS, $c=$ '+str(K[i]), color=color_pal[i-1], ls = ls_pal[i-1])


plt.title(f.name)
plt.xlabel('$k$, iteration count')
plt.ylabel('$|f(\lambda^{(k)})-f^*|$')
plt.legend()
plt.show()
