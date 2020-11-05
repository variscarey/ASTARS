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
        self.name = 'Example 5: STARS vs FAASTARS Convergence with Various Fixed $j=\dim \mathcal{A}$ '
        self.fstar = 0
    
    def __call__(self,x):
        
        temp = np.arange(self.dim,dtype=float)
        weights = 2**((-1)**temp*temp)
        y = np.copy(x)
        y *= y
        ans = np.dot(weights,y) +self.sig*np.random.randn(1)

        return ans
        
f = nesterov()

params = {'legend.fontsize': 28,'legend.handlelength': 3}
plt.rcParams["figure.figsize"] = (60,40)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['font.size'] = 30
plt.rcParams['figure.titlesize'] = 'xx-large'
plt.rcParams.update(params)


np.random.seed(9)
this_init_pt = np.random.randn(f.dim)

ntrials = 20
maxit = 600

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

    while test.iter < test.maxit:
        test.step()
	    
    #update average of f
    f_avr += test.fhist
    print('STARS trial',trial,' minval',test.fhist[-1])
    
a_dims = [2]
n_a_dims = np.size(a_dims)
f2_avr = np.zeros((maxit+1,n_a_dims))
j=0

for i in a_dims:
    for trial in range(ntrials):
        test = Stars_sim(f, this_init_pt, L1 = f.L1, var = f.var, verbose = False, maxit = maxit)
        test.get_mu_star()
        test.get_h()        
        test.set_dim = True
        test.subcycle = True
        test.adim = i # hard-code adim
        test.train_method = 'GQ'
        test.adapt = 3.0*f.dim
        test.regul = test.sigma**2

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




#print(f_avr)
#print(f2_avr)
plt.semilogy(np.abs(f_avr-f.fstar),lw = 5, label='STARS', color='black')

color_pal = ['blue', 'red', 'orange']
ls_pal = [':','--','-.']

for i in range(0,n_a_dims):
    plt.semilogy(np.abs(f2_avr[:,i]-f.fstar), lw = 5, label = 'FAASTARS, fixed j='+str(a_dims[i]), ls= ls_pal[i], color = color_pal[i])

plt.title(f.name)
plt.xlabel('$k$, iteration count')
plt.ylabel('$|f(\lambda^{(k)})-f^*|$')
plt.legend()
plt.show()
