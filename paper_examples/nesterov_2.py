#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6/10/2020

@author: Jordan Hall and Varis Carey
"""

import numpy as np
import matplotlib.pyplot as plt  
from astars.stars_sim import Stars_sim
import timeit



adim = 10
dim = 50


def nesterov_2_f(x,sig=1E-6):
    ans = 0.5*(x[0]**2 + x[adim-1]**2) - x[0]
    for i in range(adim-1):
        ans += 0.5*(x[i] - x[i+1])**2
    ans += sig*np.random.randn(1)
    return ans

init_pt = np.random.randn(dim)

print(nesterov_2_f(init_pt))

ntrials = 10
maxit = 10000

f_avr = np.zeros(maxit+1)  #set equal to number of iterations + 1

for trial in range(ntrials):
    #sim setup
    test = Stars_sim(nesterov_2_f, init_pt, L1 = 4, var = 1E-12, verbose = False, maxit = maxit)
    test.STARS_only = True
    test.get_mu_star()
    test.get_h()
    # do 100 steps
    while test.iter < test.maxit:
        test.step()
        #if test.iter % 100 == 0:
            #print('iter',test.iter,test.fhist[test.iter])
    
    #update average of f
    f_avr += test.fhist  
    
f2_avr = np.zeros(maxit+1)

# Start the clock!
start = timeit.default_timer()

for trial in range(ntrials):
    #sim setup
    test = Stars_sim(nesterov_2_f, init_pt, L1 = 4, var = 1E-12, verbose = True, maxit = maxit)
    #test.STARS_only = True
    test.get_mu_star()
    test.get_h()
    test.regul = None
    test.train_method = 'GQ'
    test.adapt = 100 # Sets number of sub-cylcing steps
    test.threshold = 0.99999

    
    # do 100 steps
    while test.iter < test.maxit:
        test.step()    
        #if test.iter % 100 == 0:
         #   print('iter',test.iter,test.fhist[test.iter])
    f2_avr += test.fhist
    #print('trial',trial,' minval',test.fhist[-1])

# Stop the clock!
stop = timeit.default_timer()

# Difference stop-start tells us run time
time = stop - start

print('the time of this experiment was:    ', time/3600, 'hours')


f_avr /= ntrials
f2_avr /= ntrials


fstar = .5*(-1.0 + 1.0 / (adim + 1))
 
plt.semilogy(np.absolute(fstar-f_avr),label='Stars')
plt.semilogy(np.absolute(fstar-f2_avr), label='Astars')

plt.legend()
plt.show()
