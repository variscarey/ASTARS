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

def nesterov_2_f(x,var=1E-2):
    ans_temp = 0.5*(x[0]**2+x[9]**2) -x[0]
    my_s = 0
    for i in range(1,9):
        my_s = np.copy(my_s)+0.5*(x[i]-x[i+1])**2
    ans = ans_temp + my_s + var*np.random.randn(1)
    return ans
   
adim =10
dim = 100

init_pt=10*np.ones(dim)

print(nesterov_2_f(init_pt))

ntrials = 10
maxit = 6000
f_avr = np.zeros(maxit+1)  #set equal to number of iterations + 1

for trial in range(ntrials):
    #sim setup
    test = Stars_sim(nesterov_2_f, init_pt, L1 = 4, var = 1E-4, verbose = False, maxit = maxit)
    test.STARS_only = True
    test.get_mu_star()
    test.get_h()
    # do 100 steps
    while test.iter < test.maxit:
        test.step()
    
    #update average of f
    f_avr += test.fhist  
    
f2_avr = np.zeros(maxit+1)

# Start the clock!
start = timeit.default_timer()

for trial in range(ntrials):
    #sim setup
    test = Stars_sim(nesterov_2_f, init_pt, L1 = 4, var = 1E-4, verbose = False, maxit = maxit)
    #test.STARS_only = True
    test.get_mu_star()
    test.get_h()
    # adapt every 10 timesteps using quadratic(after inital burn)
    test.train_method = 'GQ'
    test.adapt = 250 # Sets number of sub-cylcing steps
    
    # do 100 steps
    while test.iter < test.maxit:
        test.step()    
    f2_avr += test.fhist
    print('trial',trial,' minval',test.fhist[-1])

# Stop the clock!
stop = timeit.default_timer()

# Difference stop-start tells us run time
time= stop-start

print('the time of this experiment was:    ', time)


f_avr /= ntrials
f2_avr /= ntrials
 
plt.semilogy(np.absolute(-0.5*(1-1/(adim+1))-f_avr),label='Stars')
plt.semilogy(np.absolute(-0.5*(1-1/(adim+1))-f2_avr), label='Astars')
plt.legend()
plt.show()
