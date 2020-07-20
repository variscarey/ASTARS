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


adim =10
dim = 50


def nesterov_2_f(x,sig=1E-3):
    ans = 0.5*(x[0]**2 + x[adim-1]**2) - x[0]
    for i in range(adim-1):
        ans += 0.5*(x[i] - x[i+1])**2
    ans += sig*np.random.randn(1)
    return ans

init_pt = np.random.randn(dim)

print(nesterov_2_f(init_pt))

ntrials = 1
maxit = 2500

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
        if test.iter % 100 == 0:
            print('iter',test.iter,test.fhist[test.iter])
    
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
<<<<<<< HEAD
    test.get_h() 
    # adapt every 10 timesteps using quadratic(after inital burn)
=======
    test.get_h()
    # adapt every time.adapt timesteps using quadratic(after inital burn)
>>>>>>> 8ac7077519825adbb19c0c1b8a81bfda1c5348f3
    test.train_method = 'GQ'
    test.adapt = 500 #ts number of sub-cylcing steps

    
    # do 100 steps
    while test.iter < test.maxit:
        test.step()    
        if test.iter % 100 == 0:
            print('iter',test.iter,test.fhist[test.iter])
    f2_avr += test.fhist
    print('trial',trial,' minval',test.fhist[-1])

# Stop the clock!
stop = timeit.default_timer()

# Difference stop-start tells us run time
time = stop - start

print('the time of this experiment was:    ', time/3600, 'hours')


f_avr /= ntrials
f2_avr /= ntrials

<<<<<<< HEAD
fstar = .5*(-1.0 +  1.0 / 11.0)
 

plt.semilogy(np.absolute(fstar-f_avr),label='Stars')
plt.semilogy(np.absolute(fstar-f2_avr), label='Astars')

=======
fstar = .5*(-1.0 + 1.0 / (adim + 1))
 
plt.semilogy(np.absolute(fstar-f_avr),label='Stars')
plt.semilogy(np.absolute(fstar-f2_avr), label='Astars')
>>>>>>> 8ac7077519825adbb19c0c1b8a81bfda1c5348f3
plt.legend()
plt.show()
