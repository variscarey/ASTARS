#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6/10/2020

@author: Jordan Hall and Varis Carey
"""

import numpy as np
import matplotlib.pyplot as plt  
from astars.stars_sim import Stars_sim

def nesterov_2_f(x,sig=1E-2):
    ans = 0.5*(x[0]**2+x[9]**2) -x[0]
    for i in range(9):
        ans += 0.5*(x[i]-x[i+1])**2
    ans += sig*np.random.randn(1)
    return ans
   

init_pt=np.zeros(100)

print(nesterov_2_f(init_pt))

ntrials = 2
maxit = 16000
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

for trial in range(ntrials):
    #sim setup
    test = Stars_sim(nesterov_2_f, init_pt, L1 = 4, var = 1E-4, verbose = False, maxit = maxit)
    #test.STARS_only = True
    test.get_mu_star()
    test.get_h() 
    # adapt every 10 timesteps using quadratic(after inital burn)
    test.train_method = 'GQ'
    test.adapt = 500 #ts number of sub-cylcing steps
    
    # do 100 steps
    while test.iter < test.maxit:
        test.step()    
        if test.iter % 100 == 0:
            print('iter',test.iter,test.fhist[test.iter])
    f2_avr += test.fhist
    print('trial',trial,' minval',test.fhist[-1])

f_avr /= ntrials
f2_avr /= ntrials

fstar = .5*(-1.0 +  1.0 / 11.0)
 
plt.semilogy(np.absolute(fstar-f_avr),label='Stars')
plt.semilogy(np.absolute(fstar-f2_avr), label='Astars')
plt.legend()
plt.show()
