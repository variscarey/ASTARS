#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 6/12/2020

@author: Jordan Hall and Varis Carey
"""

import numpy as np
import matplotlib.pyplot as plt  
from astars.stars_sim import Stars_sim  

# Wing function (and gradient) from Paul Constantine's Active Subspace test function library.

def wing(xx,var=1E-2):
    #each row of xx should be [Sw. Wfw, A, Lambda, q, lambda, tc, Nz, Wdg, Wp] in the (normalized) input space
    #removing ability for multiple evaluations and now just accepting a single vector input and returning a single value as output
    
    x = xx.copy()
    
    Sw = x[0]; Wfw = x[1]; A = x[2]; L = x[3]*np.pi/180.; q = x[4]
    l = x[5]; tc = x[6]; Nz = x[7]; Wdg = x[8]; Wp = x[9]
    
    return .036*Sw**.758*Wfw**.0035*A**.6*np.cos(L)**-.9*q**.006*l**.04*100**-.3*tc**-.3*Nz**.49*Wdg**.49 + Sw*Wp + var*np.random.randn(1)
    
def wing_grad(xx):
    #each row of xx should be [Sw. Wfw, A, Lambda, q, lambda, tc, Nz, Wdg, Wp] in the normalized input space
    #returns matrix whose ith row is gradient of wing function at ith row of inputs
    
    x = xx.copy()
    x = np.atleast_2d(x)
    
    Sw = x[:,0]; Wfw = x[:,1]; A = x[:,2]; L = x[:,3]*np.pi/180.; q = x[:,4]
    l = x[:,5]; tc = x[:,6]; Nz = x[:,7]; Wdg = x[:,8]; Wp = x[:,9]
    
    Q = .036*Sw**.758*Wfw**.0035*A**.6*np.cos(L)**-.9*q**.006*l**.04*100**-.3*tc**-.3*Nz**.49*Wdg**.49 #Convenience variable
    
    dfdSw = (.758*Q/Sw + Wp)[:,None]
    dfdWfw = (.0035*Q/Wfw)[:,None]
    dfdA = (.6*Q/A)[:,None]
    dfdL = (.9*Q*np.sin(L)/np.cos(L))[:,None]
    dfdq = (.006*Q/q)[:,None]
    dfdl = (.04*Q/l)[:,None]
    dfdtc = (-.3*Q/tc)[:,None]
    dfdNz = (.49*Q/Nz)[:,None]
    dfdWdg = (.49*Q/Wdg)[:,None]
    dfdWp = (Sw)[:,None]
        
    return np.hstack((dfdSw, dfdWfw, dfdA, dfdL, dfdq, dfdl, dfdtc, dfdNz, dfdWdg, dfdWp))
    
# Do STARS and ASTARS  
init_pt=5*np.random.randn(10,1)
ntrials = 2
maxit = 200
f_avr = np.zeros(maxit+1)  #set equal to number of iterations + 1

for trial in range(ntrials):
    #sim setup
    test = Stars_sim(wing, init_pt, L1 = 2, var = 1E-4, verbose = False, maxit = maxit)
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
    test = Stars_sim(wing, init_pt, L1 = 2, var = 1E-4, verbose = False, maxit = maxit)
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
