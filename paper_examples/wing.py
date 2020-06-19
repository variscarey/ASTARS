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
    
    x1 = xx.copy()
    
    ub = np.array([150, 220, 6, -10, 16, .5, .08, 2.5, 1700, .025])
    lb = np.array([200, 300, 10, 10, 45, 1, .18, 6, 2500, .08])
    
    x = lb + (ub - lb)/2.*(x1 + 1)
    #print('mapped x',x)
    
    Sw = x[0]; Wfw = x[1]; A = x[2]; L = x[3]*np.pi/180.; q = x[4]
    l = x[5]; tc = x[6]; Nz = x[7]; Wdg = x[8]; Wp = x[9]
    
    return .036*Sw**.758*Wfw**.0035*A**.6*np.cos(L)**-.9*q**.006*l**.04*100**-.3*tc**-.3*Nz**.49*Wdg**.49 + Sw*Wp + var*np.random.randn(1)

def wing_barrier(xx, mu=1E-1):
    return wing(xx) - mu * np.sum(np.log(-xx+1)+np.log(1+xx))
    
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
init_pt=2*np.random.rand(10)-1
#init_pt[0]*=50
#init_pt[0]+=150
#init_pt[1]*=80
#init_pt[1]+=220
#init_pt[2]*=4
#init_pt[2]+=6
#init_pt[3]*=20
#init_pt[4]*=29
#init_pt[5]*=.5
#init_pt[6]*=.1
#init_pt[7]*=3.5
#init_pt[8]*=800
#init_pt[9]*=.055
#init_pt[3]-=10
#init_pt[4]+=16
#init_pt[5]+=.5
#init_pt[6]+=.08
#init_pt[7]+=2.5
#init_pt[8]+=1700
#init_pt[9]+=.025

print(init_pt)


ntrials = 2
maxit = 200

f_avr = np.zeros(maxit+1)  #set equal to number of iterations + 1

for trial in range(ntrials):
    #sim setup
    test = Stars_sim(wing_barrier, init_pt, L1 = 200, var = 1E-4, verbose = True, maxit = maxit)
    test.STARS_only = True
    test.debug = True
    test.get_mu_star()
    test.get_h()
    # do 100 steps
    while test.iter < test.maxit:
        test.step()
        #if np.isnan(test.x).any:
        #    print(test.xhist[:,0:test.iter+1],test.yhist[:,0:test.iter+1],test.fhist[0:test.iter+1],test.ghist[0:test.iter+1])
        #    print(test.x)
        #    raise SystemExit('nan in current iterate')
    
    #update average of f
    f_avr += test.fhist  
    
f2_avr = np.zeros(maxit+1)

for trial in range(ntrials):
    #sim setup
    test2 = Stars_sim(wing_barrier, init_pt, L1 = 200, var = 1E-4, verbose = False, maxit = maxit)
    #test.STARS_only = True
    test2.get_mu_star()
    test2.get_h()
    test2.debug = True
    # adapt every 10 timesteps using quadratic(after inital burn)
    test2.train_method = 'GQ'
    test2.adapt = 10 # Sets number of sub-cylcing steps
    
    
    # do 100 steps
    while test2.iter < test.maxit:
        test2.step()    
    f2_avr += test2.fhist
    print('trial',trial,' minval',test2.fhist[-1])

f_avr /= ntrials
f2_avr /= ntrials
 
plt.semilogy(f_avr,label='Stars')
plt.semilogy(f2_avr, label='Astars')
plt.legend()
plt.show()
