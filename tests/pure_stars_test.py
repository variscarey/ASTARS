#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:08:23 2020

@author: Varis Carey
"""

import numpy as np


import astars
print(dir(astars))
from astars.stars_sim import Stars_sim

def testfun(x):
    return x[0]**2+x[-1]**2+np.random.normal(scale=.01)


test = True
for trials in range(99):
    init_pt=np.random.rand(10,1)
    stars_test=Stars_sim(testfun,init_pt,L1=2.0,var=1E-2)
    stars_test.get_mu_star()
    stars_test.get_h()
    while stars_test.iter < stars_test.maxit:
        stars_test.STARS_step()
    #Error Bound for additive noise
    error_bound = (4*stars_test.L1*(stars_test.dim+4)/(101)*np.linalg.norm(init_pt)**2 
        +3*np.sqrt(2)/5*np.sqrt(stars_test.var)*(stars_test.dim+4))
    err=np.mean(stars_test.fhist)
    if err > error_bound:
        print('Unit test failed',err)
        test = False
   

    
