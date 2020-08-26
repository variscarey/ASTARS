#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:08:23 2020

@author: Varis Carey
"""

import numpy as np
import matplotlib.pyplot as plt

import active_subspaces as acs
from astars.stars_sim import Stars_sim
from astars.utils.surrogates import train_rbf


#N-D function with two active variables
def testfun(x):
    return x[0]**2+x[-1]**2+np.random.normal(scale=.01)

ss = acs.subspaces.Subspaces()
#test rbf on random normal samples
lin_samp = np.random.normal(size=(15,10))
f_data = testfun(lin_samp.T)
ss,lin_surrog=train_rbf(lin_samp,f_data,noise=1E-4)



print('Active subspace from linear rbf',ss.eigenvecs)

quad_samp = np.random.normal(size=(90,10))
fq_data = testfun(quad_samp.T)
ss2,quad_surrog = train_rbf(quad_samp,fq_data,noise=1E-4)


init_pt=np.random.rand(10)
stars_test=Stars_sim(testfun,init_pt,L1=2.0,var=1E-2,maxit=80)
stars_test.get_mu_star()
stars_test.get_h()
while stars_test.iter < stars_test.maxit:
    stars_test.step()
stars_test.compute_active()
print('Active Variables after STARS run',stars_test.active)
print('Active Weights',stars_test.wts)

plt.semilogy(stars_test.fhist)
plt.figure()
plt.plot(stars_test.xhist[0,:])
plt.plot(stars_test.xhist[-1,:])
   

    
