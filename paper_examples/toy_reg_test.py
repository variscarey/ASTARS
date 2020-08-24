#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:06:33 2020

@author: cmg
"""

import numpy as np
import matplotlib.pyplot as plt

import active_subspaces as ss  
from astars.stars_sim import Stars_sim
from astars.utils.misc import subspace_dist,find_active

sigma = 1E-1

def toy_f(x,sig=sigma):
    return x[0]**2 + sig*np.random.randn(1)
    
dim = 30
init_pt= 100*np.random.rand(dim,1)
ntrials = 1
maxit = 800
f_avr = np.zeros(maxit+1)  #set equal to number of iterations + 1
true_as = np.zeros((dim,1))
true_as[0,0]=1.0
reg_weight= 100

reg_param = sigma

sub_sp = ss.subspaces.Subspaces()

#maxit random points as training data
train_set = np.random.randn(dim,maxit)

f_data = toy_f(train_set)
print('data size', f_data.shape)
#don't normalize 
sub_sp.compute(X=train_set.T,f=f_data,sstype='QPHD')
#usual threshold
adim = find_active(sub_sp.eigenvals,sub_sp.eigenvecs)
print('Subspace Distance',subspace_dist(true_as,sub_sp.eigenvecs[:,0:adim]))


for trial in range(ntrials):
    #sim setup
    test = Stars_sim(toy_f, init_pt, L1 = 2.0, var = sigma**2, verbose = False, maxit = maxit)
    test.STARS_only = True
    test.get_mu_star()
    test.get_h()
    # do 100 steps
    while test.iter < test.maxit:
        test.step()
        
    train_x = np.hstack((test.xhist[:,0:test.iter+1],test.yhist[:,0:test.iter]))
    train_f = np.hstack((test.fhist[0:test.iter+1],test.ghist[0:test.iter]))
    train_x = train_x.T
    sub_sp.compute(X=train_x,f=train_f,sstype='QPHD')
    adim = find_active(sub_sp.eigenvals,sub_sp.eigenvecs)
    print('Subspace Distance, no regul',subspace_dist(true_as,sub_sp.eigenvecs[:,0:adim]))
    gquad = ss.utils.response_surfaces.PolynomialApproximation(N=2)
    gquad.train(train_x, train_f, regul = None) #regul = self.var)
    # get regression coefficients
    b, A = gquad.g, gquad.H
    C = np.outer(b, b.transpose()) + 1.0/3.0*np.dot(A, A.transpose())
    sub_sp.eigenvals,sub_sp.eigenvecs = ss.subspaces.sorted_eigh(C)
    adim = find_active(sub_sp.eigenvals,sub_sp.eigenvecs)
    print('Subspace Distance, direct call with no regul',subspace_dist(true_as,sub_sp.eigenvecs[:,0:adim]))
    print('Active subspace, first vector',sub_sp.eigenvecs[:,0])
    print(gquad.H[0,0])
    
    
    sub_sp.compute(X=train_x,f=train_f,sstype='QPHD',regul = reg_weight*reg_param)
    adim = find_active(sub_sp.eigenvals,sub_sp.eigenvecs)
    print('Subspace Distance, regul',subspace_dist(true_as,sub_sp.eigenvecs[:,0:adim]))
    #update average of f
    f_avr += test.fhist  
    
    
#direct call for debugging
    gquad = ss.utils.response_surfaces.PolynomialApproximation(N=2)
    gquad.train(train_x, train_f, regul =reg_weight*reg_param) #regul = self.var)
    # get regression coefficients
    b, A = gquad.g, gquad.H
    C = np.outer(b, b.transpose()) + 1.0/3.0*np.dot(A, A.transpose())
    sub_sp.eigenvals,sub_sp.eigenvecs = ss.subspaces.sorted_eigh(C)
    adim = find_active(sub_sp.eigenvals,sub_sp.eigenvecs)
    print('Subspace Distance, direct call with regul',subspace_dist(true_as,sub_sp.eigenvecs[:,0:adim]))
    print(gquad.H[0,0])
    print('Active subspace, first vector',sub_sp.eigenvecs[:,0])
    
f2_avr = np.zeros(maxit+1)
plt.semilogy(test.fhist)



