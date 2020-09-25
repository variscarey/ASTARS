#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:06:33 2020

@author: cmg
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

import active_subspaces as ss   
from astars.stars_sim import Stars_sim
from astars.utils.misc import subspace_dist, find_active

class toy2:
        def __init__(self, mag = 1.0, dim = 20, weights = None, sig = 1E-6):
                self.mag = mag
                self.dim = dim
                self.L1 = self.mag*self.dim*2.0
                self.sig = sig
                self.var = self.sig**2
                self.name = 'Toy 2'
                self.fstar = 0.0
                if weights is None:
                    self.weights = np.ones(self.dim)
                else:
                    self.weights = weights
                self.true_as = self.weights / np.linalg.norm(self.weights)
                
   
        def __call__(self, x):
            return self.mag*(np.dot(self.weights,x)**2) + self.sig*np.random.randn(1)


            
#sphere function, was toy 1
           
class sphere:
        def  __init__(self, mag = 1.0, dim = 20, adim = 1, sig = 1E-3):
              self.dim = dim
              self.adim = adim
              self.sig = sig
              self.mag = mag
        
              self.L1 = 2.0*self.mag
              self.var = self.sig**2
              self.name = 'Active Sphere'
              self.fstar = 0
              self.true_as = np.eye(self.dim,M=self.adim)
            
        def __call__(self,X):
            return np.sum(X[0:self.adim]**2) + self.sig*np.random.randn(1)
 
class nesterov_2_f:
    
    def __init__(self, dim = 50, adim = 5, sig = 1E-4):
        self.dim = dim
        self.adim = adim
        self.sig = sig
        
        self.L1 = 4.0
        self.var = self.sig**2
        self.name = 'Nesterov 2'
        self.fstar = .5*(-1 + 1 / (self.adim + 1))
        self.true_as = np.eye(self.dim,M=self.adim)
    
    def __call__(self,x):
        
        ans = 0.5*(x[0]**2 + x[self.adim-1]**2) - x[0]
        for i in range(self.adim-1):
            ans += .5*(x[i] - x[i+1])**2
            if ans.ndim == 1:
                ans += self.sig*np.random.randn(1)
            else:
                ans += self.sig*np.random.randn(ans.size)
        return ans
    
    
    
#def toy_f(x,var=1E-6):
#    return mag*(np.dot(weights,x))**2 + var*np.random.randn(1)
#true_as = weights / np.linalg.norm(weights)
#toy_l1 = mag*dim*2.0




toy2f = toy2()
sph = sphere()
nest = nesterov_2_f()


#for f in {nest}:
for f in {toy2f}:
    print('RUNNING PROBLEM ',f.name)
    dim = f.dim
    init_pt = 1000*np.random.randn(dim)
    ntrials = 10
    tr_stop = (dim+2)*(dim+1)//2
    maxit = 3*tr_stop
    #maxit = 1000
    f2_avr = np.zeros(maxit+1)
    f_avr = np.zeros(maxit+1)
    for trial in range(ntrials):
    #sim setup
        test = Stars_sim(f, init_pt, L1 = None, var = None, verbose = False, maxit = maxit)
        test.STARS_only = True
        print('Inital L1',test.L1)
        print('Inital var',test.var)
        test.update_L1 = True
        test.get_mu_star()
        test.get_h()
    # do training steps
        while test.iter < tr_stop:
            test.step()
    
    #update average of f and save for start of astars...?
        f_avr += test.fhist  
        learned_L1 = copy.deepcopy(test.L1)
        learned_var = copy.deepcopy(test.var)
        last_iterate = copy.deepcopy(test.x)
        f2_sto = np.zeros(maxit+1)
        f2_sto = copy.deepcopy(test.fhist)
        print(f2_sto)
    # do remaining steps
        while test.iter < test.maxit:
            test.step()
    

        f_avr += test.fhist         
 

    #sim setup for astars
        test2 = Stars_sim(f, last_iterate, L1 = learned_L1, var = learned_var, verbose = False, maxit = maxit-tr_stop)
        print('Inital L1',test2.L1)
        print('Inital var',test2.var)
        test2.get_mu_star()
        test2.get_h()
        test2.update_L1 = True
        # adapt every 10 timesteps using quadratic(after inital burn)
        test2.train_method = 'GQ'
        test2.adapt = 3*f.dim # Sets number of sub-cylcing steps
        #test.regul *= 100
        #test.debug = True
        test2.regul = None #test.sigma
        test2.threshold = .95
        test2.fhist = f2_sto
        test2.iter = tr_stop+1
        # do 100 steps
        while test2.iter < test2.maxit:
            test2.step()  
            if test2.iter % (2*f.dim) == 0 and test2.active is not None:
                print('Step',test2.iter,'Active dimension',test2.active.shape[1])
                print('Subspace Distance',subspace_dist(f.true_as,test2.active))
                print('Leading Direction',test2.active[:,0])
            # Normalization test
            #sub_sp = ss.subspaces.Subspaces()
            #train_x=np.hstack((test.xhist[:,0:test.iter+1],test.yhist[:,0:test.iter]))
            #train_f=np.hstack((test.fhist[0:test.iter+1],test.ghist[0:test.iter]))
            #sub_sp.compute(X=train_x.T,f=train_f,sstype='QPHD')
            #usual threshold
            #adim = find_active(sub_sp.eigenvals,sub_sp.eigenvecs)
            #print('Subspace Distance, no scaling, raw call',subspace_dist(true_as,sub_sp.eigenvecs[:,0:adim]))
        f2_avr += test2.fhist
        
        print(f2_avr)
        
        print('ASTARS trial',trial,' minval',test2.fhist[-1])
        print('Leading Active Variable',test2.active[:,0])
    
    

    f_avr /= ntrials
    f2_avr /= ntrials
 
    plt.semilogy(np.abs(f_avr-f.fstar),label='Stars')
    plt.semilogy(np.abs(f2_avr-f.fstar), label='Astars')
    plt.axvline(tr_stop)
    plt.title(f.name)
    plt.legend()
    plt.show()

def get_mat(X,f):
    #un-normalized computation
    gquad = ss.utils.response_surfaces.PolynomialApproximation(N=2)
    gquad.train(X, f, regul =None) #regul = self.var)
    # get regression coefficients
    b, A = gquad.g, gquad.H
    C = np.outer(b, b.transpose()) + 1.0/3.0*np.dot(A, A.transpose())
    sub_sp.eigenvals,sub_sp.eigenvecs = ss.subspaces.sorted_eigh(C)
    adim = find_active(sub_sp.eigenvals,sub_sp.eigenvecs)
    print('Subspace Distance, direct call with regul',subspace_dist(true_as,sub_sp.eigenvecs[:,0:adim]))
    #print(gquad.H[0,0])
    print('Active subspace, first vector',sub_sp.eigenvecs[:,0])
    
    #mapped math
    lb = np.amin(X,axis=0).reshape(-1,1)
    ub = np.amax(X,axis=0).reshape(-1,1)
    #print(lb.shape)
    nrm_data=ss.utils.misc.BoundedNormalizer(lb,ub)

    #print(lb,ub)
    xhat=nrm_data.normalize(X)
    
    scale = .5*(ub-lb)
    print('scaling vector', scale)
    gquad.train(xhat,f, regul = None)
    b2, A2 = gquad.g, gquad.H
    C2 = np.outer(b2,b2.transpose()) + 1.0/3.0*np.dot(A2,A2.transpose())
    D = np.diag(1.0/scale.flatten())
    #print(D)
    C2 = D @ C2 @ D
    
    sub_sp.eigenvals,sub_sp.eigenvecs = ss.subspaces.sorted_eigh(C2)
    adim = find_active(sub_sp.eigenvals,sub_sp.eigenvecs)
    print('Subspace Distance, after mapping',subspace_dist(true_as,sub_sp.eigenvecs[:,0:adim]))
    #print(gquad.H[0,0])
    print('Active subspace, first vector',sub_sp.eigenvecs[:,0])
    
#get_mat(train_x.T,train_f)
