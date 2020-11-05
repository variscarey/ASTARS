#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:06:33 2020

@author: cmg
"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import timeit
import active_subspaces as ss   
from astars.stars_sim import Stars_sim
from astars.utils.misc import subspace_dist, find_active
import pandas as pd

###############################################################################
############ Set user-desired file path for storing output data!!! ############
###############################################################################
user_file_path = '.'
###############################################################################
###############################################################################
###############################################################################

class toy2:
        def __init__(self, mag = 1.0, dim = 20, weights = None, sig = 1E-3):
                self.mag = mag
                self.dim = dim
                self.L1 = self.mag*self.dim*2.0
                self.sig = sig
                self.var = self.sig**2
                self.nickname = 'Toy_2'
                self.name = 'Example 1: STARS and FAASTARS Convergence with $L_1$ and $\sigma^2$ Learning'
                self.fstar = 0.0
                if weights is None:
                    self.weights = np.ones(self.dim)
                else:
                    self.weights = weights
                self.active = self.weights / np.linalg.norm(self.weights)
                
                self.maxit = 2*dim**2
                self.threshold = 0.95
                
   
        def __call__(self, x):
            return self.mag*(np.dot(self.weights,x)**2) + self.sig*np.random.randn(1)


            
#sphere function, was toy 1
           
class sphere:
        def  __init__(self, mag = 1.0, dim = 20, adim = 10, sig = 1E-3):
              self.dim = dim
              self.adim = adim
              self.sig = sig
              self.mag = mag
        
              self.L1 = 2.0*self.mag
              self.var = self.sig**2
              self.nickname = 'Active_Sphere'
              self.name = 'Example 2: STARS and FAASTARS Convergence with $L_1$ and $\sigma^2$ Learning'
              self.fstar = 0
              self.active = np.eye(self.dim,M=self.adim)
              
              self.maxit = 4*dim**2
              self.threshold = 0.9995
            
        def __call__(self,X):
            return self.mag*np.sum(X[0:self.adim]**2) + self.sig*np.random.randn(1)
 
class nesterov_2_f:
    
    def __init__(self, dim = 50, adim = 5, sig = 1E-4):
        self.dim = dim
        self.adim = adim
        self.sig = sig
        
        self.L1 = 4.0
        self.var = self.sig**2
        self.name = 'Nesterov 2'
        self.fstar = .5*(-1 + 1 / (self.adim + 1))
        self.active = np.eye(self.dim,M=self.adim)
    
    def __call__(self,x):
        
        ans = 0.5*(x[0]**2 + x[self.adim-1]**2) - x[0]
        for i in range(self.adim-1):
            ans += .5*(x[i] - x[i+1])**2
            if ans.ndim == 1:
                ans += self.sig*np.random.randn(1)
            else:
                ans += self.sig*np.random.randn(ans.size)
        return ans
    

params = {'legend.fontsize': 28,'legend.handlelength': 3}
plt.rcParams["figure.figsize"] = (60,40)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['font.size'] = 30
plt.rcParams['figure.titlesize'] = 'xx-large'
plt.rcParams.update(params)



toy2f = toy2()
sph = sphere()
nest = nesterov_2_f() # To be worked on for extensions project/paper; unstable for now

# Start the clock!
start = timeit.default_timer()


for f in {toy2f,sph}:
    print('RUNNING PROBLEM ',f.name)
    dim = f.dim
    np.random.seed(9)
    init_pt = 10*np.random.randn(dim) #  works for toy2 and active sphere
    ntrials = 10 #250
    tr_stop = (dim+2)*(dim+1)//2
    maxit = f.maxit

    f2_avr = np.zeros(maxit+1)
    f_avr = np.zeros(maxit+1)
    
    #initialize storage for data dump
    STARS_f_sto = np.zeros((maxit+1,ntrials))
    STARS_x_sto = np.zeros((1,dim))
    STARS_L1_sto = np.zeros((maxit+1,ntrials))
    STARS_var_sto = np.zeros(ntrials)
    
    FAASTARS_f_sto = np.zeros((maxit+1,ntrials))
    FAASTARS_x_sto = np.zeros((1,dim))
    FAASTARS_L1_sto = np.zeros((maxit+1,ntrials))
    FAASTARS_var_sto = np.zeros(ntrials)

    
    
    
    for trial in range(ntrials):
    #sim setup
        test = Stars_sim(f, init_pt, L1 = None, var = None, verbose = False, maxit = maxit, true_as = f.active, train_method = 'GQ')
        test.STARS_only = True
        print('Inital L1',test.L1)
        print('Inital var',test.var)
        test.update_L1 = True
        test.get_mu_star()
        test.get_h()
    # do training steps
        while test.iter < test.tr_stop:
            test.step()
    
    #update average of f and save for start of astars...?
        
        test2 = copy.deepcopy(test)
        test2.STARS_only = False
        #test2.train_method = 'GQ'
        
        test2.adapt = f.dim # Sets number of sub-cylcing steps -- works well for toy2 and sphere
        #test.debug = True
        test2.regul = test.sigma**2 # works well for toy2 and sphere
        test2.threshold = f.threshold
        
        while test.iter < test.maxit:
            test.step()
            test2.step()
    

        f_avr += test.fhist         
        f2_avr += test2.fhist
        
        if maxit > test2.tr_stop:
            FAASTARS_adim_sto = np.zeros((maxit-test2.tr_stop-1,ntrials))
            FAASTARS_sub_dist_sto = np.zeros((maxit-test2.tr_stop-1,ntrials))        
        
        # data dump
        STARS_f_sto = np.hstack((STARS_f_sto, np.transpose([test.fhist])))
        STARS_x_sto = np.vstack((STARS_x_sto,np.transpose(test.xhist)))
        STARS_L1_sto[:,trial] = test.L1_hist
        STARS_var_sto[trial] = test.var
    
        FAASTARS_f_sto = np.hstack((FAASTARS_f_sto, np.transpose([test2.fhist])))
        FAASTARS_x_sto = np.vstack((FAASTARS_x_sto,np.transpose(test2.xhist)))
        FAASTARS_L1_sto[:,trial] = test2.L1_hist
        FAASTARS_var_sto[trial] = test2.var
        
        if maxit > test2.tr_stop:
            FAASTARS_adim_sto[:,trial] = test2.adim_hist
            FAASTARS_sub_dist_sto[:,trial] = test2.sub_dist_hist
            
            
            
    # Reads out key data into individual csv files via Pandas. (Need documentation for formatting...)
    pd.DataFrame(STARS_f_sto[:,1:np.shape(STARS_f_sto)[1]]).to_csv(user_file_path + 'STARS_f_sto_' + f.nickname + '.csv', header=None, index=None, sep='\t')
    pd.DataFrame(STARS_x_sto[1:np.shape(STARS_x_sto)[0],:]).to_csv(user_file_path + 'STARS_x_sto_' + f.nickname +  '.csv', header=None, index=None, sep='\t')
    pd.DataFrame(STARS_L1_sto[:,1:np.shape(STARS_L1_sto)[1]]).to_csv(user_file_path + 'STARS_L1_sto_'  + f.nickname + '.csv', header=None, index=None, sep='\t')
    pd.DataFrame(STARS_var_sto).to_csv(user_file_path + 'STARS_var_sto_'  + f.nickname + '.csv', header=None, index=None, sep='\t')
    
    pd.DataFrame(FAASTARS_f_sto[:,1:np.shape(FAASTARS_f_sto)[1]]).to_csv(user_file_path + 'FAASTARS_f_sto_'  + f.nickname + '.csv', header=None, index=None, sep='\t')
    pd.DataFrame(FAASTARS_x_sto[1:np.shape(FAASTARS_x_sto)[0],:]).to_csv(user_file_path + 'FAASTARS_x_sto_'  + f.nickname + '.csv', header=None, index=None, sep='\t')
    pd.DataFrame(FAASTARS_L1_sto[:,1:np.shape(FAASTARS_L1_sto)[1]]).to_csv(user_file_path + 'FAASTARS_L1_sto_'  + f.nickname + '.csv', header=None, index=None, sep='\t')
    pd.DataFrame(FAASTARS_var_sto).to_csv(user_file_path + 'FAASTARS_var_sto_'  + f.nickname + '.csv', header=None, index=None, sep='\t')
    
    if maxit > test2.tr_stop:
        pd.DataFrame(FAASTARS_adim_sto[:,1:np.shape(FAASTARS_adim_sto)[1]]).to_csv(user_file_path + 'FAASTARS_adim_sto_'  + f.nickname + '.csv', header=None, index=None, sep='\t')  
        pd.DataFrame(FAASTARS_sub_dist_sto[:,1:np.shape(FAASTARS_sub_dist_sto)[1]]).to_csv(user_file_path + 'FAASTARS_sub_dist_sto_'  + f.nickname + '.csv', header=None, index=None, sep='\t')  

            
           
        print('FAASTARS trial',trial,' minval',test.fhist[-1])
   
        
        print(f2_avr)
        print('STARS trial',trial,' minval',test.fhist[-1])
        print('ASTARS trial',trial,' minval',test2.fhist[-1])
        print('Leading Active Variable',test2.active[:,0])
    
    
    # Stop the clock!
    stop = timeit.default_timer()

    # Difference stop-start tells us run time
    time = stop - start
    print('the time of this experiment was:    ', time/3600, 'hours')
    

    f_avr /= ntrials
    f2_avr /= ntrials
 
    plt.semilogy(np.abs(f_avr-f.fstar),label='STARS', color = 'red', lw = 5)
    plt.semilogy(np.abs(f2_avr-f.fstar), label='FAASTARS', color = 'black', lw = 5, ls = '--')
    plt.axvline(tr_stop)
    plt.title(f.name)
    plt.xlabel('$k$, iteration count')
    plt.ylabel('$|f(\lambda^{(k)})-f^*|$')    
    plt.legend()
    plt.show()

