#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:06:33 2020

@author: cmg
"""

import numpy as np
import matplotlib.pyplot as plt
import timeit
import active_subspaces as ss   
from astars.stars_sim import Stars_sim
from astars.utils.misc import subspace_dist, find_active
import pandas as pd

###############################################################################
############ Set user-desired file path for storing output data!!! ############
###############################################################################
user_file_path = '/home/ccm/Desktop/'
###############################################################################
###############################################################################
###############################################################################

class toy2:
        def __init__(self, dim = 15, sig = 1E-3):

                self.dim = dim
                self.data_dim = 3
                self.L1 = 4.0 #2.0*self.data_dim # This is the L1 for the linear scalarization, not f
                self.sig = sig
                self.var = (self.data_dim**2)*self.sig**2 # This is the noise for the linear scalarization, not f
                self.name = 'Example 1: STARS, FAASTARS, and ASTARS Convergence'
                self.nickname = 'toy_2'
                self.fstar = 0.5

                self.active = np.eye(dim,5)
                
                self.maxit = 1000
                self.ntrials = 100
                self.adapt = dim
                self.regul = self.var
                self.threshold = 0.999
                self.initscl = 10.0
            
   
        def __call__(self, x):
            return np.array([x[0]**2 + self.sig*np.random.randn(1), (x[0]-1)**2 + self.sig*np.random.randn(1), np.sum(x[1:5]**2) + self.sig*np.random.randn(1)])



    
    


#plotting parameters and definitions
toy2f = toy2()

print(np.shape(toy2f(np.random.randn(2)))[0])


params = {'legend.fontsize': 28,'legend.handlelength': 3}
plt.rcParams["figure.figsize"] = (60,40)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['font.size'] = 30
plt.rcParams['figure.titlesize'] = 'xx-large'
plt.rcParams.update(params)

stars_full, sf_ls = 'red', '--'
active_stars_learned, lr_ls = 'black', '-.'
active_stars_ref, rf_ls = 'blue', ':'


# Start the clock!
start = timeit.default_timer()

for f in {toy2f}:

    dim = f.dim
    np.random.seed(9)
    init_pt = f.initscl*np.random.randn(dim)
    ntrials = f.ntrials
    maxit = f.maxit

    f3_avr = np.zeros(maxit+1)
    f2_avr = np.zeros(maxit+1)
    f_avr = np.zeros(maxit+1)
    
    #initialize storage for data dump
    STARS_f_sto = np.zeros((maxit+1,1))
    STARS_x_sto = np.zeros((1,dim))
    ASTARS_f_sto = np.zeros((maxit+1,1))
    ASTARS_x_sto = np.zeros((1,dim))
    FAASTARS_f_sto = np.zeros((maxit+1,1))
    FAASTARS_x_sto = np.zeros((1,dim))    
    


    data_dim = np.shape(toy2f(init_pt))[0]
    print(data_dim)
    if data_dim > 1:
        def lin_scalar(x):
            s=0
            for i in range(data_dim):
                s += f(x)[i]
            return s
    print(lin_scalar(init_pt))      
    
    # STARS
    for trial in range(ntrials):
    #sim setup
        test = Stars_sim(lin_scalar, init_pt, L1 = f.L1, var = f.var, verbose = False, maxit = maxit)
        test.STARS_only = True
        test.get_mu_star()
        test.get_h()
    # do 100 steps
        while test.iter < test.maxit:
            test.step()
    
    #update average of f
        f_avr += test.fhist
        
        # data dump
        STARS_f_sto = np.hstack((STARS_f_sto, np.transpose([test.fhist])))
        STARS_x_sto = np.vstack((STARS_x_sto,np.transpose(test.xhist)))
        
        print('STARS trial',trial,' minval',test.fhist[-1], 'minimizer', test.x, 'linear scalarization objective value', lin_scalar(test.x))

    # FAASTARS
    for trial in range(ntrials):
        #sim setup
        test = Stars_sim(lin_scalar, init_pt, L1 = f.L1, var = f.var, verbose = False, maxit = maxit, true_as = f.active)
        test.get_mu_star()
        test.get_h()
        # adapt every f.adapt timesteps using quadratic(after inital burn)
        test.train_method = 'GQ'
        test.adapt = f.adapt # Sets number of sub-cylcing steps

        #test.debug = True
        test.regul = f.regul
        test.threshold = f.threshold
        
        # do 100 steps
        while test.iter < test.maxit:
            test.step()

        f2_avr += test.fhist
        
        if maxit > test.tr_stop:
            FAASTARS_adim_sto = np.zeros((maxit-test.tr_stop-1,1))
            FAASTARS_sub_dist_sto = np.zeros((maxit-test.tr_stop-1,1))
        
        
        # data dump
        FAASTARS_f_sto = np.hstack((FAASTARS_f_sto, np.transpose([test.fhist])))
        FAASTARS_x_sto = np.vstack((FAASTARS_x_sto,np.transpose(test.xhist)))
        if maxit > test.tr_stop:
            FAASTARS_adim_sto = np.hstack((FAASTARS_adim_sto, np.transpose([test.adim_hist])))
            FAASTARS_sub_dist_sto = np.hstack((FAASTARS_sub_dist_sto, np.transpose([test.sub_dist_hist])))
        print('FAASTARS trial',trial,' minval',test.fhist[-1], 'minimizer', test.x, 'linear scalarization objective value', lin_scalar(test.x))
    
    # ASTARS
    for trial in range(ntrials):
        
        test = Stars_sim(lin_scalar, init_pt, L1 = f.L1, var = f.var, verbose = False, maxit = maxit)
        test.active = f.active
        test.get_mu_star()
        test.get_h()
        test.adapt = 0
    # do 100 steps
        while test.iter < test.maxit:
            test.step()
    
    #update average of f
        f3_avr += test.fhist  
        
        # data dump
        ASTARS_f_sto = np.hstack((ASTARS_f_sto, np.transpose([test.fhist])))
        ASTARS_x_sto = np.vstack((ASTARS_x_sto,np.transpose(test.xhist)))
             
        
        print('True ASTARS trial',trial,' minval',test.fhist[-1], 'minimizer', test.x, 'linear scalarization objective value', lin_scalar(test.x))
        
    f_avr /= ntrials
    f2_avr /= ntrials
    f3_avr /= ntrials
    

    # Reads out key data into individual csv files via Pandas. (Need documentation for formatting...)
    pd.DataFrame(STARS_f_sto[:,1:np.shape(STARS_f_sto)[1]]).to_csv(user_file_path + 'STARS_f_sto_' + f.nickname + '.csv', header=None, index=None, sep='\t')
    pd.DataFrame(STARS_x_sto[1:np.shape(STARS_x_sto)[0],:]).to_csv(user_file_path + 'STARS_x_sto_' + f.nickname +  '.csv', header=None, index=None, sep='\t')
    pd.DataFrame(ASTARS_f_sto[:,1:np.shape(ASTARS_f_sto)[1]]).to_csv(user_file_path + 'ASTARS_f_sto_' + f.nickname + '.csv', header=None, index=None, sep='\t')
    pd.DataFrame(ASTARS_x_sto[1:np.shape(ASTARS_x_sto)[0],:]).to_csv(user_file_path + 'ASTARS_x_sto_'  + f.nickname + '.csv', header=None, index=None, sep='\t')
    pd.DataFrame(FAASTARS_f_sto[:,1:np.shape(FAASTARS_f_sto)[1]]).to_csv(user_file_path + 'FAASTARS_f_sto_'  + f.nickname + '.csv', header=None, index=None, sep='\t')
    pd.DataFrame(FAASTARS_x_sto[1:np.shape(FAASTARS_x_sto)[0],:]).to_csv(user_file_path + 'FAASTARS_x_sto_'  + f.nickname + '.csv', header=None, index=None, sep='\t')
    if maxit > test.tr_stop:
        pd.DataFrame(FAASTARS_adim_sto[:,1:np.shape(FAASTARS_adim_sto)[1]]).to_csv(user_file_path + 'FAASTARS_adim_sto_'  + f.nickname + '.csv', header=None, index=None, sep='\t')  
        pd.DataFrame(FAASTARS_sub_dist_sto[:,1:np.shape(FAASTARS_sub_dist_sto)[1]]).to_csv(user_file_path + 'FAASTARS_sub_dist_sto_'  + f.nickname + '.csv', header=None, index=None, sep='\t')      
        
    # Stop the clock!
    stop = timeit.default_timer()

    # Difference stop-start tells us run time
    time = stop - start
    print('the time of this experiment was:    ', time/3600, 'hours')
 
    plt.semilogy(np.abs(f_avr-f.fstar),lw = 5,label='STARS',color=stars_full, ls=sf_ls)
    plt.semilogy(np.abs(f2_avr-f.fstar), lw = 5, label='FAASTARS (Approx $\\tilde{\mathcal{A}}$)',color=active_stars_learned ,ls=lr_ls)
    plt.semilogy(np.abs(f3_avr-f.fstar), lw = 5,label = 'ASTARS (True $\mathcal{A}$)',color=active_stars_ref ,ls=rf_ls)
    plt.title(f.name)
    plt.xlabel('$k$, iteration count')
    plt.ylabel('$|f(\lambda^{(k)})-f^*|$')
    plt.legend()
    plt.show()

