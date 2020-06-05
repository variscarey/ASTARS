#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 05:50:46 2020

@author: Varis Carey and Jordan Hall
"""


## ECNoise: (f, x_b, h, M, mult) -> (sigma_hat, xvals, fvals)

## Inputs:  f: noisy black-box function; 
##          x_b: base point, Px1 col vec;
##        h: discretization level which is 0.01 by default;
##          M: number of samples to use, M must be >3 and default is 7 (suggested in paper); 
##          mult: a boolean handle for whether we have add/mult noise.
##        (We default to additive noise, so we set mult=False. User can set mult=True if needed.)

## Outputs: sigma_hat: an estimator to the variance in noise of f;
##        xvals: the M x-values stored as an PxM array
##          fvals: the M function values stored as an Mx1 column vector

## Cost:    M evals of f

# We only need numpy for ECNoise
import numpy as np




def ECNoise(f,x_b,h=1E-2,M=3,mult_flag=False):
    '''
    ECNoise: (f, x_b, h, M, mult) -> (sigma_hat, xvals, fvals, L1

## Inputs:  f: noisy black-box function; 
##          x_b: base point, Px1 col vec;
##        h: discretization level which is 0.01 by default;
##          M: number of samples to use, M must be >3 and default is 7 (suggested in paper); 
##          mult: a boolean handle for whether we have add/mult noise.
##        (We default to additive noise, so we set mult=False. User can set mult=True if needed.)

## Outputs: sigma_hat: an estimator to the variance in noise of f;
##        xvals: the M x-values stored as an PxM array
##          fvals: the M function values stored as an Mx1 column vector

## Cost:    M evals of f

# We only need numpy for ECNoise
    '''

    
    
    # Start with sigma_hat at 0 (No Noise)
    sigma_hat = 0

    # Throw error for M too small
    if M < 1:
        return print('Please choose M>1.')
    x_b=x_b.flatten()
    #initialize points
    xvals = np.empty((x_b.size,2*M+1))
    fvals = np.empty(2*M+1)


    # Hard-coded random normalized direction to sample in
    p_u = np.random.randn(x_b.size)

    # Normalize
    p = p_u/np.linalg.norm(p_u)

    # Form and store the x and f values
    
    index = 0
    for i in np.arange(-M,M+1):
        intx = x_b + (i)*p*h
        xvals[:,index] = intx
        fvals[index] = f(intx)
        index += 1

    gamma = 1.0
    diff_col = np.zeros((2*M+1,2*M+1))
    diff_col[:,0] = fvals
    est= np.zeros(2*M)
    
    d_sign = np.zeros(2*M,dtype=np.bool_)
    # entry j will be 1 if there are sign changes, 0 if not
    
    # Form difference table
    for i in range(1,2*M+1):
        if i==1:
            diff_col[0:-i,i] = diff_col[1:,i-1] - diff_col[0:-i,i-1]
            if np.sum(np.abs(diff_col[0:-i,i]) < np.finfo(float).tiny)  > M:
                print(np.abs(diff_col[0:-i,i]) < np.finfo(float).tiny)
                print('h is too small. Try 100*h next. Default is h=0.01.')
                print('diff column',diff_col[:,i])
                sigma_hat = [-2]
                return sigma_hat
        else:
           diff_col[0:-i,i] = diff_col[1:-i+1,i-1] - diff_col[0:-i,i-1] 
        
        # Compute the estimates for the noise level
        gamma *= 0.5 * i / (2 * (i-1) + 1) 
        est[i-1] = np.sqrt(gamma * np.mean(diff_col[0:-i,i]**2))

        # Determine sign changes
        d_sign[i-1] = ((np.sum(diff_col[:,i] > 0)) and np.sum(diff_col[:,i] < 0) > 0)
    
    with (np.printoptions(precision = 4, suppress = True)):
        print(diff_col)
        print('noise estimates',est)


    # Determine noise level
    for k in range(0,2*M-2):
        dmin = np.min(est[k:k+3])
        dmax = np.max(est[k:k+3])
        if dmax <= 4*dmin and d_sign[k] == 1:
            sigma_hat = est[k]
            if mult_flag is False:
                noise_array = [sigma_hat, xvals, fvals]
                level = k+1
                print('Using difference level',k+1)
                break
                #return noise_array
            else:
                noise_array = [sigma_hat/fvals[0]**2, xvals, fvals] 
                break
                #return noise_array
        else:
            sigma_hat = 0

    # If we haven't found a sigma_hat, then h is too large (M and W)
    if sigma_hat == 0:
        # return print('h is too large. Try 0.01*h next. Default is h=0.01.')
        print(est)
        print(d_sign)
        sigma_hat = [-1]
        return sigma_hat
    
    #NEW, DETERMINE L1 after noise found
    f2d = fvals[0:-2]+fvals[2:]-2*fvals[1:-1]
    print('difference quotients',f2d)
    if np.max(np.abs(f2d)) > 100*noise_array[0]:
        L1 = np.max(np.abs(f2d))/h**2
    else:
        print('Warning: h may be too small for accurate L1')
        f2d=np.abs(fvals[0]+fvals[-1]-2*fvals[M])
        L1=np.max(f2d)/(M*h)**2
    print('L1=',L1)
    return noise_array,L1,level
    



def get_L1(xdata,fdata,var,degree = 3):
    ''' 
    computes approximate L1 Lipschitz constant by using 1D polynomial
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    from .reg_ls import ls_l2reg
    
    #scale data on (0,1)
    map=1/(xdata[-1]-xdata[0])
    
    #vandermone matrix with normalized coefficeints
    X=np.vander(xdata*map,N=degree+1,increasing=True)
    coeff=ls_l2reg(X,var,fdata)
    print('fit coefficients on [0,1]',coeff)
  
    poly=np.polynomial.Polynomial( coeff,domain=[xdata.min(),xdata.max()],window=[0,1])
    
    plt.plot(poly.linspace()[0],poly.linspace()[1])
    plt.scatter(xdata,fdata)
    plt.show()
    
    #TODO: put in non-grid search solve for L1.
    temp=(poly.deriv(2)).linspace()
    plt.plot(temp[0],temp[1])
    plt.show()
   
    return np.max(np.absolute(temp[1]))










