# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 10:20:16 2020

@author: varis
"""


import numpy as np
def subspace_dist(obasis,obasis2):
    if obasis.ndim == 1:
        obasis=obasis.reshape(-1,1)
    if obasis2.ndim == 1:
        obasis2=obasis2.reshape(-1,1)
    d=np.maximum(obasis.shape[1],obasis2.shape[1])
    return np.sqrt(d-np.sum(obasis.T@obasis2)**2)

def find_active(eigval,eigvec,threshold = .95):
    target = threshold * np.sum(eigval)
    svar=0
    adim=0
    while svar < target:
        svar += eigval[adim]
        adim+=1
    
    print('Subspace Dimension',adim)
    print(eigval[0:adim])
    print('Subspace',eigvec[:,0:adim])

    return adim 