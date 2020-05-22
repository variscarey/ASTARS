#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:08:23 2020

@author: Varis Carey
"""

import sys
import numpy as np
sys.path.append('../astars')

from astars import stars_sim

def testfun(x):
    return x[0]**2+x[-1]**2


for trials in range(100):
    init_pt=np.random.rand(10)
    stars_test=ASTARS_sim(testfun,init_pt,L1=2.0,var=1E-2)
    stars_test.get_mustar()
    stars_test.get_h()
    while stars_test.iter < stars_test.maxit:
        stars_test.stars_step()
    print(trials,stars_test.f)

    
