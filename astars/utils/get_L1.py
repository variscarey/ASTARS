def get_L1(xdata,fdata,var):
    import numpy as np

#   ''' computes approximate L1 Lipschitz constant by using 1D polynomial'''

    weight=var*np.ones(xdata.size)
    degree=min(xdata.size-1,5)
    poly=np.polynomial.polynomial.Polynomial.fit(xdata,fdata,degree,w=weight)
    temp=(poly.deriv(2)).linspace()
    return np.max(np.absolute(temp[1]))


    
    
