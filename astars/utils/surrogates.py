def train_rbf(xtrain,ftrain,noise=None,cutoff=.9):
    import numpy as np
    
#   from active_subspaces.utils.response_surfaces import PolynomialApproximation
    from active_subspaces.utils.response_surfaces import RadialBasisApproximation 
    import active_subspaces as ac
    
    ss=ac.subspaces.Subspaces()
    dim=xtrain.shape[1]
    if ftrain.size > (dim+1)*(dim+2)/2:
        rb_approx=RadialBasisApproximation(N=2)
        #print('Quadratic Monomials')
    else:
        rb_approx=RadialBasisApproximation(N=1)
    if noise is None:
        rb_approx.train(xtrain,ftrain) 
    else:
        rb_approx.train(xtrain,ftrain,v=noise*np.ones(ftrain.shape))
    #New-use large MC set on [-1,1] to predict df 
    #TODO compute exact integral of polynomials, RBF on [-1,1]? 
    #TODO or use unnormalized variables and treat as ind guassian RV?
    mcpts = 2*np.random.rand(100**2,dim)-1
    [temp,df]=rb_approx.predict(mcpts,compgrad=True)
    ss.compute(df=df)  
    return ss,rb_approx
