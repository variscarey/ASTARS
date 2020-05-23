def train_rbf(xtrain,ftrain,noise=None,cutoff=.9):
    from active_subspaces.utils.response_surfaces import PolynomialApproximation
    from active_subspaces.utils.response_surfaces import RadialBasisApproximation 
    from active_subspaces.utils.response_surfaces import RadialBasisApproximation 
    
    ss=ac.subspaces.Subspaces()
    dim=xtrain.shape[1]
    if ftrain.size > (dim+1)*(dim+2)/2:
        rb_approx=RadialBasisApproximation(N=2)
        #print('Quadratic Monomials')
    else:
        rb_approx=RadialBasisApproximation(N=1)
    if noise==None:
        rb_approx.train(xtrain,ftrain) 
    else:
        rb_approx.train(xtrain,ftrain,v=noise*np.ones(ftrain.shape))
    [temp,df]=rb_approx.predict(xtrain,compgrad=True)
    ss.compute(df=df)
    return ss,rb_approx
