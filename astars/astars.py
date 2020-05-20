import numpy as np
import active_subspaces as ac


def get_mu_star(var,L_1,N):
    
	return ((8*var*N)/(L_1**2*(N+6)**3))**0.25



def get_h(L1,N):

	return 1/(4*L1*(N+4))


def get_mult_mu_star(var,L_1,N):
	return ((16*var*N)/((L_1**2)*(1+3*var)*(N+6)**3))**0.25


def STARS(x_init,F,mu_star,h,active=None,mult=False,wts=None):
    
	'''     
	x_init: initial x value, size Px1
    
	F: function we wish to minimize
; note that lower f's are evaluations of F(*)
    
	mu_star: smoothing parameter
    
	h: step length 
    
	active: Should be size Pxk
; default is none
	mult: Set to true if mult noise; default is add noise
	wts: Should be size Px1, filled with the weights
	     Default is None.
	     If using active vars, user should provide wts=ss.eigenvals    
    
	'''
    
    
	# Evaluate noisy F(x_init)
    
	f0 = F(x_init)


	# Compute mult_mu_star_k (if needed)
	if mult is False:
		mu_star = mu_star
	else:
		mu_star = mu_star*abs(f0)**0.5    
    
	
	# Draw a random vector of same size as x_init
    
	if active is None:
        
		u = np.random.normal(0,1,(np.size(x_init),1))
        
    
	else:
        
		act_dim = active.shape[1]

        
		if wts is None:
			lam = np.random.normal(0,1,(act_dim,1))

		else:
			lam = np.zeros((act_dim,1))
			for i in range(act_dim):
				lam[i]=np.random.normal(0,wts[0][0]/wts[i][0])        
		
		u = active@lam
    
    
    
	
	# Form vector y, which is a random walk away from x_init
   
	y = x_init + (mu_star)*u
    
    
	
	# Evaluate noisy F(y)
    
	g = F(y)
    
    
	
	# Form finite-difference "gradient oracle"
    
	s = ((g - f0)/mu_star)*u 
    
    
	
	# Take descent step in direction of -s smooth by h to get next iterate, x_1
    
	x = x_init - (h)*s
   
    
	
	# Evaluate noisy F(x_1)
    
	f1 = F(x)
    
    
	
	# Form upper bound for L_1
    
	#d1 = (f0-g)*(1/mu_star)
	d1 = (f0-g)
    
	#d2 = (g-f1)*(1/h)
	d2 = (g-f1)
    
	avg_step = .5*mu_star**2+.5*h**2
    
	L_1_B = abs(d2-d1)/avg_step    
        
    
	
	return [x, f1, y, g, x_init, f0, L_1_B]


	
	#if f1<=f0:
		#return [x, f1, y, g, x_init, f0, L_1_B]


	#else:
		#return [x_init, f0, y, g, x_init, f1, L_1_B]


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

## Here, slice from initial data
def astars_update_active(func,xinit,noise,L_1,finit=None,max_it=100,update_per=5,verbose=True):
    #from active_subspaces.utils.response_surfaces import PolynomialApproximation
    ''' xinit=initial training data '''
    
    dim=xinit.shape[0] #? check
    #print(dim)
    xhist_ad=np.copy(xinit)
    if finit==None:
        #generate initial data
        fhist_ad=func(xinit)
    else:
        fhist_ad=np.copy(finit)
    yhist_ad=np.zeros(0)
    ghist_ad=np.copy(0)
    x=np.copy(xinit)
    #get initial hyperparameters
    mu_star=get_mu_star(noise,L_1,dim)
    h=get_h(L_1,dim)
    #initial full space
    sub=np.eye(dim)
    ss=ac.subspaces.Subspaces()
    for i in range(max_it):
        if i>dim and i%update_per==0:
            if verbose:
                print('Computing Active Subspace after ',i,' steps')
            trainx=np.hstack((xhist_ad,yhist_ad))
            trainf=np.vstack((fhist_ad,ghist_ad))
            if verbose:
                print('Training Data Size',trainx.shape)
                print('Training Output Size',trainf.shape)
            #compute mixed RBF + subspace approximation
            ss,rbf=train_rbf(trainx.transpose(),trainf,noise=noise)
        #look for 90% variation
            our_wts=np.sqrt(ss.eigenvals)
            total_var=np.sum(our_wts)
            svar=0
            adim=0
            while svar < .9*total_var:
                svar += our_wts[adim]
                adim+=1
            if verbose:
                print('Subspace Dimension',adim)
                #print(our_wts)
            sub=ss.eigenvecs[:,0:adim]
        ##update ASTARS parameters
            mu_star=get_mu_star(noise,L_1,adim)
            h=get_h(L_1,adim)
    ## take ASTARS STEP
        [x,f,y,fy,p,q,L1B]=STARS(x,func,mu_star,h,active=sub)
        xhist_ad=np.hstack((xhist_ad,x))
        fhist_ad=np.vstack((fhist_ad,f))
        if yhist_ad.size > 0:
            yhist_ad=np.hstack((yhist_ad,y))
            ghist_ad=np.vstack((ghist_ad,fy))
        else:
            yhist_ad=np.array(y)
            ghist_ad=np.array(fy)
    #L1Bhist=np.vstack((L1Bhist,L1B))
    return xhist_ad,yhist_ad,fhist_ad,ghist_ad