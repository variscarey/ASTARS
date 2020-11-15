
print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))


from .utils.stars_param import get_L1,ECNoise
#from .utils.surrogates import train_rbf
from .utils.misc import find_active, subspace_dist
from scipy.special import gamma

import numpy as np
import active_subspaces as ac


class Stars_sim:
    
    def __init__(self,f,x_start,f_obj = None, L1=None,var=None,mult=False,verbose=False,maxit=100,train_method=None,true_as=None): 

        #constructor input processing
        self.f = f
        self.L1 = L1
        self.x = np.copy(x_start)
        self.var = var #variance of additive noise
        self.mult = mult
        self.verbose = verbose
        self.maxit = maxit
        self.STARS_only = False
        self.train_method = train_method
        self.set_dim = False
        self.subcycle = False
        self.threshadapt = False
        self.cycle_win = 10
        self.lasso = False
        self.f_obj = f_obj
        
        self.as_comp_step = None
        
      
        self.sphere = False

        self.active_step = 0
        self.pad_train = 1.0
        self.explore_weight = 1.0

        
        #default internal settings, can be modified.
        self.update_L1 = False
        self.active = None
        self.adim = None ###
        self.true_as = true_as
        self.iter = 0
        self.Window = None  #integer, window size for surrogate construction
        self.debug = False
        self.adapt = 5  #adapt active subspace every 5 steps.
        self.use_weights = False
        self.wts = None
        #process initial input vector
        if self.x.ndim > 1:
            self.x = self.x.flatten()
        self.dim = self.x.size
        self.threshold = .95 #set active subspace total svd threshold
        
        #subcycling
        self.subcycle_on = False # for subcycling (for now) --J
        self.project = False
        self.directions = None
        self.slope_weight = 1.0
        
        self.norm_surrogate = True #normalize surrogate to -1,1 and compute.
        
        
        #preallocate history arrays for efficiency
        self.x_noise = None
        self.xhist = np.zeros((self.dim,self.maxit+1))
        self.fhist = np.zeros(self.maxit+1)
        self.ghist = np.zeros(self.maxit)
        self.yhist = np.zeros((self.dim,self.maxit))
        self.L1_hist = np.zeros(self.maxit+1)
        
        if self.train_method == 'GQ':
            self.tr_stop = (self.dim + 1) * (self.dim + 2)// 2 # for quads only!
        elif self.train_method == 'GL':
            self.tr_stop = self.dim + 1 // 2
        else:
            self.tr_stop = self.dim #for LL, need to work out if sharp?
            
        if self.maxit > self.tr_stop:
            self.adim_hist = np.zeros(self.maxit-self.tr_stop-1)
            if self.true_as is not None:
                self.sub_dist_hist = np.zeros(self.maxit-self.tr_stop-1)

        if self.var is None:
            h=0.1
            while self.var is None:
                if self.verbose:
                    print('Calling ECNoise to determine noise variance,h=',h)
                sigma,temp=ECNoise(self.f,self.x,h=h,mult_flag=self.mult)
                #print(type(temp))
                if sigma > 0:
                    #ECNoise success
                    self.var=sigma**2
                    if self.verbose:
                        print('Approx. variance of noise=',self.var)
                    self.x_noise=temp[0]
                    self.f_noise=temp[1]
                    if self.L1 is None:
                        self.L1 = temp[2]
                        print('L1 from EC Noise', self.L1)
                elif temp == -1:
                    h /= 100
                elif temp == -2:
                    h *= 100   
                    
        print('Input L1',L1,'Input Variance',var)
        if L1 is None and var is None:
            print('Determining initial L1 from ECNoise Data')
  
            x_1d=h*np.arange(0,self.x_noise.shape[1])
            self.L1=get_L1(x_1d,self.f_noise,self.var)
        
            print('Approximate value of L1=',self.L1)
        self.xhist[:,0] = self.x
        self.fhist[0] = self.f(self.x)
        self.sigma = np.sqrt(self.var)
        self.regul = self.var
            
    def get_mu_star(self):
        if self.active is None:
            N = self.dim
        else:   
            N = self.active.shape[1]
        var = self.var
        L_1 = self.L1
        if not self.mult:
            self.mu_star = ((8*var*N)/(L_1**2*(N+6)**3))**0.25
        else:
            self.mu_star = ((16*var*N)/((L_1**2)*(1+3*var)*(N+6)**3))**0.25

    def get_h(self):
        L1=self.L1
        if self.active is None:
            N = self.dim
        else:
            N = self.active.shape[1]
        self.h=1/(4*L1*(N+4))

    def step(self):
        '''    
        Takes 1 step of STARS or ASTARS depending on current object settings
        '''
    
        f0=self.fhist[self.iter]

        if not self.STARS_only:
            if self.active is None or (self.adapt > 0  and self.iter%self.adapt == 0):
                if self.train_method is None or self.train_method == 'LL':
                    if 2*self.iter > self.pad_train*self.dim:
                        self.compute_active() 
                        self.active_step = self.iter  
                elif self.train_method == 'GQ':
                    if 2*self.iter + 1 > self.pad_train *( .5*(2 + self.dim) * (1+self.dim)):
                        self.compute_active()
                        self.active_step = self.iter
                elif self.train_method == 'GL':
                    if 2*self.iter + 1 > self.pad_train*.5 *(self.dim +1):
                        self.compute_active()
                        self.active_step = self.iter
        # Compute mult_mu_star_k (if needed)
        if self.mult is False:
            mu_star = self.mu_star
        else:
            mu_star = self.mu_star*abs(f0)**0.5    
        # Draw a random vector of same size as x_init
    
        if self.STARS_only:
            u = np.random.normal(0,1,(self.dim))
        elif self.active is None:
           if self.use_weights is False or self.wts is None:
               u = np.random.normal(0,self.explore_weight,(self.dim))
           else: 
                u = self.wts * np.random.randn(self.dim)
            #if self.sphere is True and self.debug is True:
            #    print('original step length',np.linalg.norm(u))
            #    u /= np.linalg.norm(u)
            #    u *= np.sqrt(2)*gamma((self.dim+1)/2)
            #    u /= gamma(self.dim/2)
                
        else: 
            act_dim=self.active.shape[1]
            if self.use_weights is False or self.wts is None:
                lam = np.random.normal(0,1,(act_dim))
            else:
                lam = np.zeros(act_dim)
                lam = self.wts[0:act_dim] * np.random.randn(act_dim) 
            if self.sphere is True:
                lam /= np.linalg.norm(lam)
                lam *= np.sqrt(2)*gamma((act_dim+1)/2)
                lam /= gamma((act_dim/2))
            u = self.active@lam
            if self.debug is True:
                print('Iteration',self.iter,'Oracle step=',u)
            #print(u.shape)
            if self.maxit > self.tr_stop and self.iter > self.tr_stop:
                self.adim_hist[self.iter-self.tr_stop-1] = act_dim
                if self.true_as is not None:
                    self.sub_dist_hist[self.iter-self.tr_stop-1] = subspace_dist(self.true_as,self.active)
            
    
        # Form vector y, which is a random walk away from x_init
        y = self.x + mu_star*u
        # Evaluate noisy F(y)
        g = self.f(y)
        # Form finite-difference "gradient oracle"
        s = ((g - f0)/self.mu_star)*u 
        # Take descent step in direction of -s smooth by h to get next iterate, x_1
        self.x -= (self.h)*s
        # stack here
        self.iter+=1
        self.xhist[:,self.iter]=self.x
        self.fhist[self.iter]=self.f(self.x) #compute new f value
        self.yhist[:,self.iter-1]=y
        self.ghist[self.iter-1]=g
        self.L1_hist[self.iter-1]=self.L1
        
        # Check for stagnation of convergence regardless of method
        if (self.subcycle is True or self.threshadapt is True) and self.active is not None:
            if self.iter - self.active_step > np.minimum(20, self.adapt/2) and self.threshold<1-self.var:
                #neccesary to not do this every step!
                #self.active_step = self.iter
                fsamp = self.fhist[-self.cycle_win+self.iter+1:self.iter+1]
                poly = np.polyfit(np.arange(self.cycle_win),fsamp,1)
                
                # PRINT SUBSPACE DIST FOR DIAGNOSTICS
                
                
                # Old tries at slowness check:
                #if poly[0] > -4*(self.adim+4)*self.L1/(self.iter+1):
                #if poly[0] > -4*self.L1*(self.adim+4)/(5*np.sqrt(2)):
                #if poly[0] > - 1 / (4 * (self.adim + 4)):
                
                # If too slowly, user can optionally apply either Adaptive Thresholding or Active Subcycling.

                if poly[0] > - self.slope_weight * self.dim * self.sigma / (2 ** 0.5):
                    self.active_step = self.iter
                    print('Iteration ',self.iter)
                    print('Bad Average recent slope',poly[0])
                    
                    # Adaptive Thresholding
                    if self.threshadapt is True:
                        norm_e_vals = self.eigenvals / np.sum(self.eigenvals)
                        
                        if self.adim < self.dim:
                            self.adim += 1
                            if self.adim < self.dim:
                                self.threshold = np.sum(norm_e_vals[0:self.adim])
                                self.active = np.hstack((self.active,self.inactive[:,0].reshape(-1,1)))
                                self.get_mu_star()
                                self.get_h()
                                print('Threshold was increased to', self.threshold, 'for the user due to slow convergence.')
                                print('Active subspace dimension is now', self.adim)
                        else:
                            print('Active Subspace is full Suspace, no increase')
                    # Active Subcycling
                    else:
                        if self.sub_method == 2: 
                            self.subcyle_on = True
                            self.project = True
                            #append current basis to list of bases
                            if self.inactive is None:
                                print('Exhausted all dimensions')
                                print('Reverting to ASTARS at iteration ', self.iter)
                                #self.STARS_only = True
                                self.project = False
                                self.active = None
                                self.directions = None
                                self.subcycle = True
                                #check?
                                self.update_L1 = False
                                self.compute_active()
                                print('Active Dimension after Subcycle',self.adim)
                            else:
                                print('Performing ASTARS in the inactive variables, total dimension = ',self.inactive.shape[1])
                                self.directions = self.inactive
                                self.subcyle_on = True
                                self.project = True    
                                self.update_L1 = True
                                if self.directions.shape[1] > 1:
                                    self.compute_active()
                                else:
                                    self.active = self.directions
                                    self.inactive = None
                                print('Subcycle Active Variables dimension',self.adim)
                                #if self.verbose:
                                print('Subcycle Active Variables')
                                print(self.active)
                           
                        elif self.sub_method ==1:
                            if self.subcycle_on is False:
                                self.active = self.inactive
                                self.adim = self.active.shape[1]
                                print('active subcycling has kicked in, dim of I is:  ',self.adim)
                            
                                self.get_mu_star()
                                self.get_h()
                                self.adapt = 0 # turn off adapt
                                self.subcycle_on = True
                            elif self.subcycle_on is True:
                                self.adapt = self.dim
                                #self.Window = (self.dim)*(self.dim+1) //2
                                print('Subcycle ended, recomputing active Subspace at iteration', self.iter)
                                self.compute_active()
                                self.subcycle_on = False
                    
    
        if self.update_L1 is True and (self.active is None or self.train_method != 'GQ') and self.mult is False:
            #call get update_L1:  approximates by regularized quadratic
            #not implemented for multiplicative noise yet 
            #1Dx data
            nrmu=np.linalg.norm(u)
            x1d=np.array([0,mu_star*nrmu,-self.h*np.dot(s,u/nrmu)])
            f1d=np.array([f0,self.ghist[self.iter-1],self.fhist[self.iter]])
            L1=get_L1(x1d,f1d,self.var,degree = 2)
            if self.verbose is True:
                print('Local L1 value',L1)
            if L1 > self.L1:
                self.L1=L1
                print('Updated L1 to',self.L1)
                print('Updating hyperparameters')
                self.get_mu_star()
                self.get_h()
            
        if self.debug:
            print(self.iter,self.fhist[self.iter])
            with (np.printoptions(precision = 4, suppress = True)):
                print(self.x)
            

    def compute_active(self):
    
        if self.as_comp_step is None:
            self.as_comp_step = np.array(self.iter)
        else:
            self.as_comp_step = np.append(self.as_comp_step,self.iter)

        ss=ac.subspaces.Subspaces()
        
        if self.f_obj is None:
            train_x,train_f = self.assemble_data()
        else:
            train_x,train_f = self.f_obj.inputs, self.f_obj.outputs
    
        if self.project is True:
            train_x = train_x @ self.directions
            #should be samples by subspace dim but may need to transpose
            
        if self.verbose:
            print('Computing Active Subspace after ',self.iter,' steps')
            print('Training Data Size',train_x.shape)
            print('Training Output Size',train_f.shape)
            if self.train_method is None:
                print('Using RBF+Poly')
            else:
                print('Using ',self.train_method)
    
        #Normalize data for as
        if self.norm_surrogate is True:
            train_x = self.normalize_data(train_x)
            if self.verbose:
                print('Bounds for surrogate')
                print(np.vstack((self.lb,self.ub)))
       
        
        
        
        if self.train_method is None:          
            if train_f.size > (self.dim+1)*(self.dim+2)/2:
                gquad = ac.utils.response_surfaces.PolynomialApproximation(N=2)
                gquad.train(train_x, train_f, regul = self.regul, lasso=self.lasso) #regul = self.var)
                # get regression coefficients
                b, A = gquad.g, gquad.H

                # compute variation of gradient of f, analytically
                # normalization assumes [-1,1] inputs from above
            
                C = np.outer(b, b.transpose()) + 1.0/3.0*np.dot(A, A.transpose())
                if self.norm_surrogate is True:
                    D = np.diag(1.0/(.5*(self.ub-self.lb).flatten()))
                    C = D @ C @ D
                ss.eigenvals,ss.eigenvecs = ac.subspaces.sorted_eigh(C)
                self.surrogate = gquad
            elif train_f.size > (self.dim + 1):
                #use local linear models instead!
                #gquad = ac.utils.response_surfaces.PolynomialApproximation(N=1)
                #gquad.train(train_x, train_f, regul = self.regul)
                #b = gquad.g
                #C = np.outer(b, b.transpose()) #+ 1.0/3.0*np.dot(A, A.transpose())
                #D = np.diag(1.0/(.5*(ub-lb).flatten()))
                #C = D @ C @ D
                #ss.eigenvals,ss.eigenvecs = ac.subspaces.sorted_eigh(C)
                df = ac.gradients.local_linear_gradients(train_x, train_f.reshape(-1,1)) 
                #chain rule for LL
                df = df / (.5*(self.ub-self.lb).flatten())
                ss.compute(df=df, nboot=0)
                self.surrogate = None
        elif self.train_method == 'LL':
            #Estimated gradients using local linear models
            #print(train_x.size,train_f.size)
            df = ac.gradients.local_linear_gradients(train_x, train_f.reshape(-1,1)) 
            #chain rule for LL
            df = df / (.5*(self.ub-self.lb).flatten())
            ss.compute(df=df, nboot=0)
            self.surrogate = None
        
        elif self.train_method == 'GQ':
            #use global quadratic surrogate
            
            gquad = ac.utils.response_surfaces.PolynomialApproximation(N=2)
            gquad.train(train_x, train_f, regul = self.regul, lasso = self.lasso) #regul = self.var)

            # get regression coefficients
            b, A = gquad.g, gquad.H

            # compute variation of gradient of f, analytically
            # normalization assumes [-1,1] inputs from above
            
            C = np.outer(b, b.transpose()) + 1.0/3.0*np.dot(A, A.transpose())
            if self.norm_surrogate is True:
                D = np.diag(1.0/(.5*(self.ub-self.lb).flatten()))
                C = D @ C @ D
            ss.eigenvals,ss.eigenvecs = ac.subspaces.sorted_eigh(C)
        
            #print('Condition number',gquad.cond)
            if self.verbose is True:
                print('Surrogate Rsqr',gquad.Rsqr)
            self.surrogate = gquad
            if self.norm_surrogate is True:
                self.surr_domain = np.hstack((self.lb,self.ub))
        elif self.train_method == 'GL':
            #use global linear surrogate
            glin = ac.utils.response_surfaces.PolynomialApproximation(N=1)
            glin.train(train_x, train_f, regul = self.regul, lasso = self.lasso) #regul = self.var)

            # get regression coefficients
            b = glin.g

            # compute variation of gradient of f, analytically
            # normalization assumes [-1,1] inputs from above
            
            C = np.outer(b, b.transpose()) # + 1.0/3.0*np.dot(A, A.transpose())
            if self.norm_surrogate is True:
                D = np.diag(1.0/(.5*(self.ub-self.lb).flatten()))
                C = D @ C @ D
            ss.eigenvals,ss.eigenvecs = ac.subspaces.sorted_eigh(C)
            
            

        if self.set_dim is False:

            self.adim = find_active(ss.eigenvals, ss.eigenvecs, threshold = self.threshold)
    
            
        if self.verbose or self.debug:
            print('Subspace Dimension',self.adim)
            print(ss.eigenvals[0:self.adim])
            print('Subspace',ss.eigenvecs[:,0:self.adim])
            
        if self.update_L1 is True and self.surrogate is not None:
            mapH = D @ gquad.H @ D
            
            sur_L1 = np.linalg.norm(mapH)
            if self.verbose is True:
                print('L1 from surrogate',sur_L1)
            if self.project is True:
                self.L1 = np.minimum(sur_L1,self.L1)
            else:
                self.L1 = sur_L1

        if self.directions is not None:
            ss.eigenvecs = self.directions @ ss.eigenvecs 
            
        if self.verbose is True or self.debug is True:
            print('Subspace Dimension',self.adim)
            print('Eigenvalues',ss.eigenvals[0:self.adim].flatten())
            print('Subspace',ss.eigenvecs[:,0:self.adim])
            
        self.active=ss.eigenvecs[:,0:self.adim]
        if self.adim < ss.eigenvecs.shape[1]:
            self.inactive=ss.eigenvecs[:,self.adim:]
        else:
            self.inactive = None
        
        self.wts = 1/np.sqrt(ss.eigenvals[0:self.adim].flatten())
        self.wts /= self.wts[0]
        self.wts = np.minimum(10*np.ones(self.wts.shape),self.wts)

        self.eigenvals = ss.eigenvals
     
        ##update ASTARS parameters
        self.get_mu_star()
        self.get_h()


    def assemble_data(self):
        if self.Window is not None: 
            start = np.maximum(0,self.iter - self.Window)
            train_x=np.hstack((self.xhist[:,start:self.iter+1],self.yhist[:,start:self.iter]))
            train_f=np.hstack((self.fhist[start:self.iter+1],self.ghist[start:self.iter]))
        else:
            if self.x_noise is None:
                train_x=np.hstack((self.xhist[:,0:self.iter+1],self.yhist[:,0:self.iter]))
                train_f=np.hstack((self.fhist[0:self.iter+1],self.ghist[0:self.iter]))
 
            else:
                train_x=np.hstack((self.x_noise,self.xhist[:,1:self.iter+1],self.yhist[:,0:self.iter]))
                train_f=np.hstack((self.f_noise,self.fhist[1:self.iter+1],self.ghist[0:self.iter]))
            
        return train_x.T,train_f.reshape(-1,1)
    
    def normalize_data(self,train_x):
    
        lb = np.amin(train_x,axis=0)
        ub = np.amax(train_x,axis=0)
        self.lb = lb
        self.ub = ub
        nrm_data=ac.utils.misc.BoundedNormalizer(self.lb,self.ub)
        train_x=nrm_data.normalize(train_x)
        
    
        return train_x