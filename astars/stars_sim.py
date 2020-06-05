print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))


from .utils.stars_param import get_L1,ECNoise
from .utils.surrogates import train_rbf

import numpy as np
import active_subspaces as ac


class Stars_sim:
    
    def __init__(self,f,x_start,L1=None,var=None,mult=False,verbose=False,maxit=100,train_method=None): 

        #constructor input processing
        self.f = f
        self.L1 =L1
        self.x = x_start
        self.var = var #variance of additive noise
        self.mult = mult
        self.verbose = verbose
        self.maxit = maxit
        self.STARS_only = False
        self.train_method = train_method
        
        #default internal settings, can be modified.
        self.update_L1 = False
        self.active = None
        self.iter = 0
        self.Window = None  #integer, window size for surrogate construction
        self.debug = False
        self.adapt = 5  #adapt active subspace every 5 steps.
        self.use_weights = False
        #process initial input vector
        if self.x.ndim > 1:
            self.x=self.x.flatten()
        self.dim = self.x.size
        
        #preallocate history arrays for efficiency
        self.xhist = np.zeros((self.dim,self.maxit+1))
        self.fhist = np.zeros(self.maxit+1)
        self.ghist = np.zeros(self.maxit)
        self.yhist = np.zeros((self.dim,self.maxit))

        if self.var is None:
            h=0.1
            while self.var is None:
                if self.verbose:
                    print('Calling ECNoise to determine noise variance,h=',h)
                temp=ECNoise(self.f,self.x,h=h,mult_flag=self.mult)
            
                if temp[0][0] > 0: 
                    #ECNoise success
                    self.var=temp[0][0]**2
                    if self.verbose:
                        print('Approx. variance of noise=',self.var)
                    self.x_noise=temp[0][1]
                    self.f_noise=temp[0][2]
                elif temp[0][0] == -1:
                    h /= 100
                elif temp[1][0] == -2:
                    h *= 100
            #allocate xhist and fhist to use training data as well
        if self.L1 is None:
            if self.verbose is True:
                print('Determining initial L1 from ECNoise Data')
  
            x_1d=h*np.arange(0,self.x_noise.shape[1])
            self.L1=get_L1(x_1d,self.f_noise,self.var)
            
            if self.verbose is True:
                print('Approximate value of L1=',self.L1)
        self.xhist[:,0] = self.x
        self.fhist[0] = self.f(self.x)
            
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

    def STARS_step(self):
        '''    
        Takes 1 step of STARS or ASTARS depending on current object settings
        '''
    
        f0=self.fhist[self.iter]

        if not self.STARS_only:
            if self.active is None or (self.adapt > 0  and self.iter%self.adapt == 0):
                if self.train_method is None:
                    if 2*self.iter > self.dim:
                        self.compute_active()   
                elif self.train_method == 'GQ':
                    if 2*self.iter + 1 > .5*(2 + self.dim) * (1+self.dim):
                        self.compute_active()
                        
        # Compute mult_mu_star_k (if needed)
        if self.mult is False:
            mu_star = self.mu_star
        else:
            mu_star = self.mu_star*abs(f0)**0.5    
        # Draw a random vector of same size as x_init
    
        if self.active is None:
            u = np.random.normal(0,1,(self.dim))
        else: 
            act_dim=self.active.shape[1]
            if self.use_weights is False:
                lam = np.random.normal(0,1,(act_dim))
            else:
                lam = np.zeros((act_dim))
                for i in range(act_dim):
                    lam[i]=np.random.normal(0,self.wts[0][0]/self.wts[i][0])        
            u = self.active@lam
            print(u.shape)
    
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
    
        #if self.update_L1 is True and self.mult is False:
            #call get update_L1:  approximates by regularized quadratic
            #not implemented for multiplicative noise yet 
            #1Dx data
            #x1d=np.array([0,np.linalg.norm(u),np.linalg.norm(s)])
            #L1=get_L1(x1d,[f0,g,self.fhist[self.iter]],self.var)
            #self.L1=np.max(L1,self.L1)
        if self.debug:
            print(self.iter,self.fhist[self.iter])
            

    def compute_active(self):
    

        ss=ac.subspaces.Subspaces()
        #TODO: include training data x_noise,f_noise
        if self.Window is None:
            train_x=np.hstack((self.xhist[:,0:self.iter+1],self.yhist[:,0:self.iter]))
            train_f=np.hstack((self.fhist[0:self.iter+1],self.ghist[0:self.iter]))
        else:
            train_x=np.hstack((self.xhist[:,-self.Window:],self.yhist[:,-self.Window:]))
            train_f=np.hstack((self.fhist[-self.Window:],self.ghist[-self.Window:]))
        if self.verbose:
            print('Computing Active Subspace after ',self.iter,' steps')
            print('Training Data Size',train_x.shape)
            print('Training Output Size',train_f.shape)
            if self.train_method is None:
                print('Using RBF+Poly')
            else:
                print('Using ',self.train_method)
    
        #Normalize data for as
        lb = np.amin(train_x,axis=1).reshape(-1,1)
        ub = np.amax(train_x,axis=1).reshape(-1,1)
        nrm_data=ac.utils.misc.BoundedNormalizer(lb,ub)

        #print(lb,ub)
        train_x=nrm_data.normalize(train_x.T)
        #print(train_x.shape)
        
        
        
        if self.train_method is None:
            #determine appropriate training method (RBF+polynomial)
            ss,rbf=train_rbf(train_x,train_f,noise=self.var)
        elif self.train_method == 'LL':
            #Estimated gradients using local linear models
            df = ac.gradients.local_linear_gradients(train_x, train_f) 
            ss.compute(df=df, nboot=0)
        elif self.train_method == 'GQ':
            #use global quadratic surrogate
            gquad = ac.utils.response_surfaces.PolynomialApproximation(N=2)
            gquad.train(train_x, train_f)
            #print(gquad.poly_weights)
            dummy,df = gquad.predict(train_x, compgrad=True)
            ss.compute(df=df, nboot=0)
        #look for 90% variation
        out_wts=np.sqrt(ss.eigenvals)
        total_var=np.sum(out_wts)
        svar=0
        adim=0
        while svar < .9*total_var:
            svar += out_wts[adim]
            adim+=1
        if self.verbose:
            print('Subspace Dimension',adim)
            print(out_wts)
            #print('Subspace',ss.eigenvecs[:,0:adim])
        if self.update_L1 is True and self.train_method != 'LL':
            if self.train_method == 'GQ':
                d2f = gquad.comp_hessian(train_x)
                temp = np.abs(d2f[0,:,:])
                print('|Hessian| on mapped domain',temp)
                scale = .5*(ub-lb)
                print('Variable Scalings',scale)
                scale = scale @ scale.T
                #print(scale)

                
                #sufficient for quadratic response surface
                self.L1 = np.amax(temp/scale)
                if self.verbose:
                    print('Updated L1 to',self.L1)
            #if self.train_method = None and rbf.N >= 2:
                
        self.active=ss.eigenvecs[:,0:adim]
        self.wts=out_wts
        ##update ASTARS parameters
        self.get_mu_star()
        self.get_h()


