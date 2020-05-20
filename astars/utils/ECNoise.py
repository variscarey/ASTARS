## ECNoise: (f, x_b, h, M, mult) -> (sigma_hat, xvals, fvals)

## Inputs:  f: noisy black-box function; 
##          x_b: base point, Px1 col vec;
##	    h: discretization level which is 0.01 by default;
##          M: number of samples to use, M must be >3 and default is 7 (suggested in paper); 
##          mult: a boolean handle for whether we have add/mult noise.
##	    (We default to additive noise, so we set mult=False. User can set mult=True if needed.)

## Outputs: sigma_hat: an estimator to the variance in noise of f;
##	    xvals: the M x-values stored as an PxM array
##          fvals: the M function values stored as an Mx1 column vector

## Cost:    M evals of f

# We only need numpy for ECNoise
import numpy as np

def ECNoise(f, x_b, h=0.01, M=7, mult=False):

	# Start with sigma_hat at 0
	sigma_hat = 0

	# Throw error for M too small
	if M<=3:
		return print('Please choose M>3.')

	#reshape if row vector
	x_b.reshape(-1,1)
	# Grab the dimension of the domain of f

	P = x_b.shape[0]

	# Hard-coded normalized direction to sample in (either random or np.ones)
	#if x_b.ndim > 1:
	p_u = np.random.rand(P,1)
	#p_u = np.ones((P,1))
	#else:
	#	p_u = np.random.rand(P)
	#p_u = np.ones(P)
	# Normalize
	p = p_u/np.linalg.norm(p_u)

	# Initiate storage for x values (xvals) and their f values (fvals)
	xvals = np.copy(x_b)
	fvals = np.zeros((M,1))

	# Form and store the x/f values
	for i in range(0,M):
		intx = x_b + (i*h)*p
		if i>=1:
			xvals = np.hstack((xvals,intx))
		else:
			xvals = xvals
		fvals[i,0] = f(intx)

	# Test h
	fmin = np.min(fvals); fmax = np.max(fvals)
	diff = fmin - fmax
	if diff > np.max([np.abs(fmin),np.abs(fmax)])*0.1:
		# return print('h is too large. Try h/100 next. Default is h=0.01.') 
		sigma_hat = [-1]
		return sigma_hat

	# Form difference table
	gamma = 1
	diff_col = np.copy(fvals) # purposely overwritten at each step j below
	est = np.zeros((M-1,1)) # stores estimators
	d_sign = np.zeros((M-1,1)) # entry j will be 1 if there are sign changes, 0 if not
	
	for j in range(0,M-1):
		for i in range(0,M-j-1):
			diff_col[i,0] = diff_col[i+1,0] - diff_col[i,0]

		# h is too small when more than half the function values are equal (cite: More' and Wild)
		if j == 1 and np.shape(np.nonzero(diff_col))[1] < M/2:
			# return print('h is too small. Try 100*h next. Default is h=0.01.')
			sigma_hat = [-2]
			return sigma_hat
		else:
			h=h
		
		# Compute the estimates for the noise level
		gamma = 0.5*((j+1)/(2*(j+1)-1))*gamma
		est[j,0] = np.sqrt(gamma*np.mean(diff_col[0:M-j,0])**2)

		# Determine sign changes
		emin = np.min(diff_col[0:M-j,0]); emax = np.max(diff_col[0:M-j,0])
		if emin*emax<0:
			d_sign[j,0] = 1
		else:
			d_sign[j,0] = 0

	# Determine noise level
	for k in range(0,M-3):
		dmin = np.min(est[k:k+2,0]); dmax = np.max(est[k:k+2,0])
		if dmax <= 4*dmin and d_sign[k,0] == 1:
			sigma_hat = est[j,0]
			if mult is False:
				noise_array = [sigma_hat, xvals, fvals]
				return noise_array
			else:
				noise_array = [sigma_hat/fvals[0]**2, xvals, fvals] 
				return noise_array
		else:
			sigma_hat = 0

	# If we haven't found a sigma_hat, then h is too large (M and W)
	if sigma_hat == 0:
		# return print('h is too large. Try 0.01*h next. Default is h=0.01.')
		sigma_hat = [-1]
		return sigma_hat
