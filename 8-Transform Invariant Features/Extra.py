def gaussian1DKernel(sigma:int, scale:int=5 ):
    """
    sigma: stadard deviation of the gaussian distribution.
    scale: Range of x times the stadard deviatiion for the x values where we'll evaluate the gaussian.    
    Scale fixed to 5, following the rule of 3std = 99.7% of the data. We make sure we have at least that much.
    
    g: Gaussian distribution given a variance. Column Vector -> (n,1)
    gx: Gaussian distribution derivative given a variance. Column Vector -> (n,1)
    
    If instead of doing it 1D we were doing 2D Kerels, would be much more computational expensive
    from run time O(2*n) --> O(n^2)
    """
    # Width of the Gaussian
    h = np.ceil(scale*sigma)
    x = np.arange(-h, h+1)
    
    # Cte of the Gaussian
    c = (1)/(np.sqrt(2*np.pi)*sigma)
    c_exp = np.exp((-x**2)/(2*sigma**2))
    
    # Gaussian equation 
    g = c*c_exp
    
    # First derivative
    gx = (-x/sigma**2)*g
        
    # Second derivative 
    gxx = (-x/sigma**2)*gx - 1/sigma**2*g
    
    # Reshape to (n,1)
    g, gx, gxx = g.reshape(-1,1), gx.reshape(-1,1), gxx.reshape(-1,1)
    
    return g, gx, gxx 