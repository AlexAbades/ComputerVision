def cornerDetector(im, sigma:int, epsilon:int, k:int=0.06, tau:int=None):
    """
    Apply threshold 
    Apply Non maximum suppresion, then filter with threshold tau.
        (I(x, y) − I(x′, y′)) > 0 ∀ x′ ∈ N(x, y) 
    Where N(x, y) is a neighbourhood around the point (x, y)
        (I(x, y) − I(x′, y′)) > 0 = (I(x, y) > I(x′, y′)
    """

        
    r = harrisMeasure(im, sigma, epsilon, k)
    
#     Create Threshold 
    if not tau:
        tau = 0.8*np.max(r)
        
    idx = np.where(r>tau)
    r_t = np.zeros(im.shape)
    r_t[idx] = r[idx]
    
    # Non maximum suppresion 
    # aplicar find peak from algorithms to sund peaks in each row and each column
#     r_cor = np.ones(im.shape)
    r_zeros = np.zeros(im.shape)
    
    # r(x, y) > r(x, y + 1)
    Up = (r_t[:-1,:] > r_t[1:,:])
    # r(x, y) ≥ r(x, y − 1) 
    Low = (r_t[:-1,:] <= r_t[1:,:])
    
    
    # r(x, y) > r(x + 1, y)
    R = (r_t[:,:-1] > r_t[:,1:])
    
    # r(x, y) ≥ r(x − 1, y)
    Lef = (r_t[:,:-1] <= r_t[:,1:])
    
    # Get columns
    col, row = im.shape
    
    # Create a row of false (if we want to consider the boudaries, we should use True )
    falsy_column = np.array([False] * row)
    falsy_row = np.array([False] * col)
    
    # Append Columns
    R = np.column_stack([R, falsy_column])
    Left = np.column_stack([falsy_column, Lef])
    Up = np.vstack([Up, falsy_row])
    Low = np.vstack([falsy_row, Low])

    
    # Get index when R and L are true 
    idx_x = np.where(R & Left)
    # Get index when Low and Up are true
    idx_y = np.where(Up & Low)
    # Cast all values that satisfy the condition to it's real value 
    r_zeros[idx_x] = r_t[idx_x]
    r_zeros[idx_y] = r_t[idx_y]
    
    
    
#     # r(x, y) > r(x, y + 1)
#     R = (r_zeros[:-1,:] < r_zeros[1:,:])
#     # r(x, y) ≥ r(x − 1, y)
#     L = ~R
    
    
        
#     # r(x, y) > r(x + 1, y)
#     idx_right = np.where(r_t[:-1,:] < r_t[1:,:])
#     r_cor[idx_right] = 0
    
#     # r(x, y) ≥ r(x − 1, y) 
#     idx_left = np.where(r_t[1:,:] <= r_t[:-1,:])
#     r_cor[idx_left] = 0
    
#     # r(x, y) > r(x, y + 1) ∧
#     idx_up = np.where(r_t[:,:-1] <= r_t[:,1:])
#     r_cor[idx_up] = 0
    
#     # r(x, y) ≥ r(x, y − 1) 
#     idx_down = np.where(r_t[:,1:] <= r_t[:,:-1])
#     r_cor[idx_down] = 0
    
# #     idx = np.where(r_cor == 1)
    
    
    
    ## 
    
#     Then once we have applies the non maximal supression we end with a matrix of 0 and 1, that can be seen as a matrix of true and False.
#  once we have that we should pass it into the matrix r en select only the true values, or the 1

    return r_zeros, R