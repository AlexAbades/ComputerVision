def nonMaxsupression4neig(I, boundaries:bool=False, eight:bool=False):
    """ 
    Apply Non maximum suppresion.
        (I(x, y) − I(x′, y′)) > 0 ∀ x′ ∈ N(x, y) 
    Where N(x, y) is a 4 neighbourhood around the point (x, y)
        (I(x, y) − I(x′, y′)) > 0 = (I(x, y) > I(x′, y′)
    """
    col,row, d = I.shape
    
    # Create false columns and row for the columns and rows supressed 
    if boundaries:
        falsy_r = np.ones((col,d))
        falsy_c = np.ones((row,d))        
    else:
        falsy_r = np.zeros((col,d))
        falsy_c = np.zeros((row,d))
    # ToDo: Change var names, they are confusing, falsy_c should be a column vector
    # ToDo: optimize, code, when not using boundaries, it could be fatser not casting the falsy columns and rows 
    
    # Remember (columns x rows)
    # Upper neighbour
    # r(x, y) > r(x, y + 1)
    Up = I[:-1,:,:] >= I[1:,:,:]
    Up_f = np.zeros(I.shape)
    Up_f[-1,:,:] = falsy_c
    Up_f[:-1,:,:] = Up
    Up_f = Up_f == 1
    
    # Lower neighbour 
    # r(x, y) ≥ r(x, y − 1) 
    Low = I[1:,:,:] > I[:-1,:,:] 
    Low_f = np.zeros(I.shape)
    Low_f[0,:,:] = falsy_c
    Low_f[1:,:,:] = Low
    Low_f = Low_f == 1 
    
    # Right Neigbhour
    # r(x, y) > r(x + 1, y)
    Right = I[:,:-1,:] >= I[:,1:,:]
    Right_f = np.zeros(I.shape)
    Right_f[:,-1,:] = falsy_r
    Right_f[:,:-1,:] = Right 
    Right_f = Right_f == 1 
    
    # Left Neighbour 
    # r(x, y) ≥ r(x − 1, y)
    Left = I[:,1:,:] > I[:,:-1,:]
    Left_f = np.zeros(I.shape)
    Left_f[:,0,:] = falsy_r
    Left_f[:,1:,:] = Left
    Left_f = Left_f == 1
        
    # Find index where all conditions are True
    idx = np.where(Left_f & Right_f & Up_f & Low_f)
    # Create an empty image 
    I_tmp = np.zeros(I.shape)
    # Map the values that fulfill the condition on the empty matrix
    I_tmp[idx] = I[idx]
    
    if eight:
        # We want to use it in 8 neighbouts, return an array of Ture where condition is fullfiled
        return I_tmp != 0 
    else:
        return I_tmp
    

def nonMaxsupression8neig(I, boundaries:bool=False, boolean:bool=False):
    """
    Apply Non maximum suppresion.
        (I(x, y) − I(x′, y′)) > 0 ∀ x′ ∈ N(x, y) 
    Where N(x, y) is a 8 neighbourhood around the point (x, y)
        (I(x, y) − I(x′, y′)) > 0 = (I(x, y) > I(x′, y′)
    """
    
    # Not sure if we'll pass it with d dimensions 
    col,row, d = I.shape
    
    # Create false columns and row for the columns you erase 
    if boundaries:
        falsy_r = np.ones((col,d))
        falsy_c = np.ones((row,d))
    else:
        falsy_r = np.zeros((col,d))
        falsy_c = np.zeros((row,d))
    
       
    # Comparing Top left Corner
    TL = I[1:,1:,:] > I[:-1,:-1,:]
    TL_f = np.zeros(I.shape)
    TL_f[0,:,:] = falsy_c
    TL_f[:,0,:] = falsy_r
    TL_f[1:,1:,:] =  TL 
    TL_f = TL_f == 1
    
    # Comparing Top right Corner 
    TR = I[1:,:-1,:] >= I[:-1,1:,:]
    TR_f = np.zeros(I.shape)
    TR_f[0,:,:] = falsy_c
    TR_f[:,-1,:] = falsy_r
    TR_f[1:,:-1,:] = TR
    TR_f = TR_f == 1
    
    # Comparing Bottom Right Corner
    BR = I[:-1,:-1,:] >= I[1:,1:,:] 
    BR_f = np.zeros(I.shape)
    BR_f[-1,:,:] = falsy_c
    BR_f[:,-1,:] = falsy_r    
    BR_f[:-1,:-1,:] = BR
    BR_f = BR_f == 1
    
    # Comparing Bottom Left Corner 
    BL = I[:-1,1:,:] > I[1:,:-1,:]
    BL_f = np.zeros(I.shape)
    BL_f[-1,:,:] = falsy_c
    BL_f[:,0,:] = falsy_r
    BL_f[:-1,1:,:] = BL
    BL_f = BL_f == 1
    
    # Find index where conditions are fulfilled 
    idx_corners = np.where(TL_f & TR_f & BL_f & BR_f)
    # Create en empty ndarray    
    corners = np.zeros(I.shape)
    corners[idx_corners] = 1
    corners = corners == 1
    
    neig4 = nonMaxsupression4neig(I, eight=True)
    
    idx_8 = np.where(corners & neig4)
    
    neig8 = np.zeros(I.shape)
    neig8[idx_8] = I[idx_8]
    if boolean:
        return neig8 != 0 
    else:
        return neig8

    
def nonMaxsupression4neig_multilayer(Ilayer, Iabow, layer:str='above', boundaries:bool=False, eight:bool=False):
    """ 
    Apply Non maximum suppresion.
        (I(x, y) − I(x′, y′)) > 0 ∀ x′ ∈ N(x, y) 
    Where N(x, y) is a 4 neighbourhood around the point (x, y)
        (I(x, y) − I(x′, y′)) > 0 = (I(x, y) > I(x′, y′)
    
    Ilayer: Image we want to compare
    Iabow: I in layer above or below
    boundaries: False to detect peaks in boundaries False for not detecting 
    eight: True returns a boolean array, False returns the values that are peaks iin 4 nieghbour
    """
    
    # Check Images have the same dimension:
    if Ilayer.shape != Iabow.shape:
        print('Images introduced do not have the same dimensions')
        return False
    
    # Get the dimensions of the image 
    col,row = Ilayer.shape

    # Create false columns and rows so we can recreate thw image shape
    falsy_r = [boundaries]*col
    falsy_c = [boundaries]*row

    # Differentiate if it's the layer above or the layer below 
    if layer.lower() == 'above':
        # Check if the pixel in layer l is higher than the pixel in layer l ± 1
        # ToDo: Greater or equal or greater?
        Itslef = Ilayer >= Iabow

        # Check if the pixel in layer l is higher than the Right Neigbhour pixel in layer l ± 1
        # rl(x, y) > rl±1(x + 1, y)
        Right = Ilayer[:,:-1] >= Iabow[:,1:]
        Right = np.column_stack([Right, falsy_r])

        # Check if the pixel in layer l is higher than the Left Neigbhour pixel in layer l ± 1
        # rl(x, y) ≥ rl±1(x − 1, y)
        Left = Ilayer[:,1:] >= Iabow[:,:-1]
        Left = np.column_stack([falsy_r, Left])

        # Check if the pixel in layer l is higher than the Upper Neigbhour pixel in layer l ± 1
        # rl(x, y) > rl±1(x, y + 1)
        Up = Ilayer[:-1,:] >= Iabow[1:,:]
        Up = np.vstack([Up, falsy_c])

        # Check if the pixel in layer l is higher than the Lower Neigbhour pixel in layer l ± 1
        # rl(x, y) ≥ rl±1(x, y − 1) 
        Low = Ilayer[1:,:] >= Iabow[:-1,:] 
        Low = np.vstack([falsy_c, Low])

        # Find index where all conditions are True
        idx = np.where(Itslef & Left & Right & Up & Low)
        # Create an empty image 
        I_tmp = np.zeros((col,row))
         # Map the values that fulfill the condition on the empty matrix
        I_tmp[idx] = Ilayer[idx]
        
    # If it's the layer below, we'll compare only grater     
    elif layer.lower() == 'below':
        # Check if the pixel in layer l is higher than the pixel in layer l ± 1
        # ToDo: Greater or equal or greater?
        Itslef = Ilayer > Iabow

        # Check if the pixel in layer l is higher than the Right Neigbhour pixel in layer l ± 1
        # rl(x, y) > rl±1(x + 1, y)
        Right = Ilayer[:,:-1] > Iabow[:,1:]
        Right = np.column_stack([Right, falsy_r])

        # Check if the pixel in layer l is higher than the Left Neigbhour pixel in layer l ± 1
        # rl(x, y) ≥ rl±1(x − 1, y)
        Left = Ilayer[:,1:] > Iabow[:,:-1]
        Left = np.column_stack([falsy_r, Left])

        # Check if the pixel in layer l is higher than the Upper Neigbhour pixel in layer l ± 1
        # rl(x, y) > rl±1(x, y + 1)
        Up = Ilayer[:-1,:] > Iabow[1:,:]
        Up = np.vstack([Up, falsy_c])

        # Check if the pixel in layer l is higher than the Lower Neigbhour pixel in layer l ± 1
        # rl(x, y) ≥ rl±1(x, y − 1) 
        Low = Ilayer[1:,:] > Iabow[:-1,:] 
        Low = np.vstack([falsy_c, Low])
    else:
        print('layer attribute must be above/below')
        return False 
    
        
    # Find index where all conditions are True
    idx = np.where(Itslef & Left & Right & Up & Low)
    # Create an empty image 
    I_tmp = np.zeros((col,row))
     # Map the values that fulfill the condition on the empty matrix
    I_tmp[idx] = Ilayer[idx]
    
    if eight:
        # We want to use it in 8 neighbouts, return an array of Ture where condition is fullfiled
        return I_tmp != 0
    else:
        return I_tmp



    def nonMaxsupression8neig_multilayer(Ilayer, Iabow, layer:str='above', boundaries:bool=False, boolean: bool=False):
    """
    Apply Non maximum suppresion.
        (I(x, y) − I(x′, y′)) > 0 ∀ x′ ∈ N(x, y) 
    Where N(x, y) is a 8 neighbourhood around the point (x, y)
        (I(x, y) − I(x′, y′)) > 0 = (I(x, y) > I(x′, y′)
    
    Ilayer: Image we want to compare
    Iabow: I in layer above or below
    boundaries: False to detect peaks in boundaries False for not detecting 
    boolean: Return a Boolean Array if True and the numbers of the peaks if False. 
    
    """
    
    # Call 4 neighbour so we make checkings.
    neig4 = nonMaxsupression4neig_multilayer(Ilayer, Iabow, layer= layer, eight=True)
    
    shape = Ilayer.shape
    col,row = shape
    
    # Create false columns and row for the 
    falsy_r = [boundaries]*col
    falsy_c = [boundaries]*row
    
    if layer.lower() == 'above':
        # Check if the pixel in layer l is higher than the Top Left Corner Neigbhour pixel in layer l ± 1
        # rl(x, y) > rl±1(x-1, y-1)
        TL = Ilayer[1:,1:] >= Iabow[:-1,:-1]
        TL_f = np.zeros(shape)
        TL_f[0,:] = falsy_c
        TL_f[:,0] = falsy_r
        TL_f[1:,1:] =  TL 
        TL_f = TL_f == 1

        # Check if the pixel in layer l is higher than the Top Right Corner Neigbhour pixel in layer l ± 1
        # rl(x, y) > rl±1(x-1, y+1)
        TR = Ilayer[1:,:-1] >= Iabow[:-1,1:]
        TR_f = np.zeros(shape)
        TR_f[0,:] = falsy_c
        TR_f[:,-1] = falsy_r
        TR_f[1:,:-1] = TR
        TR_f = TR_f == 1

        # Check if the pixel in layer l is higher than the Bottom Right Corner Neigbhour pixel in layer l ± 1
        # rl(x, y) > rl±1(x+1, y+1)
        BR = Ilayer[:-1,:-1] >= Iabow[1:,1:] 
        BR_f = np.zeros(shape)
        BR_f[-1,:] = falsy_c
        BR_f[:,-1] = falsy_r    
        BR_f[:-1,:-1] = BR
        BR_f = BR_f == 1

        # Check if the pixel in layer l is higher than the Bottom Left Corner Neigbhour pixel in layer l ± 1
        # rl(x, y) > rl±1(x+1, y-1)
        BL = Ilayer[:-1,1:] >= Iabow[1:,:-1]
        BL_f = np.zeros(shape)
        BL_f[-1,:] = falsy_c
        BL_f[:,0] = falsy_r
        BL_f[:-1,1:] = BL
        BL_f = BL_f == 1
        
    elif layer.lower() == 'below':
        # Check if the pixel in layer l is higher than the Top Left Corner Neigbhour pixel in layer l ± 1
        # rl(x, y) > rl±1(x-1, y-1)
        TL = Ilayer[1:,1:] > Iabow[:-1,:-1]
        TL_f = np.zeros(shape)
        TL_f[0,:] = falsy_c
        TL_f[:,0] = falsy_r
        TL_f[1:,1:] =  TL 
        TL_f = TL_f == 1

        # Check if the pixel in layer l is higher than the Top Right Corner Neigbhour pixel in layer l ± 1
        # rl(x, y) > rl±1(x-1, y+1)
        TR = Ilayer[1:,:-1] > Iabow[:-1,1:]
        TR_f = np.zeros(shape)
        TR_f[0,:] = falsy_c
        TR_f[:,-1] = falsy_r
        TR_f[1:,:-1] = TR
        TR_f = TR_f == 1

        # Check if the pixel in layer l is higher than the Bottom Right Corner Neigbhour pixel in layer l ± 1
        # rl(x, y) > rl±1(x+1, y+1)
        BR = Ilayer[:-1,:-1] > Iabow[1:,1:] 
        BR_f = np.zeros(shape)
        BR_f[-1,:] = falsy_c
        BR_f[:,-1] = falsy_r    
        BR_f[:-1,:-1] = BR
        BR_f = BR_f == 1

        # Check if the pixel in layer l is higher than the Bottom Left Corner Neigbhour pixel in layer l ± 1
        # rl(x, y) > rl±1(x+1, y-1)
        BL = Ilayer[:-1,1:] > Iabow[1:,:-1]
        BL_f = np.zeros(shape)
        BL_f[-1,:] = falsy_c
        BL_f[:,0] = falsy_r
        BL_f[:-1,1:] = BL
        BL_f = BL_f == 1
    
    # Find index where conditions are fulfilled 
    idx_corners = np.where(TL_f & TR_f & BL_f & BR_f)
    # Create en empty ndarray of shape Image    
    corners = np.zeros(shape)
    corners[idx_corners] = 1
    corners = corners == 1
    
    idx_8 = np.where(corners & neig4)
    
    neig8 = np.zeros(shape)
    neig8[idx_8] = Ilayer[idx_8]
    
    if boolean:
        return neig8 != 0 
    else:
        return neig8
 