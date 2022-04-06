import numpy as np 
import matplotlib.pyplot as plt

def box3d(n):
    """
    Given a n number of points, it crates a line with n equidistant points in all the edges of a square plus 3 edges in the 
    coordinate origin.
    It returns a matrix M of (3xn) with the 3d space coordinates of all the points.
    """
    
    # Define some variables
    m = np.linspace(-0.5,0.5,n)
    # Define the cte for the outside edges
    c = np.ones(n)*0.5
    # Cte for the inside edges
    c2 = np.zeros(n)
    # Empty 2d matrix to store all the values 
    B = np.array([[], [], []])
    # Signs for the outside edges
    signs = np.array([[1,1],[-1,1],[1,-1],[-1,-1]])
    
    for j in range(3):

        for i in range(4):
            if j == 0:
                A = np.array(m)
                A = np.vstack([A,c*signs[i][0]])
                A = np.vstack([A,c*signs[i][1]])
                B = np.concatenate((B,A),1)
            if j == 1:
                A = np.array(c*signs[i][0])
                A = np.vstack([A,m])
                A = np.vstack([A,c*signs[i][1]])
                B = np.concatenate((B,A),1)
            if j == 2:
                A = np.array(c*signs[i][0])
                A = np.vstack([A,c*signs[i][1]])
                A = np.vstack([A,m])
                B = np.concatenate((B,A),1)
    
        A= A*0
        A[j] = m
        B = np.concatenate((B,A),1)
        
    # We have to get rid of the outliers. We have 3 arrays than share the same vertex
    B = [tuple(row) for row in B.T]
    B = np.unique(B, axis=0)
    return B.T



def projectpoints(K:np.matrix, R:np.matrix,t:np.array,P:np.matrix):
    """
    K : Camera matrix. 
    R: Rotation Matrix.
    t: Translation array.
    P: Coordinate matrix of points in a 3d space.
    Given a matrix P of points in a 3d space, it projects this points into a 2d plane, supposedly the camera plane.
    It returns p3d, the points in an homogenous coordinates and p2d the points into a 2d plane.And the Projection Matrix.
    
    """
    
    # Check that the translation array it's a 2d array with the desired dimensions. To concatenate it later.
    try:
        m,n = t.shape
    except ValueError:
        t = np.expand_dims(t, axis=1)
    if t.shape != (3, 1):
        t = t.T
    # Get the shape of the matrix P 
    m,n = P.shape
    # Obtain number of columns we want to cast. In case is one point or a matrix of points
    if m == 3:
        P = np.vstack([P, np.ones((1,n))])
    else:
        P = np.vstack([P.T, np.ones((1,m))])
        
    # Concatenate the rotation matrix "R" and the tranlation array "t"
    R = np.concatenate((R,t), axis=1)
    # Create the projection matrix
    # Create the projection in 2d in homogenous coordinates (3th row it0s the scale)
    Pm = K@R
    p3d = Pm@P
    
    # Get the 2d coordinates by dividing by the scale
    qx = p3d[0]/p3d[-1]
    qy = p3d[1]/p3d[-1]
    
    # Generate the 2d coordinate matrix of the points captured on 3d and projected into a 2d plane
    p2d = np.concatenate((qx, qy))
    
    return p3d, p2d, Pm




def plot3d(*A, fsize:tuple=(15,15), lim: tuple= None):
    """
    A: Coordinate Matrix.
    fsize: figure size of the plot, by default (15,15)
    lim: limits of the plot, by default not set.
    Function to plot into a 3d space a matrix of coordinates.
    """
    
    
    fig = plt.figure(figsize=fsize)
    ax = fig.add_subplot(121, projection='3d')
    
    if lim:
        max, min = lim[0], lim[1]
        
        ax.set_xlim(min,max)
        ax.set_ylim(min,max)
        ax.set_zlim(min,max)
        
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    for e in A:
        e = np.array(e)
        ax.plot(e[0], e[1], e[2], 'o')




def plot2d(A, fsize:tuple=(6,6), lim: tuple= None):
    """
    A: Coordinate Matrix.
    fsize: figure size of the plot, by default (6,6)
    lim: limits of the plot, by default not set.
    Function to plot into a 2d space a matrix of coordinates.
    """
    
    fig = plt.figure(figsize=fsize)
    ax = plt.subplot()
    
    if lim:
        min, max = lim[0], lim[1]
        
        ax.set_xlim(min,max)
        ax.set_ylim(min,max)
        
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    A = np.array(A)
    ax.plot(A[0], A[1], 'o')
    


def crossOp(v):
    """
    v: Vector. The dimensions doesn't matter. Must be a 3d vector
    Cv : It's the cross operation between the vector and itself. It follows the skew-symmetric matrix form 
    """
    # Check the vector has appropiate form, we want a (3,) 
    if v.shape == (3,1):
        v = np.squeeze(v, axis=1)
    elif v.shape == (1,3):
        v = np.squeeze(v, axis=0)
    else:
        pass
            
    Cv = np.matrix(np.array([[0, -v[2], v[1]],[v[2], 0, -v[0]], [-v[1], v[0], 0]]))
    return Cv

def squeezdim(v):
    """
    Function that squeezes a vetor into a the form (n,)
    v: vector of dimension n.     
    """
    try:
        m,n = v.shape
    except ValueError:
        return v
    if m>n: 
        v = np.squeeze(v, axis=1)
    else:
        v = np.squeeze(v, axis=0)
    return v


def normalize(P, scale:int=None, points:int=None):
    """
    Given a set of points, Estimates the T matrix which standarizes these points. The given points have to be InHomogenous 
    form and in the form of (dims, points). 
    Being dims the dimensions of the point its given. 2D points = 2, 3D points = 3.
    It follows the following equation:
    
    z = (x-mu_x)/std_x
    
    Can work with all dimensions.
    """
    m,n = P.shape
    if m != 3 and m != 2:
        print('Check the dimensions of your set of points. It should be in inhomogenous form, 2D or 3D points. \nThis is'
              '[sx, sy, z] or [X, Y, Z], which means shapes (2,n) or (3,n). \nThe shape of the given set of points is: \n',
             P.shape)
        return False 
    # calculate the means and unpack them in lists
    *mus, = np.mean(P, axis=1)
    *std, = np.std(P, axis=1)
    # Transform lists into arryas so we can operate with them 
    mus = np.array(mus)
    std = np.array(std)
    # Create the T matrix
    T = np.column_stack([np.diag(1/std),-mus/std])
    # If we want to perserve the scale:
    if scale:
        f = np.zeros((T.shape[1]))
        f[-1] = 1
        T = np.vstack((T,f))
    if points:
        # (3,n)
        points = T@np.vstack((P,np.ones((P.shape[1]))))
        return points, T
    return T

def matrix_B(q, Q):
    """
    Creates a B matrix applying Kroneker product. 
        -1st. Corssproduct on vector q in itslef (qxq):  Uses the approach of a skew symetric matrix.
        -2nd. Krokener product: B = Qi⊗[qi]x
        
    We have to option for estimate B: 
        - Estimate the Homography matrix:
            We want to relate a set of points on 2D with a set of points in 2D. 
            q = [x, y]
            Q = [x, y]
        - Estimate the Projection Matrix:
            We want to calculate the projection matrix given a set of points in 3D and it's projections into a 2D plane 
            q = [x, y]
            Q = [X, Y, Z]
    
    In both cases the points MUST be in Inhomogenous form. 
    """
    
    # Squeez dimesions of q to apply Cross Operation 
    q = squeezdim(q)
    # Transform into homgenous
    q = np.append(q, 1)
    # Create the Skeweness matrix [p1]x
    q = crossOp(q)
    
    # Squeeze vector Q
    Q = squeezdim(Q)
    # Transform into homgenous
    Q = np.append(Q, 1)
    
    # Appply Kroneker Product
    for i in range(len(Q)):
        # Cjeck if the first element it's 0 to create matrix
        if not i:
            B = Q[i] * q
        else:  # If it's not the first element just concatenate the matrices
            B = np.concatenate((B, Q[i]*q),1)
    return B


def projection_matrix(Q, P):
    """
    Given a set of 3D points and it's projections into a 2D plane. We estimate the projection Matrix P.
    It's estimated calculating the B Matrix and solving with singular value decomposition. Where the eigenvector 
    associated with the minimum singular value it's the estimation of the Projection Matrix.
    
    Q: A matrix of q points in homogenous form. where q are the projections into a plane of 3D points. [sx, sy, s]
    Q -> (3,n)
    P: A matrix of p points. Where P are 3D points of the form [X, Y, Z] or [X, Y, Z, 1]
    P-> (3,n), (4,n)
    PrM: An estimation of the projection matrix.
    
    """
    # Transform into arrays
    Q = np.array(Q)
    P = np.array(P)
    # Check if Q it's only one point or a matrix. 
    try:
        mq,nq = Q.shape # We want (3,n). 
        if mq != 3:  # The dimensions are switched. Could be (1,3), (8,3)
            Q = Q.T
            mq,_ = Q.shape 
            if mq != 3:
                print('The vector q Must be in Homogenous form: [sx, sy, s], and it is: ')
                print(Q)
                return False
        elif mq == nq:  # (3,3) We can't know which is the direction 
            print('Make sure that the rows of the matrix are the axis, i.e., (3,n). Being n the number of points')
            print(Q)
        else:
            pass
    except ValueError:
        # Check if the point it's on the form of [sx, sy, s]
        mq, = Q.shape
        if mq != 3:
            # The array doesn't have 3 dimensions. 
            raise ValueError('The vector q Must be in Homogenous form: [sx, sy, s], and it is: ', Q)
        else:
            # # The array has dimensions (3,). correct, add one dim so (3,1)
            Q = np.expand_dims(Q, axis = 1)
            
        
    # Check if P it's only one point or a matrix. 
    try:
        m_p,n_p = P.shape # We want (3,n) or (4,n)
        if m_p != 3 and m_p != 4:  # The dimensions are switched. Could be (1,3), (8,3), (1,4), (8,4)
            P = P.T
            m_p,_ = P.shape 
            if m_p != 3 and m_p != 4: 
                # Check the matix has the appropiate dimensions 
                print('The vector p Must be in Homogenous or Inhomogoenous fom: [X,Y,Z] or [X,Y,Z,1], and it is: ')
                print(P)
                return False
        elif m_p == n_p:  # (3,3) or (4,4) We can't know which is the direction 
            print('Make sure that the rows of the matrix are the axis, i.e., (3,n) or (4,n). Being n the number of points')
            print(P)
        else:
        # The matix has the appropiate dimensions
            pass
    except ValueError:
        # Check if the point it's on the form of [sx, sy, s]
        m_p, = P.shape
        if m_p != 3 and m_p != 4:
            # The array doesn't have 3 dimensions. 
            raise ValueError('The vector q Must be in Homogenous form: [sx, sy, s], and it is: ', P)
        else:
            # The array has dimensions (3,) or (4,1). correct, add one di so (3,1) or (4,1)
            P = np.expand_dims(P, axis = 1)

    
    # Check that both matrices have the same number of points 
    if Q.shape[1] != P.shape[1]:
        print('Not the same amount of points Q has: ', Q.shape, 'and P: ', P.shape)
        return False 
    
    # Create Matrix B
    # p⊗[q]_x (3,1) or (4,1)
    _,n = P.shape
    for i in range(n):
        # check if it's the first time we call the function 
        if not i:
            B = matrix_B(Q[:,i],P[:,i])
        else:
            B = np.concatenate([B, matrix_B(Q[:,i], P[:,i])])
    
    # Apply singular value decomposition 
    scores, s, eigV = np.linalg.svd(B)
    # Find the min eigenvalue, eigenvalue = s**2, min(s) = min (eigenValue)
    idx = np.where(s==min(s))
    
    PrM = eigV[idx].reshape((3,4),order='F')
    PrM = PrM/PrM[-1,-1]

    return PrM



def hest(Q1, Q2):
    """
    Estimation of the homography matrix. 
    Following the equation: q1 x H*q2:
        q1: 2D points in inhomogeneous form. [x, y].T 
        q2: 2D points in inhomogeneous form. [x, y].T
        H: Homography matrix (3,3)
    
    Parameters:
        -Q1 set of 2D points. (2,n)
        -Q2 set of 2D points. (2,n)
    Given a set of points, (min should be 4 points, as the homography has 8 degrees of freedom) calculates the B matrix
    and applies singular value decomposition (SVD) to estimate the homography matrix H.
       
    """
    
    # Check the dimensions and if the same poits were given.
    Q1 = np.array(Q1)
    Q2 = np.array(Q2)
    if Q1.shape[0] != 2:
        print("aa The dimensions of Q1 are wrong. it should be (2,n) and it's", Q1.shape)
        return
    elif Q2.shape[0] != 2:
        print("The dimensions of Q2 are wrong. it should be (2,n) and it's", Q2.shape)
        return
    # Check the same number of pooints were given 
    if Q1.shape != Q2.shape:
        print("Different number of points were given, check that Q1 and Q2 have same dimensions")
        return
    
    # Transform into homogenous form.
    n = Q1.shape[1]
#     hom = np.ones(n)
#     Q1 = np.vstack([Q1, hom])
#     Q2 = np.vstack([Q2, hom])
    
    # Calculate B Matrix 
    for i in range(n):
        if not i:
            B = matrix_B(Q1[:,i],Q2[:,i])
        else:
            B = np.concatenate([B, matrix_B(Q1[:,i], Q2[:,i])])
    
    # Estimation of H: SVD
    _, s, eigenvec = np.linalg.svd(B)
    # Find the min Eigenvalue
    idx = np.where(s==min(s))
    # Select the eigenvector correspondent to the eigenvalue
    H = eigenvec[idx]
    # Reshape the array into a matrix form 
    H = H.reshape((3,3), order='F')
    # Scale it 
    H = H/H[-1,-1]

    return H

def hom(Q):
    """
    Function that transforms into homogenous form. 
    Q has to be in the form of (2,n) or (3, n)
    Returns the matrix Q but with an extra row of ones
    """
    n = Q.shape[1]
    Q = np.vstack([Q, np.ones(n)])
    return Q

def inhom(Q):
    """
    Return Q in inhomogenous form 
    Given Q divides by the scale:
    [sx, sy, s]/s = [x, y, 1] and we return [x, y]
    In case of a 3D point as the scale is 1 we divide by one and return everything but the last column.
    """
    s = Q[-1,:]
    Q = (Q/s)[:-1,:]
    return Q


def projectMat(K, R, t):
    """
    Given the Camera Matrix, the Rotation Matrix and the translation vector. Creates the Projection Matrix.
    P = K*[R t]
    K-> Camera Matrix
    R-> Rotation Matrix
    t-> Translation vector
    
    P-> Projection Matrix.
    """
    
    if K.shape != (3,3):
        print('Check your Camera Matrix, it should have dimensions (3,3) and it has: ', K.shape)
    if R.shape != (3,3):
        print('Check your Rotatio Matrix, it should have dimensions (3,3) and it has: ', R.shape)
    if t.shape == (3,1):
        pass 
    else:
        t = squeezdim(t).reshape(-1,1)
    
    Rt = np.column_stack([R, t])
    P = K@Rt
    
    return P







## TRIANGULATION 

def matrix_B_tirnagulation(q, P):
    """
    Creates  B matrix for Triangulation. Follows the next equation:
        qh = P*Q
    Where qh is a 2D point in homoogenous form [sx, sy, s], P is the Projection Matrix and Q it's a 3D point in homogenous
    coordinates [X, Y, Z, 1].
    We can rearange it as:
        s * [x, y].T = [p(1)*Q, p(2)*Q].T
    Where p(i) = row vector of P. P(1) = P[0,:] (1,3) and the scale s = p(3)*Q:
        0 = [p(3)*x-p(1), p(3)*y-p(2)].T*Q
    
    PARAMETERS
    q: Point in homogenous fom [sx, sy, s] -> (3,1)
    P: projection matrix K*[R t] -> (3,4)
    RETURNS
    B: B Matrix (2,4)
    """

    q = squeezdim(q)
    
    if q.shape != (3,):
        print('q is not in Homogenous form: ', q.shape)
        return False
    # Transform into inhomogeous form [x, y, 1] in case it's not in homogenous form. 
    q = q/q[-1]
    
    
    #  Extract parameters 
    x = q[0]
    y = q[1]
    p_1 = squeezdim(P[0,:])
    p_2 = squeezdim(P[1,:])
    p_3 = squeezdim(P[2,:])
    # Create B
    B = np.array([(p_3*x - p_1), (p_3*y-p_2)])
    
    return B



def triangulate(qs:list, P:list):
    """
    Find points in 3D given a set of points in 2D and the Cameras' Projection Matrices used to obtain those 2D points.
    Using Linear algorithm.
    Create a Matrix B fof (n*2,4) where n is the number of cameras. Check Triabngulate_B for further doc.
    One We have B we apply singular value decomposition.
        arg min||B*Q||_2
    Wehere we impose the constrain of:
        ||Q||_2 = 1
    To specify that Q hasn't to be 0.
    
   PARAMETERS
    qs: List of points in 2D points in homogenous from. Scale Matters. 
    P: List of Projection Matrix.
    
    RETURNS
    Q: Estimated point in 3D in Homogenous coordinates [X, Y, Z, 1]
    """
    
    # Check The same samount of points and cameras.
    if len(qs) != len(P):
        print('Different number of 2D points and Projection matrix.')
        return False
    
    # Get the number of cameras
    n = len(P)
    # Set step for the B matrix 
    step = 2

    # Initializate B
    B = np.zeros((step*n, 4))
    
    for p,q,i in zip(P,qs, range(0,n*step,step)):
        B[i:i+step] = matrix_B_tirnagulation(q,p)

    _,s,eigVec = np.linalg.svd(B)
    # Find Min 
    idx = np.where(s==min(s))
    # Select the eigenvector correspondent to the eigenvalue
    Q = eigVec[idx].T
    # Normalize 
    Q = Q/Q[-1]
    
    return  Q