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
    if t.shape != (1,3):
        t = np.expand_dims(t, axis=1)
    # Get the shape of the matrix P 
    m,n = P.shape
    # Obtain number of columns we want to cast. In case is one point or a matrix of points
    if m == 3:
        P = np.vstack([P, np.ones((1,n))])
   
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




def plot3d(A, fsize:tuple=(15,15), lim: tuple= None):
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
    A = np.array(A)
    
    ax.plot(A[0], A[1], A[2], 'o')




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