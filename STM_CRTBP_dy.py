import numpy as np

def dynamicsSTM_CR3BP(t, y, mu):
    """
    Computes the dynamics and STM derivatives for the CR3BP.

    Parameters
    ----------
    t : float
        Time (not used, included for compatibility with ODE solvers)
    y : ndarray
        42-element vector containing STM (first 36 elements) and state (last 6 elements)
    mu : float
        Mass parameter of the CR3BP

    Returns
    -------
    dy : ndarray
        42-element derivative vector
    """

    # Split STM and state
    s = y[:36]        # STM elements
    x = y[36:42]      # State elements

    # Zero and identity matrices
    O = np.zeros((3,3))
    I = np.eye(3)

    #---------------- 1st derivatives of g(r) ----------------#
    def dg1dx(x1, y1, z1):
        return (1 - (1-mu) * (((x1+mu)**2 + y1**2 + z1**2)**(-3/2) - 
                              3*(x1+mu)**2*((x1+mu)**2 + y1**2 + z1**2)**(-5/2)) -
                mu * (((x1+mu-1)**2 + y1**2 + z1**2)**(-3/2) - 
                      3*(x1+mu-1)**2*((x1+mu-1)**2 + y1**2 + z1**2)**(-5/2)))

    def dg1dy(x1, y1, z1):
        return (3*(1-mu)*y1*(x1+mu)*((x1+mu)**2 + y1**2 + z1**2)**(-5/2) +
                3*mu*y1*(x1+mu-1)*((x1+mu-1)**2 + y1**2 + z1**2)**(-5/2))

    def dg1dz(x1, y1, z1):
        return (3*(1-mu)*z1*(x1+mu)*((x1+mu)**2 + y1**2 + z1**2)**(-5/2) +
                3*mu*z1*(x1+mu-1)*((x1+mu-1)**2 + y1**2 + z1**2)**(-5/2))

    def dg2dx(x1, y1, z1):
        return dg1dy(x1, y1, z1)

    def dg2dy(x1, y1, z1):
        return (1 - (1-mu) * (((x1+mu)**2 + y1**2 + z1**2)**(-3/2) - 
                              3*y1**2*((x1+mu)**2 + y1**2 + z1**2)**(-5/2)) -
                mu * (((x1+mu-1)**2 + y1**2 + z1**2)**(-3/2) - 
                      3*y1**2*((x1+mu-1)**2 + y1**2 + z1**2)**(-5/2)))

    def dg2dz(x1, y1, z1):
        return (3*(1-mu)*z1*y1*((x1+mu)**2 + y1**2 + z1**2)**(-5/2) +
                3*mu*z1*y1*((x1+mu-1)**2 + y1**2 + z1**2)**(-5/2))

    def dg3dx(x1, y1, z1):
        return dg1dz(x1, y1, z1)

    def dg3dy(x1, y1, z1):
        return dg2dz(x1, y1, z1)

    def dg3dz(x1, y1, z1):
        return (- (1-mu) * (((x1+mu)**2 + y1**2 + z1**2)**(-3/2) - 
                             3*z1**2*((x1+mu)**2 + y1**2 + z1**2)**(-5/2)) -
                mu * (((x1+mu-1)**2 + y1**2 + z1**2)**(-3/2) - 
                      3*z1**2*((x1+mu-1)**2 + y1**2 + z1**2)**(-5/2)))

    # Submatrices
    def G(x1, y1, z1):
        return np.array([[dg1dx(x1,y1,z1), dg1dy(x1,y1,z1), dg1dz(x1,y1,z1)],
                         [dg2dx(x1,y1,z1), dg2dy(x1,y1,z1), dg2dz(x1,y1,z1)],
                         [dg3dx(x1,y1,z1), dg3dy(x1,y1,z1), dg3dz(x1,y1,z1)]])

    H = np.array([[0, 2, 0],
                  [-2, 0, 0],
                  [0, 0, 0]])

    # State Matrix
    def A(x1, y1, z1):
        top = np.hstack((O, I))
        bottom = np.hstack((G(x1,y1,z1), H))
        return np.vstack((top, bottom))

    # Reshape STM
    STM = s.reshape((6,6)).T

    # Compute STM derivative
    A_val = A(x[0], x[1], x[2])
    dSTM = A_val @ STM
    dstm = dSTM.ravel(order='C')

    #dstm = dSTM.T.flatten()

    # Two-body vector field derivatives
    r1 = np.sqrt((x[0]+mu)**2 + x[1]**2 + x[2]**2)
    r2 = np.sqrt((x[0]+mu-1)**2 + x[1]**2 + x[2]**2)

    dx = np.array([
        x[3],
        x[4],
        x[5],
        x[0] + 2*x[4] - ((1-mu)*(x[0]+mu))/r1**3 - mu*(x[0]-(1-mu))/r2**3,
        x[1] - 2*x[3] - ((1-mu)*x[1])/r1**3 - mu*x[1]/r2**3,
        -((1-mu)*x[2])/r1**3 - (mu*x[2])/r2**3
    ])

    # Combine STM and state derivatives
    dy = np.concatenate([dstm, dx])
    #print(dy)
    return dy
