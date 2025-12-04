import numpy as np

def CRTBP(t, x_vect, mu3B):
    """
    Computes the equations of motion for the Circular Restricted Three-Body Problem (CRTBP).

    Parameters
    ----------
    t : float
        Time (required for ODE solvers, not used here)
    x_vect : ndarray
        6-element vector containing position [x, y, z] and velocity [vx, vy, vz]
    mu3B : float
        Mass parameter of the three-body system

    Returns
    -------
    dx_vect : ndarray
        6-element derivative vector [dx, dy, dz, ddx, ddy, ddz]
    """

    dx_vect = np.zeros(6)

    x, y, z = x_vect[0], x_vect[1], x_vect[2]
    vx, vy, vz = x_vect[3], x_vect[4], x_vect[5]

    r1 = np.sqrt((x + mu3B)**2 + y**2 + z**2)
    r2 = np.sqrt((x + mu3B - 1)**2 + y**2 + z**2)

    # Equations of motion
    dx_vect[0] = vx
    dx_vect[1] = vy
    dx_vect[2] = vz
    dx_vect[3] = x - (1 - mu3B)*(x + mu3B)/r1**3 - mu3B*(x + mu3B - 1)/r2**3 + 2*vy
    dx_vect[4] = (1 - (1 - mu3B)/r1**3 - mu3B/r2**3)*y - 2*vx
    dx_vect[5] = ((mu3B - 1)/r1**3 - mu3B/r2**3)*z

    return dx_vect
