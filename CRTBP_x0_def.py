import numpy as np
from STM_CRTBP import stateTransCR3BP
from planeCross import planeCross

def halo(xx0, T0_in, mu, max_iter=25, eps=1e-10):
    """
    Refine initial conditions for a periodic CR3BP orbit (Halo, Lyapunov, DRO, etc.)

    Parameters
    ----------
    xx0 : ndarray
        Initial guess for state vector [6]
    T0_in : float
        Initial guess for period
    mu : float
        CR3BP mass parameter
    max_iter : int
        Maximum number of Newton-Raphson iterations
    eps : float
        Convergence tolerance

    Returns
    -------
    xx0_out : ndarray
        Corrected initial state [6]
    T0_out : float
        Corrected period
    """

    x0 = xx0.copy()
    T0 = T0_in
    Fnorm = 1.0
    iter_count = 0

    while Fnorm > eps and iter_count < max_iter:
        tspan = [0, T0]

        options = {
            'rtol': 1e-13,
            'atol': 1e-20,
            'events': [planeCross]
        }

        STM, state, T0 = stateTransCR3BP(tspan, x0, mu, options)


        x = state
        F = np.array([x[1], x[3], x[5]])  # MATLAB indices: [2,4,6]

        Fnorm = np.linalg.norm(F)

        # Compute intermediate terms
        r1 = np.sqrt((x[0] + mu)**2 + x[1]**2 + x[2]**2)
        r2 = np.sqrt((x[0] + mu - 1)**2 + x[1]**2 + x[2]**2)
        ddx = x[0] - (1 - mu)*(x[0] + mu)/r1**3 - mu*(x[0] + mu - 1)/r2**3 + 2*x[4]
        ddz = ((mu - 1)/r1**3 - mu/r2**3) * x[2]

        # Construct DF matrix (4x3 in MATLAB, careful with shapes)
        DF = np.array([
            [STM[1,0], STM[1,4], STM[1,5], x[4]],
            [STM[3,0], STM[3,4], STM[3,5], ddx],
            [STM[5,0], STM[5,4], STM[5,5], ddz]
        ])

        # Newton-Raphson correction
        Xold = np.array([x0[0], x0[4], x0[5], T0])
        # Solve least-squares problem (DF * DF^T) \ F
        invDF = np.linalg.solve(DF @ DF.T, F)
        Xnew = Xold - DF.T @ invDF

        # Update variables
        x0[0] = Xnew[0]
        x0[4] = Xnew[1]
        x0[5] = Xnew[2]
        T0 = Xnew[3]

        iter_count += 1

    xx0_out = x0[:6]
    T0_out = T0

    return xx0_out, T0_out
