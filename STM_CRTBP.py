import numpy as np
from scipy.integrate import solve_ivp
from STM_CRTBP_dy import dynamicsSTM_CR3BP

def stateTransCR3BP(tspan, x0, mu, options):
    """
    Computes the State Transition Matrix (STM) for the CR3BP along a given trajectory.

    Parameters
    ----------
    tspan : list or ndarray
        [t0, tf] time span [s]
    x0 : ndarray
        6-element initial state vector
    mu : float
        CR3BP mass parameter
    rtol : float
        Relative tolerance for ODE solver
    atol : float
        Absolute tolerance for ODE solver

    Returns
    -------
    STM : ndarray
        6x6 State Transition Matrix at final time
    state : ndarray
        6-element state vector at final time
    T0 : float
        Twice the maximum of the time array (analogous to MATLAB code)
    """

    # Initial STM (identity) and combined state vector
    y0 = np.zeros(42)
    I6 = np.eye(6)
    y0[:36] = I6.flatten('F')  # Column-major flattening
    y0[36:] = x0

    def planeCross_test(t, y):
        """
        Event function to detect when a specific state component crosses zero.
        
        Parameters
        ----------
        t : float
            Current time (required by solve_ivp)
        y : ndarray
            Current state vector (42 elements: 36 STM + 6 states)

        Returns
        -------
        position : float
            Value to monitor for zero crossing
        """
        if t == 0:
            position = 1.0  # Avoid triggering at t=0
        else:
            position = y[37]
        # Additional attributes for solve_ivp event
        return position
    
    planeCross_test.terminal = True   # Halt integration at zero crossing    
    planeCross_test.direction = 0     # Zero crossing from either direction

    # Integrate the full 42-state system
    sol = solve_ivp(
        fun=lambda t, y: dynamicsSTM_CR3BP(t, y, mu),
        t_span=tspan,
        y0=y0,
        method='DOP853',
        rtol=options["rtol"],
        atol=options["atol"],
        events=planeCross_test
    )

    # Extract STM and state at final time
    sol_y = sol.y.T

    y_final = sol_y[-1, 0:36]
    STM = y_final.reshape((6,6))   # default order='C'
    state = sol_y[-1, 36:]

    # T0 is twice the maximum integration time (as in MATLAB code)
    T0 = 2 * max(sol.t_events[0])

    return STM, state, T0
