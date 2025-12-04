def planeCross(t, y):
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
    position = y[37]  # Python is 0-indexed, MATLAB 38 -> y[37]
    # Additional attributes for solve_ivp event
    planeCross.terminal = True   # Halt integration at zero crossing    
    planeCross.direction = 0     # Zero crossing from either direction
    return position

