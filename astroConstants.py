import numpy as np
import warnings

def astroConstants(in_ids):
    """
    Returns astrodynamic-related physical constants.

    Parameters
    ----------
    in_ids : list or ndarray
        List of constant identifiers to retrieve.

    Returns
    -------
    out : ndarray
        Array of constant values corresponding to the input identifiers.
    """

    # Dictionary mapping identifiers to constants
    constants = {
        1: 6.67259e-20,          # Universal gravity constant G [km^3/(kg*s^2)]
        2: 149597870.691,        # Astronomical Unit AU [km]
        3: 6.955e5,              # Sun mean radius [km]
        4: 1.32712440017987e11,  # Sun GM [km^3/s^2]
        5: 299792.458,           # Speed of light [km/s]
        6: 9.80665,              # Standard gravity [m/s^2]
        7: 384400,               # Earth-Moon distance [km]
        8: 84381.412/3600*np.pi/180,  # Obliquity J2000 [rad]
        9: 0.1082626925638815e-2,     # Earth J2
        11: 2.203208e4,          # Mercury GM [km^3/s^2]
        12: 3.24858599e5,        # Venus GM
        13: 3.98600433e5,        # Earth GM
        14: 4.2828314e4,         # Mars GM
        15: 1.26712767863e8,     # Jupiter GM
        16: 3.79406260630e7,     # Saturn GM
        17: 5.79454900700e6,     # Uranus GM
        18: 6.83653406400e6,     # Neptune GM
        19: 9.81601000000e2,     # Pluto GM
        20: 4902.801,            # Moon GM
        21: 2439.7,              # Mercury radius [km]
        22: 6051.8,              # Venus radius
        23: 6371.01,             # Earth radius
        24: 3389.9,              # Mars radius
        25: 69911,                # Jupiter radius
        26: 58232,                # Saturn radius
        27: 25362,                # Uranus radius
        28: 24624,                # Neptune radius
        29: 1151,                 # Pluto radius
        30: 1738.0,               # Moon radius
        31: 1367,                 # Solar flux [W/m^2 at 1 AU]
        32: 365.25                # Days in a Julian year
    }

    out = np.zeros(len(in_ids))

    for i, key in enumerate(in_ids):
        if key in constants:
            out[i] = constants[key]
        else:
            warnings.warn(f"Constant identifier {key} is not defined!", UserWarning)
            out[i] = 0.0

    return out
