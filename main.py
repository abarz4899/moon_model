import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from CRTBP_dyn import CRTBP
from CRTBP_x0_def import halo
from astroConstants import astroConstants
from frame_transformation import cr3bp2moon_inertial

# -------------------------------
# Initialization
# -------------------------------

mu = 0.012150584269940
LU = astroConstants([7])[0]  # Transform from LU to km
TU = 27.321661 * 24 * 3600 / (2 * np.pi)  # Transform from TU to seconds

earth_coord = np.array([-mu, 0, 0])
moon_coord = np.array([1 - mu, 0, 0])

# Dynamics function
dyn = lambda t, x_v: CRTBP(t, x_v, mu)

# Tolerances for ODE solver
rtol = 1e-13
atol = 1e-20

# -------------------------------
# Find IC orbit 1 (Halo orbit)
# -------------------------------
#xx01 = np.array([1.06092, 0, -0.07349, 0, 0.3415, 0])
# xx01 = np.array([1.060, 0, -0.0734, 0, 0.341, 1e-4])
# T01 = 3.22473  # Halo with ~14 days period

# xx01_ok, T01_ok = halo(xx01, T01, mu)

# sol01 = solve_ivp(
#     dyn,
#     [0, T01_ok],
#     xx01_ok,
#     method='RK45',  # RK45 is closest to ode113 in SciPy
#     rtol=rtol,
#     atol=atol
# )

# -------------------------------
# Find IC orbit 2 (DRO orbit)
# -------------------------------
#xx02 = np.array([0.8944016860, 0, 0, 0, 0.4737129104, 0])
#T02 = 1.3868167909756  # DRO with ~6 days period

xx02 = np.array([3.5866379329881432E-1,	-3.4741501576342322E-23,	3.0245229067336969E-23,	1.2283366687296770E-12,	1.7230868937505937E+0,	2.1951256201511247E-23])
T02 = 6.1814452901176562E+0  # DRO with ~27.4 days period

xx02_ok, T02_ok = halo(xx02, T02, mu)

sol02 = solve_ivp(
    dyn,
    [0, T02_ok],
    xx02_ok,
    method='RK45',
    rtol=rtol,
    atol=atol
)

# -------------------------------
# Plotting
# -------------------------------
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#ax.plot(sol01.y[0], sol01.y[1], sol01.y[2], 'k', linewidth=2, label='Halo')
ax.plot(sol02.y[0], sol02.y[1], sol02.y[2], 'r', linewidth=2, label='DRO')

ax.scatter(moon_coord[0], moon_coord[1], moon_coord[2], s=100, c='b', marker='o', label='Moon')

ax.set_xlabel('x [LU]')
ax.set_ylabel('y [LU]')
ax.set_zlabel('z [LU]')
ax.legend()
ax.grid(True)
plt.show()
#plt.show(block=False)

xx02_moon = cr3bp2moon_inertial(sol02.y, sol02.t, {'mu': mu, 'LU': LU, 'TU': TU})

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#ax.plot(sol01.y[0], sol01.y[1], sol01.y[2], 'k', linewidth=2, label='Halo')
ax.plot(xx02_moon[0,:], xx02_moon[1,:], xx02_moon[2,:], 'r', linewidth=2, label='DRO')

ax.scatter(0, 0, 0, s=100, c='b', marker='o', label='Moon')

ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')
ax.set_zlim(-1, 1)
ax.legend()
ax.grid(True)
plt.show()
#plt.show(block=False)