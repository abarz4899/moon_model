import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, RK45
from CRTBP_dyn import CRTBP
from CRTBP_x0_def import halo
from astroConstants import astroConstants
from frame_transformation import cr3bp2moon_inertial, cr3bp2LL_moon
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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

#xx0 = np.array([1.06092, 0, -0.07349, 0, 0.3415, 0])
#T0 = 3.22473  # Halo with ~14 days period

xx0 = np.array([0.8944016860, 0, 0, 0, 0.4737129104, 0])
T0 = 1.3868167909756  # DRO with ~6 days period

#xx0 = np.array([3.5866379329881432E-1,	-3.4741501576342322E-23,	3.0245229067336969E-23,	1.2283366687296770E-12,	1.7230868937505937E+0,	2.1951256201511247E-23])
#T0 = 6.1814452901176562E+0  # DRO with ~27.4 days period from NASA catalog

#xx0 = np.array([1.0382, 0, -0.1914, 0, -0.1359, 0])
#T0 = 7.5 * 24 * 3600 / TU  # NRHO with ~7.5 days period https://www.researchgate.net/publication/374542949_Summary_of_a_Phase_0A_Study_Report_for_a_Communication_Satellite_Constellation_for_DIANA_Lunar_Infrastructure

xx0, T0 = halo(xx0, T0, mu)

# -------------------------------
# Solve CRTBP
# -------------------------------

sol = solve_ivp(
    dyn,
    [0, 3*T0],
    xx0,
    method='RK45',
    rtol=rtol,
    atol=atol
)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(sol.y[0]*LU, sol.y[1]*LU, sol.y[2]*LU, 'r', linewidth=2, label='Orbit')

ax.scatter(moon_coord[0]*LU, moon_coord[1]*LU, moon_coord[2]*LU, s=100, c='b', marker='o', label='Moon')

ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')
#ax.set_zlim(-1, 1)
ax.legend()
ax.grid(True)
plt.show(block=False)
plt.pause(0.001)


""" fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot(sol.y[0]*LU, sol.y[1]*LU, 'r', linewidth=2, label='Orbit')

ax.scatter(moon_coord[0]*LU, moon_coord[1]*LU, s=100, c='b', marker='o', label='Moon')

ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
#ax.set_zlim(-1, 1)
ax.legend()
ax.grid(True)
plt.show(block=False)
plt.pause(0.001) """

# -------------------------------
# Transform to Moon inertial frame
# -------------------------------

xx_moon = cr3bp2moon_inertial(sol.y, sol.t, {'mu': mu, 'LU': LU, 'TU': TU})

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xx_moon[0,:], xx_moon[1,:], xx_moon[2,:], 'r', linewidth=2, label='Orbit')

ax.scatter(0, 0, 0, s=100, c='b', marker='o', label='Moon')

ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')
ax.legend()
ax.grid(True)
plt.show(block=False)
plt.pause(0.001)

# -------------------------------
# Get ground tracks
# -------------------------------

xx_LLmoon = cr3bp2LL_moon(sol.y, {'mu': mu, 'LU': LU, 'TU': TU})

plt.figure()
plt.plot(sol.t * TU / 86400, xx_LLmoon[0,:]*LU) # time in days
plt.title('Radial distance from Moon center')
plt.xlabel('Time [days]')
plt.ylabel('Radial distance [km]')
plt.grid(True)
plt.show(block=False)
plt.pause(0.001)

img = mpimg.imread('moon_colormap_1500.jpg')

plt.figure(figsize=(10, 5))
plt.imshow(
    img,
    extent=[-180, 180, -90, 90],   # [lon_min, lon_max, lat_min, lat_max]
    origin='upper'
)

plt.scatter(xx_LLmoon[2,:], xx_LLmoon[1,:], color = 'red', s=1, label="Ground Tracks")
plt.plot(xx_LLmoon[2,0], xx_LLmoon[1,0], marker = 'D', color = 'blue', markersize=8, label="Start")
plt.plot(xx_LLmoon[2,-1], xx_LLmoon[1,-1], marker = 'D', color = 'green', markersize=8, label="End")
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.xlabel('Longitude [deg]')
plt.ylabel('Latitude [deg]')
plt.title('Ground Tracks')
plt.grid(True)
plt.legend()
#plt.show(block=False)
plt.show()
plt.pause(0.001)

""" from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris
from astropy.coordinates import get_body_barycentric

t = Time('2025-01-01T00:00:00')

with solar_system_ephemeris.set('de432s'):
    moon_bary = get_body_barycentric('moon', t)
    sun_bary = get_body_barycentric('sun', t)

moon2sun = sun_bary - moon_bary
print("Moon to Sun distance [km]: ", moon2sun.norm().to_value('km')) """