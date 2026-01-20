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
GROUND_TRACKS = True
MOON_INERTIAL = False

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


# L1 Lyapunov

# L2 Lyapunov
#xx0 = np.array([1.0188102500504621E+0, -4.6220965452066168E-28, 1.6424314718601130E-34, -2.6241323694828198E-14, 8.6459249758658030E-1, -5.0273542395967592E-33])
#T0 = 4.7094975197876332E+0
# Halo L1
#xx0 = np.array([8.2344865129852762E-1, 5.2829529136685427E-28, 3.2463478024080249E-2, 3.9055875236491744E-16, 1.4215181413480191E-1, 8.8847624615278027E-15])
#T0 = 2.7499366348673266E+0
#Vertical L1 interesting GT?
#xx0 = np.array([8.8312157578340489E-1, 1.6027903642003450E-23, -2.3876674528695140E-13, -3.2728974470972733E-13, -1.8591243252026587E-1, -8.6852602011187485E-1])
#T0 = 6.0715755023809246E+0
#Axial L1
#xx0 = np.array([7.8585318872543597E-1, 1.3134717035274563E-28, -2.8138698969165921E-15, -2.3243583760117189E-15, 4.2634078505235912E-1, 1.1572253769504327E-1])
#T0 = 3.9605804173088153E+0
# Butterfly L1
#xx0 = np.array([9.0449918625038561E-1, -9.5514319778147684E-26, 1.4343748441078497E-1, -6.6139894504539067E-15, -5.6148214907237454E-2, 1.7468706047148683E-14])
#T0 = 3.8057023404994972E+0
#Resonant 31
xx0 = np.array([2.8307943366760674E-1, 5.6922683065964986E-20, -3.3879965447437840E-24, 9.2726682489108495E-14, 1.8586404724644123E+0, 3.2464040587919200E-23])
T0 = 6.5088980791892030E+0
#Moon centered orbits are all flat on the Equator of the Moon

#xx0, T0 = halo(xx0, T0, mu)
x0_km = xx0[0:3] * LU
v0_kms = xx0[3:6] * LU / TU
T0_sec = T0 * TU
print("Initial position [km]: ", x0_km)
print("Initial velocity [km/s]: ", v0_kms)
print("Orbit period [s]: ", T0_sec)

# -------------------------------
# Solve CRTBP
# -------------------------------
prop_time = T0  

sol = solve_ivp(
    dyn,
    [0, prop_time],
    xx0,
    method='RK45',
    rtol=rtol,
    atol=atol
)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol.y[0]*LU, sol.y[1]*LU, sol.y[2]*LU, 'r', linewidth=2, label='Orbit')
ax.scatter(moon_coord[0]*LU, moon_coord[1]*LU, moon_coord[2]*LU, s=50, c='b', marker='o', label='Moon')
ax.set_xlabel('x [km]')
ax.set_ylabel('y [km]')
ax.set_zlabel('z [km]')
plt.title('CRTBP Orbit')
#ax.set_zlim(-1, 1)
ax.legend()
ax.grid(True)
plt.show(block=False)
plt.pause(0.001)
#plt.show()


# -------------------------------
# Transform to Moon inertial frame
# -------------------------------
if MOON_INERTIAL == True:
    xx_moon = cr3bp2moon_inertial(sol.y, sol.t, {'mu': mu, 'LU': LU, 'TU': TU})

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xx_moon[0,:], xx_moon[1,:], xx_moon[2,:], 'r', linewidth=2, label='Orbit')

    ax.scatter(0, 0, 0, s=100, c='b', marker='o', label='Moon')

    ax.set_xlabel('x [km]')
    ax.set_ylabel('y [km]')
    ax.set_zlabel('z [km]')
    plt.title('Orbit in Moon Inertial Frame')
    ax.legend()
    ax.grid(True)
    plt.show(block=False)
    plt.pause(0.001)

# -------------------------------
# Get ground tracks
# -------------------------------
if GROUND_TRACKS == True:
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