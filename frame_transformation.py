import numpy as np

def cr3bp2moon_inertial(x_cr3bp, t_v, param):
    #print("Transforming to moon inertial frame...")
    mu = param['mu']
    LU = param['LU']
    TU = param['TU']

    m = x_cr3bp.shape[0]
    n = x_cr3bp.shape[1]
    x_inertial = np.zeros((m, n))

    for i in range(n):
        t = t_v[i]
        pos_cr3bp = x_cr3bp[0:3, i]
        vel_cr3bp = x_cr3bp[3:6, i]

        At = np.array([[np.cos(t), -np.sin(t), 0],
                       [np.sin(t),  np.cos(t), 0],
                       [0,          0,         1]])
        
        J = np.array([[0, 1, 0],
                      [-1, 0, 0],
                      [0, 0, 0]])
        
        translation_matrix = np.array([np.cos(t), np.sin(t), 0.0])
        translation_matrix_dot = np.array([-np.sin(t), np.cos(t), 0.0])

        pos_inertial = At @ pos_cr3bp - (1 - mu) * translation_matrix
        
        vel_inertial = -At @ J @ pos_cr3bp + At @ vel_cr3bp - (1 - mu) * translation_matrix_dot

        x_inertial[0:3, i] = pos_inertial * LU
        x_inertial[3:6, i] = vel_inertial * (LU / TU)
        #print(x_inertial[0:3, i])
        #print(x_inertial[3:6, i])   

    return x_inertial

def cr3bp2LL_moon(x_cr3bp, param):
    mu = param['mu']
    LU = param['LU']
    alpha = np.deg2rad(-1.54)
    cosa = np.cos(alpha)
    sina = np.sin(alpha)

    n = x_cr3bp.shape[1]
    LL_moon = np.zeros((3, n))
    x_moon = np.zeros((3, n))

    for i in range(n):
        pos_cr3bp = x_cr3bp[0:3, i]
        x_moon[0, i] = pos_cr3bp[0]*cosa + pos_cr3bp[2]*sina - (1 - mu)
        x_moon[1, i] = pos_cr3bp[1]
        x_moon[2, i] = -pos_cr3bp[0]*sina + pos_cr3bp[2]*cosa

        x = x_moon[0, i]
        y = x_moon[1, i]
        z = x_moon[2, i]

        LL_moon[0,i] = np.linalg.norm(x_moon[:,i]) #radial distance [LU]
        lat = np.asin(z/LL_moon[0,i])
        LL_moon[1,i] = np.rad2deg(lat)  #Latitude [deg] 0 at Moon equator
        lon = np.arctan2(y, x)
        LL_moon[2,i] = np.rad2deg(lon)  #Longitude [deg]
        #LL_moon[1,i] =np.rad2deg(np.acos(x_moon[2,i]/LL_moon[0,i]))-90 #Latitude [deg] 0 at Moon equator
        #LL_moon[2,i] = np.sign(x_moon[1,i])*np.rad2deg(np.acos(x_moon[0,i]/np.sqrt(x_moon[0,i]**2+x_moon[1,i]**2)))+180 #Longitude [deg]

    return LL_moon