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