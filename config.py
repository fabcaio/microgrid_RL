import numpy as np

# #constraints
# xmax = 250; xmin = 25
# umax = [100,1000,150,150,150]
# umin = [-100,-1000,0,0,0]
# zmax = [100,1000]
# zmin = [-100,-1000]

action_dict = {'0': np.array([[0,0,0,0,0]]),
               '1': np.array([[1,0,0,0,0]]),
               '2': np.array([[0,1,0,0,0]]),
               '15': np.array([[1,1,0,0,0]]),
               '3': np.array([[0,0,1,0,0]]),
               '4': np.array([[0,0,1,1,0]]),
               '5': np.array([[0,0,1,1,1]]),
               '6': np.array([[1,0,1,0,0]]),
               '7': np.array([[1,0,1,1,0]]),
               '8': np.array([[1,0,1,1,1]]),
               '9': np.array([[0,1,1,0,0]]),
               '10': np.array([[0,1,1,1,0]]),
               '11': np.array([[0,1,1,1,1]]),
               '12': np.array([[1,1,1,0,0]]),
               '13': np.array([[1,1,1,1,0]]),
               '14': np.array([[1,1,1,1,1]])}

action_dict_SL = {'0': np.array([[0,0,0,0,0]]),
               '1': np.array([[0,0,0,0,1]]),
               '2': np.array([[0,0,0,1,0]]),
               '3': np.array([[0,0,0,1,1]]),
               '4': np.array([[0,0,1,0,0]]),
               '5': np.array([[0,0,1,0,1]]),
               '6': np.array([[0,0,1,1,0]]),
               '7': np.array([[0,0,1,1,1]]),
               '8': np.array([[0,1,0,0,0]]),
               '9': np.array([[0,1,0,0,1]]),
               '10': np.array([[0,1,0,1,0]]),
               '11': np.array([[0,1,0,1,1]]),
               '12': np.array([[0,1,1,0,0]]),
               '13': np.array([[0,1,1,0,1]]),
               '14': np.array([[0,1,1,1,0]]),
               '15': np.array([[0,1,1,1,1]]),
               '16': np.array([[1,0,0,0,0]]),
               '17': np.array([[1,0,0,0,1]]),
               '18': np.array([[1,0,0,1,0]]),
               '19': np.array([[1,0,0,1,1]]),
               '20': np.array([[1,0,1,0,0]]),
               '21': np.array([[1,0,1,0,1]]),
               '22': np.array([[1,0,1,1,0]]),
               '23': np.array([[1,0,1,1,1]]),
               '24': np.array([[1,1,0,0,0]]),
               '25': np.array([[1,1,0,0,1]]),
               '26': np.array([[1,1,0,1,0]]),
               '27': np.array([[1,1,0,1,1]]),
               '28': np.array([[1,1,1,0,0]]),
               '29': np.array([[1,1,1,0,1]]),
               '30': np.array([[1,1,1,1,0]]),
               '31': np.array([[1,1,1,1,1]])}

x_min = np.array([25])
x_max = np.array([250])
u_a_min = [-100,-1000,0,0,0,-100,-1000]
u_a_max = [100,1000,150,150,150,100,1000]
u_max = [100,1000,150,150,150]
u_min = [-100,-1000,0,0,0]
z_max = [100,1000]
z_min = [-100,-1000]

#MLD equations
Ts = 1/2 # Ts = 30m
nd = 0.99
nc = 0.9

Amld = np.array([
    [1]
])
B1 = np.array([
    [Ts/nd,0,0,0,0]
])
B2 = np.zeros((1,5))
B3 = np.array([
    [Ts*(nc-1/nd),0]
])
B5 = np.zeros((1,1))

#parameters

#ESS(battery)
Mb = 100
mb = -100

#grid
Mg = 1000
mg = -1000

#dispatchable generators
Md = 150
md = 0

eps = 1e-6

# state constraint

E2_sc = np.zeros((2,5))
E3_sc = np.zeros((2,2))
E1_sc = np.zeros((2,5))
E4_sc = np.array([
    [-1],
    [1]
])
E5_sc = np.array([
    [250],
    [-25]
])

# input constraints

E2_ic = np.zeros((10,5))
E3_ic = np.zeros((10,2))
E1_ic = np.array([
    [-1,0,0,0,0],
    [1,0,0,0,0],
    [0,-1,0,0,0],
    [0,1,0,0,0],
    [0,0,-1,0,0],
    [0,0,1,0,0],
    [0,0,0,-1,0],
    [0,0,0,1,0],
    [0,0,0,0,-1],
    [0,0,0,0,1]
])
E4_ic = np.zeros((10,1))
E5_ic = np.array([
    [100],
    [100],
    [1000],
    [1000],
    [150],
    [0],
    [150],
    [0],
    [150],
    [0],
])

# continuous auxiliary variables

#z_b

E2_zb = np.array([
    [-Mb,0,0,0,0],
    [mb,0,0,0,0],
    [-mb,0,0,0,0],
    [Mb,0,0,0,0]
])
E3_zb = np.array([
    [1, 0],
    [-1, 0],
    [1, 0],
    [-1, 0]
])
E1_zb = np.array([
    [0,0,0,0,0],
    [0,0,0,0,0],
    [1,0,0,0,0],
    [-1,0,0,0,0]
])
E4_zb = np.zeros((4,1))
E5_zb = np.array([
    [0],
    [0],
    [-mb],
    [Mb]
])

#z_grid

E2_zg = np.array([
    [0,-Mg,0,0,0],
    [0,mg,0,0,0],
    [0,-mg,0,0,0],
    [0,Mg,0,0,0]
])
E3_zg = np.array([
    [0,1],
    [0,-1],
    [0,1],
    [0,-1]
])
E1_zg = np.array([
    [0,0,0,0,0],
    [0,0,0,0,0],
    [0,1,0,0,0],
    [0,-1,0,0,0]
])
E4_zg = np.zeros((4,1))
E5_zg = np.array([
    [0],
    [0],
    [-mg],
    [Mg]
])

# discrete variales

#E2,E3,E1,E4,E5

#battery (ESS)
E2_db = np.array([
    [-mb,0,0,0,0],
    [-(Mb+eps),0,0,0,0]
])
E3_db = np.zeros((2,2))
E1_db = np.array([
    [1,0,0,0,0],
    [-1,0,0,0,0]
])
E4_db = np.zeros((2,1))
E5_db = np.array([
    [-mb],
    [-eps]
])

#grid
E2_dg = np.array([
    [0,-mg,0,0,0],
    [0,-(Mg+eps),0,0,0]
])
E3_dg = np.zeros((2,2))
E1_dg = np.array([
    [0,1,0,0,0],
    [0,-1,0,0,0]
])
E4_dg = np.zeros((2,1))
E5_dg = np.array([
    [-mg],
    [-eps]
])

#gen 1
E2_d1 = np.array([
    [0,0,-md,0,0],
    [0,0,-(Md+eps),0,0]
])
E3_d1 = np.zeros((2,2))
E1_d1 = np.array([
    [0,0,1,0,0],
    [0,0,-1,0,0]
])
E4_d1 = np.zeros((2,1))
E5_d1 = np.array([
    [-md],
    [-eps]
])

#gen 2
E2_d2 = np.array([
    [0,0,0,-md,0],
    [0,0,0,-(Md+eps),0]
])
E3_d2 = np.zeros((2,2))
E1_d2 = np.array([
    [0,0,0,1,0],
    [0,0,0,-1,0]
])
E4_d2 = np.zeros((2,1))
E5_d2 = np.array([
    [-md],
    [-eps]
])

#gen 3
E2_d3 = np.array([
    [0,0,0,0,-md],
    [0,0,0,0,-(Md+eps)]
])
E3_d3 = np.zeros((2,2))
E1_d3 = np.array([
    [0,0,0,0,1],
    [0,0,0,0,-1]
])
E4_d3 = np.zeros((2,1))
E5_d3 = np.array([
    [-md],
    [-eps]
])

# generator constraint

# E2 delta , E3 zed, E1 u, E4 x, E5

E2_gc = np.array([
    [0,0,6,0,0],
    [0,0,0,6,0],
    [0,0,0,0,6],
    [0,0,-150,0,0],
    [0,0,0,-150,0],
    [0,0,0,0,-150]
])
E3_gc = np.zeros((6,2))
E1_gc = np.array([
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1],
    [0,0,-1,0,0],
    [0,0,0,-1,0],
    [0,0,0,0,-1]
])
E4_gc = np.zeros((6,1))
E5_gc = np.zeros((6,1))

E1 = np.block([
    [E1_sc],
    [E1_ic],
    [E1_zb],
    [E1_zg],
    [E1_db],
    [E1_dg],
    [E1_d1],
    [E1_d2],
    [E1_d3],
    [E1_gc]
])

E2 = np.block([
    [E2_sc],
    [E2_ic],
    [E2_zb],
    [E2_zg],
    [E2_db],
    [E2_dg],
    [E2_d1],
    [E2_d2],
    [E2_d3],
    [E2_gc]
])

E3 = np.block([
    [E3_sc],
    [E3_ic],
    [E3_zb],
    [E3_zg],
    [E3_db],
    [E3_dg],
    [E3_d1],
    [E3_d2],
    [E3_d3],
    [E3_gc]
])

E4 = np.block([
    [E4_sc],
    [E4_ic],
    [E4_zb],
    [E4_zg],
    [E4_db],
    [E4_dg],
    [E4_d1],
    [E4_d2],
    [E4_d3],
    [E4_gc]
])

E5 = np.block([
    [E5_sc],
    [E5_ic],
    [E5_zb],
    [E5_zg],
    [E5_db],
    [E5_dg],
    [E5_d1],
    [E5_d2],
    [E5_d3],
    [E5_gc]
])

mpc_param_dict = {
    'A' : Amld,
    'B1' : B1,
    'B2' : B2,
    'B3' : B3,
    'E1' : E1,
    'E2' : E2,
    'E3' : E3,
    'E4' : E4,
    'E5' : E5,
    'x_min' : x_min,
    'x_max' : x_max,
    'u_min' : u_min,
    'u_max' : u_max,
    'z_min' : z_min,
    'z_max' : z_max,
    'u_a_min' : u_a_min,
    'u_a_max' : u_a_max
}