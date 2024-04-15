import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

#MLD equations
Ts = 1/2 # Ts = 15m
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
    # [E1_ic],
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
    # [E2_ic],
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
    # [E3_ic],
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
    # [E4_ic],
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
    # [E5_ic],
    [E5_zb],
    [E5_zg],
    [E5_db],
    [E5_dg],
    [E5_d1],
    [E5_d2],
    [E5_d3],
    [E5_gc]
])

# MIQP parameters

import gurobipy as gp
from gurobipy import GRB

n=Amld.shape[1]; m=B1.shape[1]; N_bin = B2.shape[1]; N_z = B3.shape[1]

xmax = 250; xmin = 25
umax = [100,1000,150,150,150]
umin = [-100,-1000,0,0,0]
zmax = [100,1000]
zmin = [-100,-1000]

def hybrid_fhocp(x0, N, power_res, power_load, cbuy, csell, cprod):
    mdl = gp.Model("hybridMPC")
    mdl.Params.LogToConsole = 0
    
    xmin_tile = np.tile(xmin, (N+1,1))
    xmax_tile = np.tile(xmax, (N+1,1))
    zmin_tile = np.tile(zmin, (N,1))
    zmax_tile = np.tile(zmax, (N,1))
    umin_tile = np.tile(umin, (N,1))
    umax_tile = np.tile(umax, (N,1))

    x = mdl.addMVar(shape=(N+1, n), lb=xmin_tile, ub=xmax_tile, name='x') #1*5= 5
    z = mdl.addMVar(shape=(N, N_z), lb=zmin_tile, ub=zmax_tile, name='z') #2*4= 8
    u = mdl.addMVar(shape=(N, m), lb=umin_tile, ub=umax_tile, name='u') # 5*4 = 20
    delta = mdl.addMVar(shape=(N, N_bin), vtype=gp.GRB.BINARY, name='delta') # 5*4=20, total = 53

    # 1 + 1*4 + 30*4 + 1*4= 129 (number of constraints)
    mdl.addConstr(x[0, :] == x0.reshape(Amld.shape[0],))
    for k in range(N):
        mdl.addConstr(x[k+1, :] == Amld @ x[k, :] + B1 @ u[k, :] + B2 @ delta[k, :] + B3 @ z[k,:] + B5.reshape(B1.shape[0],)) # dynamics
        mdl.addConstr(E2 @ delta[k, :] + E3 @ z[k, :] <= E1 @ u[k,:] + E4 @ x[k,:] + E5.reshape(E1.shape[0],)) # mld constraints
        mdl.addConstr(u[k,0]-u[k,1]-u[k,2]-u[k,3]-u[k,4]-power_res[k]+power_load[k] == 0) # power balance

    obj1 = sum(cbuy[k]*z[k,1] - csell[k]*z[k,1] + csell[k]*u[k,1]  for k in range(N)) # cost for power exchanged with grid
    obj2 = sum(cprod[k]*u[k,2:].sum() for k in range(N)) # cost for energy production by dispatchable generators
    mdl.setObjective(obj1 + obj2, GRB.MINIMIZE)

    mdl.optimize()
    
    return mdl

def gurobi_qp(x0, N, power_res, power_load, cbuy, csell, cprod, delta):
    mdl = gp.Model("hybridMPC")
    mdl.Params.LogToConsole = 0
    
    xmin_tile = np.tile(xmin, (N+1,1))
    xmax_tile = np.tile(xmax, (N+1,1))
    zmin_tile = np.tile(zmin, (N,1))
    zmax_tile = np.tile(zmax, (N,1))
    umin_tile = np.tile(umin, (N,1))
    umax_tile = np.tile(umax, (N,1))

    x = mdl.addMVar(shape=(N+1, n), lb=xmin_tile, ub=xmax_tile, name='x') #1*5= 5
    z = mdl.addMVar(shape=(N, N_z), lb=zmin_tile, ub=zmax_tile, name='z') #2*4= 8
    u = mdl.addMVar(shape=(N, m), lb=umin_tile, ub=umax_tile, name='u') # 5*4 = 20

    # 1 + 1*4 + 30*4 + 1*4= 129 (number of constraints)
    mdl.addConstr(x[0, :] == x0.reshape(Amld.shape[0],))
    for k in range(N):
        mdl.addConstr(x[k+1, :] == Amld @ x[k, :] + B1 @ u[k, :] + B2 @ delta[k, :] + B3 @ z[k,:] + B5.reshape(B1.shape[0],)) # dynamics
        mdl.addConstr(E2 @ delta[k, :] + E3 @ z[k, :] <= E1 @ u[k,:] + E4 @ x[k,:] + E5.reshape(E1.shape[0],)) # mld constraints
        mdl.addConstr(u[k,0]-u[k,1]-u[k,2]-u[k,3]-u[k,4]-power_res[k]+power_load[k] == 0) # power balance

    obj1 = sum(cbuy[k]*z[k,1] - csell[k]*z[k,1] + csell[k]*u[k,1]  for k in range(N)) # cost for power exchanged with grid
    obj2 = sum(cprod[k]*u[k,2:].sum() for k in range(N)) # cost for energy production by dispatchable generators
    mdl.setObjective(obj1 + obj2, GRB.MINIMIZE)

    mdl.optimize()
    
    return mdl

# states: x0 (battery), [cbuy, cprod, csale, Pload, Pres] for horizon
# state_min = np.array([25, 0, 0, 0, 0, 0])
# state_max = np.array([250, 1, 1, 1, 1000, 1000])

def print_mdl_variables(mdl):
    #for a givel GUROBI model show all variables and their respective values and index
    i=0
    for v in mdl.getVars():
        print('%s %g %i' % (v.VarName, v.X, i))
        i=i+1
    return

def build_delta(mdl, N):

    '''
    builds delta vector (N,5) from GUROBI model
    '''

    delta = []
    for i in range(8*N+1, 13*N+1):
        delta.append(mdl.getVars()[i].x)
        
    delta = np.array(delta).reshape((N,5), order='C')
    
    return delta


def state_norm(state, state_min, state_max):
    # simple state normalization when the constraints are of the form l <= x <= u    
    state_span = state_max - state_min
    
    state_norm = (state-state_min)/(state_span)
    
    state_norm = (state_norm-0.5)*2
    
    return state_norm

def state_denorm(state, state_min, state_max):
    
    state_span = state_max - state_min
    
    state_denorm = state/2 + 0.5
    
    state_denorm = state_denorm*state_span+state_min
    
    return state_denorm

def qp_feasible(mdl):
    if mdl.status == 3 or mdl.status == 4 or mdl.status == 12:
        feas = False
        return feas
    else:
        feas = True
        return feas
    
def build_stacked_input(x0, i, N, input_size, state_min, state_max,
                        cbuy, csell, cprod, power_load, power_res):
#build stacked vector of inputs x0, cbuy, csell, cprod, power_res, power_load
#size of batch is (1, N, input_size)
#note that the output is normalized
    stacked_input = np.zeros((1, N, input_size))

    x0_tmp = np.repeat(x0, N)
    cbuy_tmp = cbuy[i:i+N]
    csell_tmp = csell[i:i+N]
    cprod_tmp = cprod[i:i+N]
    power_load_tmp = power_load[i:i+N]
    power_res_tmp = power_res[i:i+N]    
        
    stacked_input[0,:,:] = np.dstack((x0_tmp, cbuy_tmp, csell_tmp, cprod_tmp, power_load_tmp, power_res_tmp)).squeeze(0)

    stacked_input = state_norm(stacked_input, state_min=state_min, state_max=state_max)

    return stacked_input

def build_stacked_input_v2(x0, i, N, state_min2, state_max2,
                        cbuy, csell, cprod, power_load, power_res):

    stacked_input = np.zeros((1, N, 4*N+1))
    # stacked_input = np.zeros((1, N, 5*N+1))

    cbuy_tmp = cbuy[i:i+N]
    csell_tmp = csell[i:i+N]
    cprod_tmp = cprod[i:i+N]
    power_load_tmp = power_load[i:i+N]
    power_res_tmp = power_res[i:i+N]
    
    power_diff_tmp = power_load_tmp - power_res_tmp

    for i in range(0,N):
        stacked_input[0,i,:] = np.concatenate((x0.reshape(1,), cbuy_tmp[i:], np.repeat(cbuy_tmp[-1:],i), csell_tmp[i:], np.repeat(csell_tmp[-1:],i), cprod_tmp[i:],
                        np.repeat(cprod_tmp[-1:],i), power_diff_tmp[i:], np.repeat(power_diff_tmp[-1:],i) ))
        # stacked_input[0,i,:] = np.concatenate((x0.reshape(1,), cbuy_tmp[i:], np.repeat(cbuy_tmp[-1:],i), csell_tmp[i:], np.repeat(csell_tmp[-1:],i), cprod_tmp[i:],
        #                 np.repeat(cprod_tmp[-1:],i), power_load_tmp[i:], np.repeat(power_load_tmp[-1:],i), power_res_tmp[i:], np.repeat(power_res_tmp[-1:],i) ))
        
    stacked_input = state_norm(stacked_input, state_min2, state_max2)
        
    return stacked_input

def build_stacked_input_zeropad(x0, i, N, state_min2, state_max2,
                        cbuy, csell, cprod, power_load, power_res):
    
    stacked_input = np.zeros((1, N, 5*N+1))

    cbuy_tmp = cbuy[i:i+N]
    csell_tmp = csell[i:i+N]
    cprod_tmp = cprod[i:i+N]
    power_load_tmp = power_load[i:i+N]
    power_res_tmp = power_res[i:i+N]    

    stacked_input[0,0,:] = np.concatenate((x0.reshape(1,), cbuy_tmp, csell_tmp, cprod_tmp, power_load_tmp, power_res_tmp))    
    stacked_input[0,1:,:] = np.zeros((N-1, 1+5*N))

    stacked_input = state_norm(stacked_input, state_min2, state_max2)
    
    return stacked_input

####parameters

# input_size=1 # dimension of state
# hidden_size=128 # number of nuerons for lstm
# num_layers=1 # number of layers of lstm
# batch_first=True # (batch, seq, feature)
# batch_size=5
# seq_len=4 #prediction horizon
# n_actions=15 # number of actions per sequence step

# hidden_size=128
# lr=1e-3
# gamma=0.9
# epsilon=0.1
# mem_size=10000

# (batch_size, N, 5)

class Network(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, lr, n_actions, batch_first=True):
        super(Network, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.n_actions = n_actions
        self.batch_first = batch_first        
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.dense = nn.Linear(hidden_size, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
    def forward(self, input, h0, c0):
        
        # input = torch.randn(batch_size, seq_len, input_size)
        # h0 = torch.randn(num_layers, batch_size, hidden_size)
        # c0 = torch.randn(num_layers, batch_size, hidden_size)

        output, (hn, cn) = self.lstm(input, (h0, c0))
        
        output = self.dense(output)

        return output
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)