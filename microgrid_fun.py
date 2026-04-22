import numpy as np

#old hybrid opt problem
import gurobipy as gp
from gurobipy import GRB
from config import Amld, B1, B2, B3, B5, E1, E2, E3, E4, E5, x_min, x_max, z_min, z_max, u_min, u_max
def hybrid_fhocp(x0, N, power_res, power_load, cbuy, csell, cprod):
    mdl = gp.Model("hybridMPC")
    mdl.Params.LogToConsole = 0
    mdl.Params.MIPGap = 1e-6
    
    n=Amld.shape[1]; m=B1.shape[1]; N_bin = B2.shape[1]; N_z = B3.shape[1]
    
    x_min_tile = np.tile(x_min, (N+1,1))
    x_max_tile = np.tile(x_max, (N+1,1))
    z_min_tile = np.tile(z_min, (N,1))
    z_max_tile = np.tile(z_max, (N,1))
    u_min_tile = np.tile(u_min, (N,1))
    u_max_tile = np.tile(u_max, (N,1))

    x = mdl.addMVar(shape=(N+1, n), lb=x_min_tile, ub=x_max_tile, name='x') #1*5= 5
    z = mdl.addMVar(shape=(N, N_z), lb=z_min_tile, ub=z_max_tile, name='z') #2*4= 8
    u = mdl.addMVar(shape=(N, m), lb=u_min_tile, ub=u_max_tile, name='u') # 5*4 = 20
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
    
    n=Amld.shape[1]; m=B1.shape[1]; N_bin = B2.shape[1]; N_z = B3.shape[1]
    
    x_min_tile = np.tile(x_min, (N+1,1))
    x_max_tile = np.tile(x_max, (N+1,1))
    z_min_tile = np.tile(z_min, (N,1))
    z_max_tile = np.tile(z_max, (N,1))
    u_min_tile = np.tile(u_min, (N,1))
    u_max_tile = np.tile(u_max, (N,1))

    x = mdl.addMVar(shape=(N+1, n), lb=x_min_tile, ub=x_max_tile, name='x') #1*5= 5
    z = mdl.addMVar(shape=(N, N_z), lb=z_min_tile, ub=z_max_tile, name='z') #2*4= 8
    u = mdl.addMVar(shape=(N, m), lb=u_min_tile, ub=u_max_tile, name='u') # 5*4 = 20

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
#old hybrid opt_problem

def print_mdl_variables(mdl):
    #for a givel GUROBI model show all variables and their respective values and index
    i=0
    for v in mdl.getVars():
        print('%s %g %i' % (v.VarName, v.X, i))
        i=i+1

def build_delta_vector(list_action, N, action_dict):
    
    '''
    builds delta vector (N,5) from LSTM Network output
    '''
    
    # from list of actions builds a np.array with the stacked deltas for each time step of the prediction horizon
        
    delta = action_dict[str(list_action[0])]
    for i in range(1,N):
        delta = np.concatenate((delta, action_dict[str(list_action[i])]))
        
    return delta

def get_state_bounds(state_opt, N, cbuy, csell, cprod, power_load, power_res):
    match state_opt:
        case 1:
            state_min = np.array([25, np.min(cbuy), np.min(csell), np.min(cprod), np.min(power_load), np.min(power_res)])
            state_max = np.array([250, np.max(cbuy), np.max(csell), np.max(cprod), np.max(power_load), np.max(power_res)])
        case 2:
            state_min = np.array([25, 0, 0, 0, np.min([np.min(power_load),np.min(power_res)]), np.min([np.min(power_load),np.min(power_res)])])
            state_max = np.array([250, 1, 1, 1, np.max([np.max(power_load),np.max(power_res)]), np.max([np.max(power_load),np.max(power_res)])])
        case 3:
            state_min = np.concatenate([
                np.array([25]), np.tile(np.min(cbuy),N), np.tile(np.min(csell),N), np.tile(np.min(cprod),N),
                np.tile(np.array([np.min(power_load-power_res)]),N)
            ])
            state_max = np.concatenate([
                np.array([250]), np.tile(np.max(cbuy),N), np.tile(np.max(csell),N), np.tile(np.max(cprod),N),
                np.tile(np.array([np.max(power_load-power_res)]),N)
            ])
        case 4:
            state_min = np.concatenate([
                np.array([25]), np.tile(np.array([0]),N), np.tile(np.array([0]),N), np.tile(np.array([0]),N),
                np.tile(np.array([np.min(power_load-power_res)]),N)
            ])
            state_max = np.concatenate([
                np.array([250]), np.tile(np.array([1]),N), np.tile(np.array([1]),N), np.tile(np.array([1]),N),
                np.tile(np.array([np.max(power_load-power_res)]),N)
            ])
        case 5:
            state_min = np.concatenate([
                np.array([25]), np.tile(np.array([np.min(cbuy)]),N), np.tile(np.array([np.min(csell)]),N), np.tile(np.array([np.min(cprod)]),N),
                np.tile(np.array([np.min(power_load)]),N), np.tile(np.array([np.min(power_res)]),N)
            ])
            state_max = np.concatenate([
                np.array([250]), np.tile(np.array([np.max(cbuy)]),N), np.tile(np.array([np.max(csell)]),N), np.tile(np.array([np.max(cprod)]),N),
                np.tile(np.array([np.max(power_load)]),N), np.tile(np.array([np.max(power_res)]),N)
            ])
        case 'SL':
            state_min = np.array([25, np.min(cbuy), np.min(cprod), np.min(csell), np.min(power_load), np.min(power_res)])
            state_max = np.array([250, np.max(cbuy), np.max(cprod), np.max(csell), np.max(power_load), np.max(power_res)])
            
    return state_min, state_max

def get_input_size(state_opt, N):
    match state_opt:
        case 1|2:
            input_size=6
        case 3|4:
            input_size=4*N+1
        case 5:
            input_size=5*N+1
    return input_size

def get_linear_reward_bounds(N):
    match N:
        case 4:
            cost_lower_bound=-866
            cost_upper_bound=1838
        case 12:
            cost_lower_bound=-1832
            cost_upper_bound=5452
        case 24:
            cost_lower_bound=-1980
            cost_upper_bound=10694
        case 48:
            cost_lower_bound=-32
            cost_upper_bound=19171
    return cost_lower_bound, cost_upper_bound

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
    
def build_stacked_input(x0, N, input_size, state_min, state_max,
                        cbuy_tmp, csell_tmp, cprod_tmp, power_load_tmp, power_res_tmp):
    #build stacked vector of inputs x0, cbuy, csell, cprod, power_res, power_load
    #size of batch is (1, N, input_size)
    #note that the output is normalized
    stacked_input = np.zeros((1, N, input_size))

    x0_tmp = np.repeat(x0, N)
        
    stacked_input[0,:,:] = np.dstack((x0_tmp, cbuy_tmp, csell_tmp, cprod_tmp, power_load_tmp, power_res_tmp)).squeeze(0)

    stacked_input = state_norm(stacked_input, state_min=state_min, state_max=state_max)

    return stacked_input

def build_stacked_input_v2(x0, N, state_min2, state_max2,
                        cbuy_tmp, csell_tmp, cprod_tmp, power_load_tmp, power_res_tmp):

    stacked_input = np.zeros((1, N, 4*N+1))
    
    power_diff_tmp = power_load_tmp - power_res_tmp

    for i in range(0,N):
        stacked_input[0,i,:] = np.concatenate((x0.reshape(1,), cbuy_tmp[i:], np.repeat(cbuy_tmp[-1:],i), csell_tmp[i:], np.repeat(csell_tmp[-1:],i), cprod_tmp[i:],
                        np.repeat(cprod_tmp[-1:],i), power_diff_tmp[i:], np.repeat(power_diff_tmp[-1:],i) ))
        
    stacked_input = state_norm(stacked_input, state_min2, state_max2)
        
    return stacked_input

def build_stacked_input_zeropad(x0, N, state_min2, state_max2,
                        cbuy_tmp, csell_tmp, cprod_tmp, power_load_tmp, power_res_tmp):
    
    stacked_input = np.zeros((1, N, 5*N+1))

    stacked_input[0,0,:] = np.concatenate((x0.reshape(1,), cbuy_tmp, csell_tmp, cprod_tmp, power_load_tmp, power_res_tmp))    
    stacked_input[0,1:,:] = np.zeros((N-1, 1+5*N))

    stacked_input = state_norm(stacked_input, state_min2, state_max2)
    
    return stacked_input

def preprocess_state(state_opt, state, N, input_size, state_min, state_max, cbuy_tmp, csell_tmp, cprod_tmp, power_load_tmp, power_res_tmp):    
    
    match state_opt:
        case 1|2:
            state = build_stacked_input(state, N, input_size, state_min=state_min, state_max=state_max,
                                cbuy_tmp=cbuy_tmp, csell_tmp=csell_tmp, cprod_tmp=cprod_tmp, power_load_tmp=power_load_tmp, power_res_tmp=power_res_tmp)
        case 3|4:
            state = build_stacked_input_v2(state, N, state_min2=state_min, state_max2=state_max,
                                cbuy_tmp=cbuy_tmp, csell_tmp=csell_tmp, cprod_tmp=cprod_tmp, power_load_tmp=power_load_tmp, power_res_tmp=power_res_tmp)
        case 5:
            state = build_stacked_input_zeropad(state, N, state_min2=state_min, state_max2=state_max,
                                cbuy_tmp=cbuy_tmp, csell_tmp=csell_tmp, cprod_tmp=cprod_tmp, power_load_tmp=power_load_tmp, power_res_tmp=power_res_tmp)
    return state