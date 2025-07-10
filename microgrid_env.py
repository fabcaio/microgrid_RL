import numpy as np
from microgrid_fun import qp_feasible, build_delta_vector #, state_norm, state_denorm
from config import action_dict, action_dict_SL
   
def cost_per_step(mdl, cbuy_tmp, csell_tmp, cprod_tmp):
    
    z_grid = mdl.X[6]
    P_grid = mdl.X[1]
    P1_dis = mdl.X[2]
    P2_dis = mdl.X[3]
    P3_dis = mdl.X[4]
    
    C_grid = cbuy_tmp[0]*z_grid - csell_tmp[0]*z_grid + csell_tmp[0]*P_grid
    C_prod = cprod_tmp[0]*(P1_dis + P2_dis + P3_dis)
    
    C_total = C_grid+C_prod
    
    return C_total

def cost_per_step_hybrid(mdl, N, cbuy_tmp, csell_tmp, cprod_tmp):
    
    z_grid = mdl.X[N+7]
    P_grid = mdl.X[N+2]
    P1_dis = mdl.X[N+3]
    P2_dis = mdl.X[N+4]
    P3_dis = mdl.X[N+5]
    
    C_grid = cbuy_tmp[0]*z_grid - csell_tmp[0]*z_grid + csell_tmp[0]*P_grid
    C_prod = cprod_tmp[0]*(P1_dis + P2_dis + P3_dis)
    
    C_total = C_grid+C_prod
    
    return C_total

def naive_controller(power_res_tmp, power_load_tmp):
    
    # '2' if pload>pres
    # '0' if pres>load
    
    # power_load_tmp = power_load[i:i+N]
    # power_res_tmp = power_res[i:i+N]    
    
    diff_vec = power_load_tmp - power_res_tmp
    
    actions = (diff_vec > 0)*2
    
    for i in range(len(diff_vec)):
        if diff_vec[i] > 6 and diff_vec[i] <= 150:
            actions[i] = 3
        elif diff_vec[i] > 150 and diff_vec[i] <= 156:
            actions[i] = 9
        elif diff_vec[i] > 156 and diff_vec[i] <= 300:
            actions[i] = 4
        elif diff_vec[i] > 300 and diff_vec[i] <= 306:
            actions[i] = 10
        elif diff_vec[i] > 306 and diff_vec[i] <= 450:
            actions[i] = 5
        elif diff_vec[i] > 450:
            actions[i] = 11
            
    return actions

def rew_naive(mdl_RL, mpc, x0, N, power_res_tmp, power_load_tmp, cbuy_tmp, csell_tmp, cprod_tmp, opt):
    
    actions = naive_controller(power_res_tmp, power_load_tmp)
    delta = build_delta_vector(actions,N,action_dict)
    
    mpc.build_opt_matrices(x0.reshape(-1,),delta.reshape(-1,),power_res_tmp, power_load_tmp, cbuy_tmp, csell_tmp, cprod_tmp)        
    mdl_naive = mpc.solve_gurobi_lp()
        
    if qp_feasible(mdl_naive)==True:
        if opt=='naive-step':
            cost_per_step_RL = cost_per_step(mdl_RL, cbuy_tmp, csell_tmp, cprod_tmp)
            cost_per_step_naive = cost_per_step(mdl_naive, cbuy_tmp, csell_tmp, cprod_tmp)        
            rew = (cost_per_step_naive-cost_per_step_RL)/np.maximum(np.abs(cost_per_step_RL), np.abs(cost_per_step_naive)+1e-4)
        elif opt=='naive-multistep':
            rew = (np.array(mdl_naive.ObjVal)-np.array(mdl_RL.ObjVal))/np.maximum(np.abs(mdl_naive.ObjVal), np.abs(mdl_RL.ObjVal) + 1e-4)
        
        # limits the reward (for learning stability): rewards up to 10% of improvement
        rew = min(rew, 0.1)
        
        if rew is not np.nan:
            rew = np.exp(rew*10)
        else:
            rew = 0
    else:
        # if the naive controller is not feasible, but the RL-based is, then give a reward
        rew = 1.5

    return rew

def cost2rew(cost, lower_bound=-450, alpha=0.0015):
#alpha=-np.log(1/30)/10
# e^(-alpha*a) = b when alpha=-log(b)/a
#     rew = 2*np.exp(-alpha*cost) - 1
    rew = np.exp(-alpha*(cost-lower_bound))
    return rew

def cost2rew2(cost, upper_bound=1838, lower_bound=-866):    
    
    """computes the reward between [-1, 2] by scaling the objective value (cost) from the QP
    
    (computed in microrid_misc)
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

    Returns:
        reward: float
    """
    
    if cost <= 0:
        rew = 1+1/lower_bound*cost
    else:
        rew = 1-1/(upper_bound)*cost
    
    return rew


def get_reward(opt_rew, mdl_RL, mpc, x0, N, power_res_tmp, power_load_tmp, cbuy_tmp, csell_tmp, cprod_tmp, upper_bound=0, lower_bound=0):
    
    match opt_rew:
        case 'linear':
            rew = cost2rew2(mdl_RL.ObjVal, upper_bound, lower_bound)
        case 'naive-step':
            rew = rew_naive(mdl_RL, mpc, x0, N, power_res_tmp, power_load_tmp, cbuy_tmp, csell_tmp, cprod_tmp, opt='naive-step')
        case 'naive-multistep':
            rew = rew_naive(mdl_RL, mpc, x0, N, power_res_tmp, power_load_tmp, cbuy_tmp, csell_tmp, cprod_tmp, opt='naive-multistep')
    
    return rew

class MicroGrid():
    def __init__(self, pred_horizon, mpc, state=np.array([[0]]), mode=np.array([[0,0,0,0,0]])): 

        self.cntr = 0  # termination counter
        self.state = state
        self.pred_horizon = pred_horizon
        self.mode = mode
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.idx_cntr = 0
        self.mpc = mpc
        
    def setState(self, state, idx_cntr, mode=np.array([[0,0,0,0,0]])):
          
        self.cntr = 0
        self.state = state.reshape(1,1)
        self.mode = mode        
        self.terminated = False
        self.truncated = False
        self.idx_cntr = idx_cntr
        
    def copyEnv(self, env):
        
        self.state = env.state.copy()
        self.cntr = env.cntr           
        self.terminated = env.terminated
        self.truncated = env.truncated
        self.idx_cntr = env.idx_cntr
        
        self.power_res_tmp = env.power_res_tmp.copy()
        self.power_load_tmp = env.power_load_tmp.copy()
        self.cbuy_tmp = env.cbuy_tmp.copy()
        self.csell_tmp = env.csell_tmp.copy()
        self.cprod_tmp = env.cprod_tmp.copy()
        
    def update_power_cost_data(self, power_res, power_load, cbuy, csell, cprod):
        self.power_res_tmp = power_res[self.idx_cntr:self.idx_cntr+self.pred_horizon].astype(np.double)
        self.power_load_tmp = power_load[self.idx_cntr:self.idx_cntr+self.pred_horizon].astype(np.double)
        self.cbuy_tmp = cbuy[self.idx_cntr:self.idx_cntr+self.pred_horizon].astype(np.double)
        self.csell_tmp = csell[self.idx_cntr:self.idx_cntr+self.pred_horizon].astype(np.double)
        self.cprod_tmp = cprod[self.idx_cntr:self.idx_cntr+self.pred_horizon].astype(np.double)
        
    def set_randState(self, power_res, power_load, cbuy, csell, cprod):
        
        x0 = np.random.rand(1,1)*225+25 #minimum battery level is 25
        x0 = np.clip(x0, 25+1e-4, 250)
        
        i = np.random.randint(17520-self.pred_horizon) #equivalent for 1 year of data Ts=30m
           
        self.state = x0      
        self.cntr = 0
        self.terminated = False
        self.truncated = False
        self.idx_cntr = i
        
        self.update_power_cost_data(power_res, power_load, cbuy, csell, cprod)
        
    def build_delta_vector(self, list_action: list) -> np.array:
    # from list of actions builds a np.array with the stacked deltas for each time step of the prediction horizon
        
        delta = action_dict[str(list_action[0])]
        for i in range(1,self.pred_horizon):
            delta = np.concatenate((delta, action_dict[str(list_action[i])]))
            
        return delta
        
    # def step(self, list_action, power_res, power_load, cbuy, csell, cprod):
    def step(self, list_action, cbuy, csell, cprod, power_res, power_load, cost_upper_bound, cost_lower_bound, opt_rew):
         
        delta = self.build_delta_vector(list_action) 
        self.mpc.build_opt_matrices(self.state.reshape(-1,),delta.reshape(-1,), self.power_res_tmp, self.power_load_tmp, self.cbuy_tmp, self.csell_tmp, self.cprod_tmp)
        mdl = self.mpc.solve_gurobi_lp()
        
        feas = qp_feasible(mdl)
        
        if feas==False:            
                            
            self.reward = np.array(-1)
            self.terminated = True
            
            info={'feasible': feas}
            
        else:                       
            rew = get_reward(opt_rew, mdl, self.mpc, self.state, self.pred_horizon, self.power_res_tmp, self.power_load_tmp, self.cbuy_tmp, self.csell_tmp, self.cprod_tmp, cost_upper_bound, cost_lower_bound)
            
            self.stage_cost = cost_per_step(mdl, self.cbuy_tmp, self.csell_tmp, self.cprod_tmp)            
            # self.state = np.clip(self.mpc.A@self.state + self.mpc.B_u_a@np.array(mdl.x[:7]), 25+1e-4, 250)
            self.state = self.mpc.A@self.state + self.mpc.B_u_a@np.array(mdl.x[:7])
            self.reward = rew
            self.cntr += 1
            self.idx_cntr += 1            
            self.update_power_cost_data(power_res, power_load, cbuy, csell, cprod)
            
            # to guarantee feasibility of the LP if 25 <= x < 25+1e-4 (related to the tolerance of the solver)
            if self.state >= 25 and self.state <= 25+1e-4:
                self.state = np.array([25+1e-4])
            
            # limites the duration of the episode
            if self.cntr >= 24:
                self.truncated = True
            
            # limits the simulated system to the length of the dataset
            if self.idx_cntr >= cbuy.shape[0]-self.pred_horizon:
                self.truncated = True
                
            info={'feasible': feas,
              'objval': mdl.ObjVal,
              'mdl': mdl,
              }
        
        return self.state, self.reward, self.terminated, self.truncated, info
    
    #for closed-loop testing
    def step_optimal(self, cbuy, csell, cprod, power_res, power_load):
        
        self.mpc.build_opt_matrices_hybrid(self.power_res_tmp, self.power_load_tmp, self.cbuy_tmp, self.csell_tmp, self.cprod_tmp)
        mdl= self.mpc.solve_hybrid_gurobi(self.state)
        
        feas = qp_feasible(mdl)
        
        if feas==False:            
                            
            self.reward = np.array(-1)
            self.terminated = True
            
            info={'feasible': feas}
        
        else:
            self.stage_cost = cost_per_step_hybrid(mdl, self.pred_horizon, self.cbuy_tmp, self.csell_tmp, self.cprod_tmp)
            self.state = np.clip(np.array(mdl.x[1]).reshape(-1,), 25+1e-4, 250)
            # self.state = np.clip(self.mpc.A@self.state + self.mpc.B_u_a@np.array(mdl.x[self.pred_horizon+1:self.pred_horizon+8]), 25, 250)
            self.cntr += 1
            self.idx_cntr +=1
            self.update_power_cost_data(power_res, power_load, cbuy, csell, cprod)
            
            
            # limites the duration of the episode
            if self.cntr >= 24:
                    self.truncated = True
                
            # limits the simulated system to the length of the dataset
            if self.idx_cntr >= cbuy.shape[0]-self.pred_horizon:
                    self.truncated = True
                    
            info={'feasible': feas,
                'objval': mdl.ObjVal,
                'mdl': mdl,
                }
            
            self.reward=0
        
        return self.state, self.reward, self.terminated, self.truncated, info
    
    #for closed-loop testing
    def step_heuristic(self, list_action, cbuy, csell, cprod, power_res, power_load):         
        
        delta = self.build_delta_vector(list_action) 
        self.mpc.build_opt_matrices(self.state.reshape(-1,),delta.reshape(-1,), self.power_res_tmp, self.power_load_tmp, self.cbuy_tmp, self.csell_tmp, self.cprod_tmp)
        mdl = self.mpc.solve_gurobi_lp()
        
        feas = qp_feasible(mdl)        
        
        if feas==False:
            
            list_action = naive_controller(self.power_res_tmp, self.power_load_tmp)
            delta = self.build_delta_vector(list_action)
            self.mpc.build_opt_matrices(self.state.reshape(-1,),delta.reshape(-1,), self.power_res_tmp, self.power_load_tmp, self.cbuy_tmp, self.csell_tmp, self.cprod_tmp)
            mdl = self.mpc.solve_gurobi_lp()
            
            feas1 = qp_feasible(mdl)
        
            if feas1==False:            
                                
                self.terminated = True
                
                info={'feasible': feas1}
            
        if feas==True or feas1==True:           
            
            self.stage_cost = cost_per_step(mdl, self.cbuy_tmp, self.csell_tmp, self.cprod_tmp)            
            self.state = np.clip(self.mpc.A@self.state + self.mpc.B_u_a@np.array(mdl.x[:7]), 25+1e-4, 250)
            self.cntr += 1
            self.idx_cntr += 1            
            self.update_power_cost_data(power_res, power_load, cbuy, csell, cprod)
            
            # limites the duration of the episode
            if self.cntr >= 24:
                self.truncated = True
            
            # limits the simulated system to the length of the dataset
            if self.idx_cntr >= cbuy.shape[0]-self.pred_horizon:
                self.truncated = True
                
            feas = feas or feas1   
                
            info={'feasible': feas,
            'objval': mdl.ObjVal,
            'mdl': mdl,
            }
        
            self.reward=0
            
        return self.state, self.reward, self.terminated, self.truncated, info
    
    #for closed-loop testing
    def step_heuristic_SL(self, list_action, cbuy, csell, cprod, power_res, power_load):         
        
        delta = build_delta_vector(list_action, self.pred_horizon, action_dict_SL)
        self.mpc.build_opt_matrices(self.state.reshape(-1,),delta.reshape(-1,), self.power_res_tmp, self.power_load_tmp, self.cbuy_tmp, self.csell_tmp, self.cprod_tmp)
        mdl = self.mpc.solve_gurobi_lp()
        
        feas = qp_feasible(mdl)        
        
        if feas==False:
            
            list_action = naive_controller(self.power_res_tmp, self.power_load_tmp)
            delta = self.build_delta_vector(list_action)
            self.mpc.build_opt_matrices(self.state.reshape(-1,),delta.reshape(-1,), self.power_res_tmp, self.power_load_tmp, self.cbuy_tmp, self.csell_tmp, self.cprod_tmp)
            mdl = self.mpc.solve_gurobi_lp()
            
            feas1 = qp_feasible(mdl)
        
            if feas1==False:            
                                
                self.terminated = True
                
                info={'feasible': feas1}
            
        if feas==True or feas1==True:           
            
            self.stage_cost = cost_per_step(mdl, self.cbuy_tmp, self.csell_tmp, self.cprod_tmp)            
            self.state = np.clip(self.mpc.A@self.state + self.mpc.B_u_a@np.array(mdl.x[:7]), 25+1e-4, 250)
            self.cntr += 1
            self.idx_cntr += 1            
            self.update_power_cost_data(power_res, power_load, cbuy, csell, cprod)
            
            # limites the duration of the episode
            if self.cntr >= 24:
                self.truncated = True
            
            # limits the simulated system to the length of the dataset
            if self.idx_cntr >= cbuy.shape[0]-self.pred_horizon:
                self.truncated = True
                
            feas = feas or feas1   
                
            info={'feasible': feas,
            'objval': mdl.ObjVal,
            'mdl': mdl,
            }
        
            self.reward=0
            
        return self.state, self.reward, self.terminated, self.truncated, info