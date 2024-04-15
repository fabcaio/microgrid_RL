import numpy as np
from microgrid_fun import gurobi_qp, qp_feasible #, state_norm, state_denorm

# data = np.load('data_costs_loads_2021_2022.npy', allow_pickle=True)
# data_2021 = data[0]; data_2022 = data[1]
# cbuy, csell, cpro, power_load, power_res = data_2022

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

# def cost2rew(cost, lower_bound=-450, alpha=0.0015):
# #alpha=-np.log(1/30)/10
# # e^(-alpha*a) = b when alpha=-log(b)/a
# #     rew = 2*np.exp(-alpha*cost) - 1
#     rew = np.exp(-alpha*(cost-lower_bound))
#     return rew

def cost2rew2(cost, upper_bound=1838, lower_bound=-866):    
    
    """computes the reward between [-1, 2] by scaling the objective value (cost) from the QP
    
    for N=4 (computed in microrid_misc)
    lower_bound = -866, upper_bound = 1838

    Returns:
        reward: float
    """
    
    if cost <= 0:
        rew = 1+1/lower_bound*cost
    else:
        rew = 1-1/(upper_bound)*cost
    
    return rew

class MicroGrid():
    def __init__(self, pred_horizon, state=np.array([[0]]), mode=np.array([[0,0,0,0,0]])): 

        self.cntr = 0  # termination counter
        self.state = state
        self.pred_horizon = pred_horizon
        self.mode = mode
        self.reward = 0
        self.terminated = False
        self.truncated = False
        self.idx_cntr = 0        
        
    def setState(self, state, idx_cntr, mode=np.array([[0,0,0,0,0]])):
          
        self.cntr = 0
        self.state = state.reshape(1,1)
        self.mode = mode        
        self.terminated = False
        self.truncated = False
        self.idx_cntr = idx_cntr
        
    def copyEnv(self, env):
        
        self.cntr = env.cntr
        self.state = env.state
        self.mode = env.mode        
        self.reward = env.reward
        self.terminated = env.terminated
        self.truncated = env.truncated
        self.idx_cntr = env.idx_cntr
        
    def set_randState(self, mode=np.array([[0,0,0,0,0]])):
        
        x0 = np.random.rand(1,1)*225+25 #minimum battery level is 25
        
        i = np.random.randint(17520-self.pred_horizon) #equivalent for 1 year of data Ts=30m
           
        self.state = x0
        self.mode = mode        
        self.cntr = 0
        self.terminated = False
        self.truncated = False
        self.idx_cntr = i
        
    def build_delta_vector(self, list_action: list) -> np.array:
    # from list of actions builds a np.array with the stacked deltas for each time step of the prediction horizon
        
        delta = action_dict[str(list_action[0])]
        for i in range(1,self.pred_horizon):
            delta = np.concatenate((delta, action_dict[str(list_action[i])]))
            
        return delta
        
    # def step(self, list_action, power_res, power_load, cbuy, csell, cprod):
    def step(self, list_action, cbuy, csell, cprod, power_load, power_res, cost_upper_bound, cost_lower_bound):
         
        delta = self.build_delta_vector(list_action)

        mdl = gurobi_qp(self.state, self.pred_horizon,
                            power_res[self.idx_cntr:self.idx_cntr+self.pred_horizon],
                            power_load[self.idx_cntr:self.idx_cntr+self.pred_horizon],
                            cbuy[self.idx_cntr:self.idx_cntr+self.pred_horizon],
                            csell[self.idx_cntr:self.idx_cntr+self.pred_horizon],
                            cprod[self.idx_cntr:self.idx_cntr+self.pred_horizon],
                            delta)
        
        feas = qp_feasible(mdl)
        
        if feas==False:            
                            
            self.reward = np.array(-1)
            self.terminated = True
            
            info={'feasible': feas}
            
        else:           
            # rew = cost2rew(mdl.ObjVal, lower_bound=-450, alpha=0.0015)
            
            rew = cost2rew2(mdl.ObjVal, cost_upper_bound, cost_lower_bound)
            
            self.state = np.array([[mdl.getVars()[1].x]])
            self.mode = action_dict[str(list_action[0])]           
            self.reward = rew
            self.cntr += 1
            self.idx_cntr += 1
            
            if self.cntr >= 24:
                self.truncated = True
                
            if self.idx_cntr >= cbuy.shape[0]-self.pred_horizon:
                self.truncated = True
                
            info={'feasible': feas,
              'objval': mdl.ObjVal,
              'mdl': mdl,
              }
        
        return self.state, self.reward, self.terminated, self.truncated, info