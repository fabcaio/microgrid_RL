import numpy as np
from microgrid_fun import preprocess_state, qp_feasible
import time
import torch

@torch.no_grad()
def get_greedy_action(network, state, h0, c0):
    state = torch.tensor(state, dtype=torch.float32)                         
    action_idx = network(state, h0, c0)
    action_idx = torch.max(action_idx, dim=2)[1].squeeze(0)
    action_idx = action_idx.numpy()
    
    return action_idx

def get_optimality_gap(network, Env, N_iter, cbuy, csell, cprod, power_load,
                       power_res, state_min, state_max, state_opt, cost_upper_bound, cost_lower_bound, opt_rew, timeout):
    """computes the open loop optimality gap (agent vs MILP) for the microgrid system
    input: trained agent, environment, number of random iterations
    output: averaged cost for agent, average optimal cost, and optimality gap"""
    
    start_time = time.time()

    cost_lstm_mem = []
    cost_optimal_mem = []
    
    network.eval()
    
    cntr_infeasible = 0
    
    N = Env.pred_horizon

    h0 = torch.zeros(network.num_layers, 1, network.hidden_size)
    c0 = torch.zeros(network.num_layers, 1, network.hidden_size)
    
    for i in range(N_iter):
        
        if time.time() >= start_time + timeout:
            break
        
        Env.set_randState(power_res, power_load, cbuy, csell, cprod)

        Env.mpc.build_opt_matrices_hybrid(Env.power_res_tmp, Env.power_load_tmp, Env.cbuy_tmp, Env.csell_tmp, Env.cprod_tmp)
        mdl = Env.mpc.solve_hybrid_gurobi(Env.state)

        state = preprocess_state(state_opt, Env.state, N, network.input_size, state_min, state_max, Env.cbuy_tmp, Env.csell_tmp, Env.cprod_tmp, Env.power_load_tmp, Env.power_res_tmp)
        action_idx = get_greedy_action(network, state, h0=h0, c0=c0)
        info = Env.step(action_idx, cbuy, csell, cprod, power_res, power_load, cost_upper_bound=cost_upper_bound, cost_lower_bound=cost_lower_bound, opt_rew=opt_rew)[4]
        
        #this exludes infeasible actions from the computation
        if qp_feasible(mdl) and (Env.terminated or Env.truncated)==False:
            
            cost_lstm = info['objval'] # - np.max(Env.csell_tmp)*(info['mdl'].getVars()[N].x - mdl.getVars()[N].x)
            
            cost_optimal_mem.append(mdl.ObjVal)
            cost_lstm_mem.append(cost_lstm)
        else:
            cntr_infeasible += 1
            
    N_iter = len(cost_lstm_mem) + cntr_infeasible
    infeas_rate = cntr_infeasible/N_iter*1000
    
    #in case there are no feasible actions  
    if len(cost_lstm_mem) == 0:        
        return np.inf, np.inf, np.inf, infeas_rate, N_iter
    else:         
        avg_cost_lstm = sum(cost_lstm_mem)/len(cost_lstm_mem)# - sum(diff_soc)/len(diff_soc)*sell_factor
        avg_cost_optimal = sum(cost_optimal_mem)/len(cost_optimal_mem)
        optimality_gap = (avg_cost_lstm-avg_cost_optimal)/avg_cost_optimal
        
    # print('optimality_gap %.2f, ctr_infeasible %.2f, avg_cost_lstm %.2f, avg_cost_optimal %.2f' % (optimality_gap*100, cntr_infeasible, avg_cost_lstm, avg_cost_optimal))
    
    return avg_cost_lstm, avg_cost_optimal, optimality_gap, infeas_rate, N_iter

def get_optimality_gap_cl(agent, Env, N_iter, cbuy, csell, cprod, power_load,
                          power_res, state_min, state_max, state_opt, cost_lower_bound, cost_upper_bound):
    """computes the *closed-loop* optimality gap (agent vs MILP) for the microgrid system
    input: trained agent, environment, number of random iterations
    output: averaged cost for agent, average optimal cost, and optimality gap"""

    cost_lstm_mem = []
    cost_optimal_mem = []
    
    diff_soc = []

    agent.epsilon = 0
    agent.Q_eval.eval()
    
    cntr_infeasible = 0
    
    N = agent.seq_len

    for i in range(N_iter):
        
        Env.set_randState()
        
        while (Env.terminated or Env.truncated)==False:           
            
            Env.set_randState()
            Env.update_power_cost_data(power_res, power_load, cbuy, csell, cprod)
            state = preprocess_state(state_opt, Env.state, N, agent.Q_eval.input_size, state_min, state_max, Env.cbuy_tmp, Env.csell_tmp, Env.cprod_tmp, Env.power_load_tmp, Env.power_res_tmp)

            Env.mpc.build_opt_matrices_hybrid(Env.power_res_tmp, Env.power_load_tmp, Env.cbuy_tmp, Env.csell_tmp, Env.cprod_tmp)
            mdl = Env.mpc.solve_hybrid_gurobi(Env.state)
            
            action_idx = agent.choose_action(state, greedy=True)
            info = Env.step(action_idx, cbuy, csell, cprod, power_load, power_res,cost_lower_bound=cost_lower_bound, cost_upper_bound=cost_upper_bound)[4]
            
            #this exludes infeasible actions from the computation
            if qp_feasible(mdl) and (Env.terminated or Env.truncated)==False:
                
                cost_lstm = info['objval'] - np.max(csell[Env.idx_cntr:Env.idx_cntr+N])*(info['mdl'].getVars()[N].x - mdl.getVars()[N].x)
                
                cost_optimal_mem.append(mdl.ObjVal)
                cost_lstm_mem.append(cost_lstm)
                diff_soc.append(info['mdl'].getVars()[N].x - mdl.getVars()[N].x)
            else:
                cntr_infeasible += 1
          
        avg_cost_lstm = sum(cost_lstm_mem)/len(cost_lstm_mem)# - sum(diff_soc)/len(diff_soc)*sell_factor
        avg_cost_optimal = sum(cost_optimal_mem)/len(cost_optimal_mem)
        optimality_gap = (avg_cost_lstm-avg_cost_optimal)/avg_cost_optimal
        
    agent.Q_eval.train()
    
    return avg_cost_lstm, avg_cost_optimal, optimality_gap, cntr_infeasible