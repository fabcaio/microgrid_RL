import numpy as np
from microgrid_fun import build_stacked_input, build_stacked_input_v2, build_stacked_input_zeropad
from microgrid_fun import hybrid_fhocp, qp_feasible

def get_optimality_gap(agent, Env, N_iter, cbuy, csell, cprod, power_load,
                       power_res, state_min, state_max, state_opt, cost_upper_bound, cost_lower_bound):
    """computes the open loop optimality gap (agent vs MILP) for the microgrid system
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
            
        match state_opt:
            case 1|2:
                state = build_stacked_input(Env.state, Env.idx_cntr, agent.seq_len, agent.input_size, state_min=state_min, state_max=state_max,
                                    cbuy=cbuy, csell=csell, cprod=cprod, power_load=power_load, power_res=power_res)
            case 3|4:
                state = build_stacked_input_v2(Env.state, Env.idx_cntr, agent.seq_len, state_min2=state_min, state_max2=state_max,
                                    cbuy=cbuy, csell=csell, cprod=cprod, power_load=power_load, power_res=power_res)
            case 5:
                state = build_stacked_input_zeropad(Env.state, Env.idx_cntr, agent.seq_len, state_min2=state_min, state_max2=state_max,
                                    cbuy=cbuy, csell=csell, cprod=cprod, power_load=power_load, power_res=power_res)
        
        mdl = hybrid_fhocp(Env.state, N,
                        power_res[Env.idx_cntr:Env.idx_cntr+N],
                        power_load[Env.idx_cntr:Env.idx_cntr+N],
                        cbuy[Env.idx_cntr:Env.idx_cntr+N],
                        csell[Env.idx_cntr:Env.idx_cntr+N],
                        cprod[Env.idx_cntr:Env.idx_cntr+N])
        
        action_idx = agent.choose_action(state, greedy=True)
        info = Env.step(action_idx, cbuy, csell, cprod, power_load, power_res, cost_upper_bound=cost_upper_bound, cost_lower_bound=cost_lower_bound)[4]
        
        #this exludes infeasible actions from the computation
        if qp_feasible(mdl) and (Env.terminated or Env.truncated)==False:
            
            cost_lstm = info['objval'] - np.max(csell[Env.idx_cntr:Env.idx_cntr+N])*(info['mdl'].getVars()[N].x - mdl.getVars()[N].x)
            
            cost_optimal_mem.append(mdl.ObjVal)
            cost_lstm_mem.append(cost_lstm)
            diff_soc.append(info['mdl'].getVars()[N].x - mdl.getVars()[N].x)
        else:
            cntr_infeasible += 1
            
    if len(cost_lstm_mem) == 0:
        #in case there are no feasible actions
        return np.inf, np.inf, np.inf, np.inf
    else:         
        avg_cost_lstm = sum(cost_lstm_mem)/len(cost_lstm_mem)# - sum(diff_soc)/len(diff_soc)*sell_factor
        avg_cost_optimal = sum(cost_optimal_mem)/len(cost_optimal_mem)
        optimality_gap = (avg_cost_lstm-avg_cost_optimal)/avg_cost_optimal
    
    return avg_cost_lstm, avg_cost_optimal, optimality_gap, cntr_infeasible

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
            
            match state_opt:
                case 1|2:
                    state = build_stacked_input(Env.state, Env.idx_cntr, agent.seq_len, agent.input_size, state_min=state_min, state_max=state_max,
                                        cbuy=cbuy, csell=csell, cprod=cprod, power_load=power_load, power_res=power_res)
                case 3|4:
                    state = build_stacked_input_v2(Env.state, Env.idx_cntr, agent.seq_len, state_min2=state_min, state_max2=state_max,
                                        cbuy=cbuy, csell=csell, cprod=cprod, power_load=power_load, power_res=power_res)
                case 5:
                    state = build_stacked_input_zeropad(Env.state, Env.idx_cntr, agent.seq_len, state_min2=state_min, state_max2=state_max,
                                        cbuy=cbuy, csell=csell, cprod=cprod, power_load=power_load, power_res=power_res)
            
            mdl = hybrid_fhocp(Env.state, N,
                            power_res[Env.idx_cntr:Env.idx_cntr+N],
                            power_load[Env.idx_cntr:Env.idx_cntr+N],
                            cbuy[Env.idx_cntr:Env.idx_cntr+N],
                            csell[Env.idx_cntr:Env.idx_cntr+N],
                            cprod[Env.idx_cntr:Env.idx_cntr+N])
            
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

def temp_scheduler(mem_cntr, n_samples=80000, final_temp = 80, d1=5000):
    """
    temperature scheduler (linear) for softmax exploration
    n_samples: when temperature achieves if final value
    final_temp: final temperature
    d1: when temperature is equal to 0.5 (approximately equivalent to the period of random exploration)
    """
        
    d1 = 5000

    if mem_cntr <= d1:
        temp = mem_cntr*0.5/d1
    else:
        temp = 0.5 + (mem_cntr-d1)*(final_temp-0.5)/(n_samples-d1)
        
    # return max(temp, final_temp)
    return temp