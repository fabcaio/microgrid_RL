import numpy as np
import torch
from microgrid_RL_agents import DQN_Agent
from microgrid_env import MicroGrid
from microgrid_mpc import MicrogridMPC
from microgrid_fun import preprocess_state
from config import mpc_param_dict
from utils import get_optimality_gap
import datetime
import time
import copy
import random

import sys
import os

start_time=time.time()

data = np.load('data_entso/data_costs_loads_2021_2022.npy', allow_pickle=True)
data_2021 = data[0]; data_2022 = data[1]
cbuy, csell, cprod, power_load, power_res = data_2022
cbuy_2021, csell_2021, cprod_2021, power_load_2021, power_res_2021 = data_2021

testing=False

if testing==False:
    N = int(sys.argv[1])
    hidden_size = int(sys.argv[2])
    lr = float(sys.argv[3])
    state_opt = int(sys.argv[4])
    final_temp = int(sys.argv[5])
    seed = int(sys.argv[6])
    gamma = float(sys.argv[7])
    exploration = sys.argv[8]
    opt_rew = sys.argv[9]
    job_idx = int(sys.argv[10])
    n_experiment = int(sys.argv[11])
    # timeout = (2*60+50)*60
    # timeout_gap = 5*60
    timeout = float(sys.argv[12])
    timeout_gap = float(sys.argv[13])
    n_threads=1
elif testing==True:
    N = 4
    hidden_size=128
    lr=1e-3
    state_opt=3
    final_temp=250
    seed=7
    gamma = 0.9
    exploration = 'softmax'
    opt_rew = 'linear'
    job_idx=999
    n_experiment=1
    timeout = 1*60
    timeout_gap = 0.2*60    
    n_threads=16
    
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

torch.set_num_threads(n_threads)

"""selects the proper bounds to scale the objective value of the LP to the [-1,2] range"""
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

seq_len = N
batch_size = 32

num_layers = 1
n_actions = 16
update_target = 500
max_mem_size = 20000
DDQN = True

N_iter = 3000

"""
defines the state normalization. it depends on how the input is given to the LSTM

1: size: (1, N, 6), build_stacked_input
2: size: (1, N, 6), build_stacked_input, 0-1 for costs, min/max(power_load, power_res)
3: size: (1, N, 4*N+1), build_stacked_input_v2, Pload-Pres
4: size: (1, N, 4*N+1), build_stacked_input_v2, 0-1 for costs, Pload-Pres
5: size: (1, N, 5*N+1), build_stacked_input_zeropad
"""   
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

## training loop

mpc = MicrogridMPC(N, mpc_param_dict, n_threads)
Env = MicroGrid(N, mpc)

match state_opt:
    case 1|2:
        input_size=6
    case 3|4:
        input_size=4*N+1
    case 5:
        input_size=5*N+1
        
agent = DQN_Agent(gamma=gamma, lr=lr, input_size=input_size, seq_len=seq_len, batch_size=batch_size,
              hidden_size=hidden_size, num_layers=num_layers, n_actions=n_actions, update_target=update_target, max_mem_size=max_mem_size, DDQN=DDQN)

scores = [0.]
rewards = [0.]

x = datetime.datetime.now()
datetime_str = '%.2d_%.2d_%.2d_%.2d_%.2d_%.2d' %(x.year % 100, x.month, x.day, x.hour, x.minute, x.second)
parameter_str1 = 'N %d, hidden_size %d, lr %.5f, state_opt %d, final_temp %d, seed %d, gamma %.2f, exploration = %s, opt_rew = %s, job_idx %d' % (N, hidden_size, lr, state_opt, final_temp, seed, gamma, exploration, opt_rew, job_idx)
parameter_str2 = 'n_test_iter %d, batch_size %d, num_layers %d, update_target %d, max_mem_size %d, DDQN %d, timeout %.2f' %(N_iter, batch_size, num_layers, update_target, max_mem_size, DDQN, timeout/3600)

print(parameter_str1 + ', date:' + datetime_str)
print(parameter_str2)

agent.Q_eval.train()
i = 0
epsilon_final = 0.001 #final eps value

network_best_reward = copy.deepcopy(agent.Q_eval)
network_best_feasibility = copy.deepcopy(agent.Q_eval)
best_avg_rewards = -np.inf
best_infeasibility = 1000

# while agent.mem_cntr < n_samples:
while time.time() < start_time + timeout:
        
    score = 0
    Env.set_randState(power_res, power_load, cbuy, csell, cprod)
    state_ = preprocess_state(state_opt, Env.state, N, input_size, state_min, state_max, Env.cbuy_tmp, Env.csell_tmp, Env.cprod_tmp, Env.power_load_tmp, Env.power_res_tmp)
    
    while (Env.terminated or Env.truncated)==False:       
        
        state = state_
        
        if exploration=='eps_greedy':
            agent.epsilon = max(epsilon_final, 1 - (1-epsilon_final)/(0.7*timeout)*(time.time()-start_time))
            action_idx = agent.choose_action(state, eps_greedy=True)
        elif exploration=='softmax':    
            temp = min(final_temp*(time.time()-start_time)/(0.7*timeout), final_temp)
            action_idx = agent.choose_action(state, softmax=True, temp=temp)        
        
        _, reward, terminated, truncated, info = Env.step(action_idx, cbuy, csell, cprod, power_res, power_load, 
                                                               cost_upper_bound=cost_upper_bound, cost_lower_bound=cost_lower_bound, opt_rew=opt_rew)
        
        state_ = preprocess_state(state_opt, Env.state, N, input_size, state_min, state_max, Env.cbuy_tmp, Env.csell_tmp, Env.cprod_tmp, Env.power_load_tmp, Env.power_res_tmp)
        
        score += reward
        
        rewards.append(reward)
            
        agent.store_transition(state, action_idx, reward, state_, terminated)
        
        # this loops allows for many rounds of training per step
        # for j in range(1):
        agent.learn()
        
        if agent.mem_cntr % 500 == 1:
                
            avg_score = np.mean(scores[-100:])            
            avg_reward = np.mean(rewards[-100:])
            
            tmp = time.localtime()
            current_time = time.strftime("%H:%M:%S", tmp)
            elapsed_time = time.time()-start_time
            
            if exploration=='eps_greedy':
                print('samples %.2f,' % agent.mem_cntr, 'average reward %.2f,' % avg_reward ,'average score %.2f,' % avg_score, 'epsilon %.3f,' % agent.epsilon,
                    'elapsed time %.2f,' %(elapsed_time/3600), 'current time ', current_time)
            elif exploration=='softmax':
                print('samples %.2f,' % agent.mem_cntr, 'average reward %.2f,' % avg_reward ,'average score %.2f,' % avg_score, 'temp %.2f,' % temp,
                    'elapsed time %.2f,' %(elapsed_time/3600), 'current time ', current_time)
                
        # stores the agent with best reward and the agent with best feasibility
        if agent.mem_cntr % 1000 == 1:
            if np.mean(rewards[-1000:]) > best_avg_rewards:
                network_best_reward = copy.deepcopy(agent.Q_eval)
            if np.sum(np.array(rewards[-1000:])==-1) < best_infeasibility:
                network_best_feasibility = copy.deepcopy(agent.Q_eval)

    scores.append(score)
    
#### Testing

# on training set

avg_cost_lstm, avg_cost_optimal, optimality_gap, infeas_rate, N_iter = get_optimality_gap(agent.Q_eval,Env, N_iter, cbuy, csell, cprod, power_load, power_res, state_min, state_max, state_opt, cost_lower_bound, cost_upper_bound, opt_rew, timeout_gap)

_, _, optimality_gap_best_reward, infeas_rate_best_reward, _ = get_optimality_gap(network_best_reward,Env, N_iter, cbuy, csell, cprod, power_load, power_res, state_min, state_max, state_opt, cost_lower_bound, cost_upper_bound, opt_rew, timeout_gap)

_, _, optimality_gap_best_feas, infeas_rate_best_feas, _ = get_optimality_gap(network_best_feasibility,Env, N_iter, cbuy, csell, cprod, power_load, power_res, state_min, state_max, state_opt, cost_lower_bound, cost_upper_bound, opt_rew, timeout_gap)

# on test_set
_, _, optimality_gap_2021, infeas_rate_2021, _ = get_optimality_gap(agent.Q_eval,Env, N_iter, cbuy_2021, csell_2021, cprod_2021, power_load_2021, power_res_2021, state_min, state_max, state_opt, cost_lower_bound, cost_upper_bound, opt_rew, timeout_gap)

_, _, optimality_gap_best_reward_2021, infeas_rate_best_reward_2021, _ = get_optimality_gap(network_best_reward,Env, N_iter, cbuy_2021, csell_2021, cprod_2021, power_load_2021, power_res_2021, state_min, state_max, state_opt, cost_lower_bound, cost_upper_bound, opt_rew, timeout_gap)

_, _, optimality_gap_best_feas_2021, infeas_rate_best_feas_2021, _ = get_optimality_gap(network_best_feasibility,Env, N_iter, cbuy_2021, csell_2021, cprod_2021, power_load_2021, power_res_2021, state_min, state_max, state_opt, cost_lower_bound, cost_upper_bound, opt_rew, timeout_gap)

#### Logging

os.makedirs('logs_results//N_%.2d//experiment_%.2d' %(N, n_experiment), exist_ok=True)

results_str = 'optimality_gap %.2f, infeas_rate %.2f, N_iter %d' %(optimality_gap*100, infeas_rate, N_iter)
with open('logs_results//N_%.2d//experiment_%.2d//log_N%.2d_exp%.2d.txt' %(N, n_experiment, N, n_experiment), 'a') as f:
    f.write('\n' + parameter_str1)
    f.write('\n' + results_str + ', date: ' + datetime_str)
    f.close()

with open('logs_results//N_%.2d//experiment_%.2d//log_N%.2d_exp%.2d_basic.txt' %(N, n_experiment, N, n_experiment), 'a') as f:
    
    if os.stat('logs_results//N_%.2d//experiment_%.2d//log_N%.2d_exp%.2d_basic.txt' %(N, n_experiment, N, n_experiment)).st_size == 0:
        f.write('job_idx\topt_gap\tinfeas_rate\thidden_size\tlr\tstate_opt\tfinal_temp\tseed\tgamma\texploration\topt_rew\tdate\n')
    
    f.write('%.3d\t%.2f\t%.2f\t%d\t%.5f\t%d\t%d\t%d\t%.2f\t%s\t%s' %(job_idx, optimality_gap*100, infeas_rate, hidden_size, lr, state_opt, final_temp, seed, gamma, exploration, opt_rew) + '\t' + datetime_str + '\n')
    f.close()
    
with open('logs_results//N_%.2d//experiment_%.2d//log_N%.2d_exp%.2d_basic_best_reward.txt' %(N, n_experiment, N, n_experiment), 'a') as f:
    
    if os.stat('logs_results//N_%.2d//experiment_%.2d//log_N%.2d_exp%.2d_basic_best_reward.txt' %(N, n_experiment, N, n_experiment)).st_size == 0:
        f.write('job_idx\topt_gap\tinfeas_rate\thidden_size\tlr\tstate_opt\tfinal_temp\tseed\tgamma\texploration\topt_rew\tdate\n')
    
    f.write('%.3d\t%.2f\t%.2f\t%d\t%.5f\t%d\t%d\t%d\t%.2f\t%s\t%s' %(job_idx, optimality_gap_best_reward*100, infeas_rate_best_reward, hidden_size, lr, state_opt, final_temp, seed, gamma, exploration, opt_rew) + '\t' + datetime_str + '\n')
    f.close()
    
with open('logs_results//N_%.2d//experiment_%.2d//log_N%.2d_exp%.2d_basic_best_feas.txt' %(N, n_experiment, N, n_experiment), 'a') as f:
    
    if os.stat('logs_results//N_%.2d//experiment_%.2d//log_N%.2d_exp%.2d_basic_best_feas.txt' %(N, n_experiment, N, n_experiment)).st_size == 0:
        f.write('job_idx\topt_gap\tinfeas_rate\thidden_size\tlr\tstate_opt\tfinal_temp\tseed\tgamma\texploration\topt_rew\tdate\n')
    
    f.write('%.3d\t%.2f\t%.2f\t%d\t%.5f\t%d\t%d\t%d\t%.2f\t%s\t%s' %(job_idx, optimality_gap_best_feas*100, infeas_rate_best_feas, hidden_size, lr, state_opt, final_temp, seed, gamma, exploration, opt_rew) + '\t' + datetime_str + '\n')
    f.close()
    
# on test set   

with open('logs_results//N_%.2d//experiment_%.2d//log_N%.2d_exp%.2d_basic_2021.txt' %(N, n_experiment, N, n_experiment), 'a') as f:
    
    if os.stat('logs_results//N_%.2d//experiment_%.2d//log_N%.2d_exp%.2d_basic_2021.txt' %(N, n_experiment, N, n_experiment)).st_size == 0:
        f.write('job_idx\topt_gap\tinfeas_rate\thidden_size\tlr\tstate_opt\tfinal_temp\tseed\tgamma\texploration\topt_rew\tdate\n')
    
    f.write('%.3d\t%.2f\t%.2f\t%d\t%.5f\t%d\t%d\t%d\t%.2f\t%s\t%s' %(job_idx, optimality_gap_2021*100, infeas_rate_2021, hidden_size, lr, state_opt, final_temp, seed, gamma, exploration, opt_rew) + '\t' + datetime_str + '\n')
    f.close()
    
with open('logs_results//N_%.2d//experiment_%.2d//log_N%.2d_exp%.2d_basic_best_reward_2021.txt' %(N, n_experiment, N, n_experiment), 'a') as f:
    
    if os.stat('logs_results//N_%.2d//experiment_%.2d//log_N%.2d_exp%.2d_basic_best_reward_2021.txt' %(N, n_experiment, N, n_experiment)).st_size == 0:
        f.write('job_idx\topt_gap\tinfeas_rate\thidden_size\tlr\tstate_opt\tfinal_temp\tseed\tgamma\texploration\topt_rew\tdate\n')
    
    f.write('%.3d\t%.2f\t%.2f\t%d\t%.5f\t%d\t%d\t%d\t%.2f\t%s\t%s' %(job_idx, optimality_gap_best_reward_2021*100, infeas_rate_best_reward_2021, hidden_size, lr, state_opt, final_temp, seed, gamma, exploration, opt_rew) + '\t' + datetime_str + '\n')
    f.close()
    
with open('logs_results//N_%.2d//experiment_%.2d//log_N%.2d_exp%.2d_basic_best_feas_2021.txt' %(N, n_experiment, N, n_experiment), 'a') as f:
    
    if os.stat('logs_results//N_%.2d//experiment_%.2d//log_N%.2d_exp%.2d_basic_best_feas_2021.txt' %(N, n_experiment, N, n_experiment)).st_size == 0:
        f.write('job_idx\topt_gap\tinfeas_rate\thidden_size\tlr\tstate_opt\tfinal_temp\tseed\tgamma\texploration\topt_rew\tdate\n')
    
    f.write('%.3d\t%.2f\t%.2f\t%d\t%.5f\t%d\t%d\t%d\t%.2f\t%s\t%s' %(job_idx, optimality_gap_best_feas_2021*100, infeas_rate_best_feas_2021, hidden_size, lr, state_opt, final_temp, seed, gamma, exploration, opt_rew) + '\t' + datetime_str + '\n')
    f.close()

#####

parameter_str2 = 'n_samples %d, n_test_iter %d, batch_size %d, num_layers %d, update_target %d, max_mem_size %d, DDQN %d, timeout %f' %(agent.mem_cntr, N_iter, batch_size, num_layers, update_target, max_mem_size, DDQN, timeout/3600)
print(parameter_str2)    
print(parameter_str1)
print(results_str + ', avg_cost_lstm %.2f, avg_cost_optimal %.2f' % (avg_cost_lstm, avg_cost_optimal) + ', date: ' + datetime_str)

dict_hyperpar = {
    'N' : N,
    'hidden_size' : hidden_size,
    'lr' : lr,
    'state_opt' : state_opt,
    'final_temp' : final_temp,
    'seed' : seed,
    'gamma' : gamma,
    'exploration' : exploration,
    'opt_rew' : opt_rew,
    'job_idx' : job_idx,
    'timeout' : timeout,
    'timeout_gap' : timeout_gap,
    'input_size': input_size,
    'state_min': state_min,
    'state_max': state_max,
    'num_layers': num_layers,
    'n_actions': n_actions,
    'update_target': update_target,
    'max_mem_size': max_mem_size,
    'DDQN': DDQN,
    'N_iter': N_iter
}

os.makedirs('weights//N_%.2d//experiment_%.2d' %(N, n_experiment), exist_ok=True)

np.save('weights//N_%.2d//experiment_%.2d//info_exp%.2d_%.3d.npy' %(N, n_experiment, n_experiment, job_idx),dict_hyperpar)
torch.save(agent.Q_eval.state_dict(), 'weights//N_%.2d//experiment_%.2d//weight_exp%.2d_%.3d_last.npy' %(N, n_experiment, n_experiment, job_idx))
torch.save(network_best_reward.state_dict(), 'weights//N_%.2d//experiment_%.2d//weight_exp%.2d_%.3d_best_reward.npy' %(N, n_experiment, n_experiment, job_idx))
torch.save(network_best_feasibility.state_dict(), 'weights//N_%.2d//experiment_%.2d//weight_exp%.2d_%.3d_best_feas.npy' %(N, n_experiment, n_experiment, job_idx))