import numpy as np
from microgrid_fun import build_stacked_input, build_stacked_input_v2, build_stacked_input_zeropad #, state_norm, state_denorm

import torch
from microgrid_RL_agents import DQN_Agent
from microgrid_env import MicroGrid
from utils import get_optimality_gap, temp_scheduler
import time
import datetime

import sys
import os

data = np.load('data_costs_loads_2021_2022.npy', allow_pickle=True)
data_2021 = data[0]; data_2022 = data[1]
cbuy, csell, cprod, power_load, power_res = data_2022

N = int(sys.argv[1])
hidden_size = int(sys.argv[2])
lr = float(sys.argv[3])
state_opt = int(sys.argv[4])
final_temp = int(sys.argv[5])
seed = int(sys.argv[6])

job_idx = int(sys.argv[7])

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

gamma = 0
seq_len = N
batch_size = 32

num_layers = 1
n_actions = 16
update_target = 100
max_mem_size = 20000
DDQN = True
noisyNet = False

#for testing
# n_samples = 101
# N_iter = 101

n_samples = 300000 + job_idx*11
N_iter = 3000 + job_idx*11

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

torch.manual_seed(seed)
np.random.seed(seed)

Env = MicroGrid(N)

match state_opt:
    case 1|2:
        input_size=6
    case 3|4:
        input_size=4*N+1
    case 5:
        input_size=5*N+1
        
agent = DQN_Agent(gamma=gamma, lr=lr, input_size=input_size, seq_len=seq_len, batch_size=batch_size,
              hidden_size=hidden_size, num_layers=num_layers, n_actions=n_actions, update_target=update_target, max_mem_size=max_mem_size, DDQN=DDQN, noisyNet=noisyNet)

scores = []
rewards = []

start_time=time.time()

# epsilon_stop = n_samples*0.7 #when the decrease stops
# epsilon_final = 0.02 #final eps value

x = datetime.datetime.now()
datetime_str = '%d_%d_%d_%d_%d_%d' %(x.year % 100, x.month, x.day, x.hour, x.minute, x.second)
parameter_str = 'N %d, job_idx %d, hidden_size %d, lr %.5f, state_opt %d, final_temp %d, seed %d, datapoints %d, n_test_iter %d, gamma %.2f, batch_size %d, num_layers %d, update_target %d, max_mem_size %d, DDQN %d, noisyNet %d' %(N, job_idx, hidden_size, lr, state_opt, final_temp, seed, n_samples, N_iter, gamma, batch_size, num_layers, update_target, max_mem_size, DDQN, noisyNet)

print(parameter_str + ', date:' + datetime_str)

agent.Q_eval.train()
i = 0
while agent.mem_cntr < n_samples:
    
    # if i % 5000 == 0:
    #     cost_rl, infeasibility_cntr = evaluate_agent(Env, agent, 1000)
    #     print('avg_cost_greedy %.2f' % cost_rl, 'infeasible_cntr %.2f' % infeasibility_cntr)
    
    score = 0
    Env.set_randState()
    
#   epsilon-greedy exploration: linear decay until epsilon_stop, then constant value
    # if agent.mem_cntr <= epsilon_stop:
    #     agent.epsilon = 1 - (1-epsilon_final)/(epsilon_stop)*agent.mem_cntr
    # else:
    #     agent.epsilon = epsilon_final
    
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
        
        temp = temp_scheduler(agent.mem_cntr, n_samples, final_temp)     
        action_idx = agent.choose_action(state, softmax=True, temp=temp)
        
        state_, reward, terminated, truncated, info = Env.step(action_idx, cbuy, csell, cprod, power_load, power_res, 
                                                               cost_upper_bound=cost_upper_bound, cost_lower_bound=cost_lower_bound)
        
        match state_opt:
            case 1|2:
                state_ = build_stacked_input(state_, Env.idx_cntr, agent.seq_len, agent.input_size, state_min=state_min, state_max=state_max,
                                    cbuy=cbuy, csell=csell, cprod=cprod, power_load=power_load, power_res=power_res)
            case 3|4:
                state_ = build_stacked_input_v2(state_, Env.idx_cntr, agent.seq_len, state_min2=state_min, state_max2=state_max,
                                    cbuy=cbuy, csell=csell, cprod=cprod, power_load=power_load, power_res=power_res)
            case 5:
                state_ = build_stacked_input_zeropad(state_, Env.idx_cntr, agent.seq_len, state_min2=state_min, state_max2=state_max,
                                    cbuy=cbuy, csell=csell, cprod=cprod, power_load=power_load, power_res=power_res)
        
        score += reward
        
        rewards.append(reward)
            
        agent.store_transition(state, action_idx, reward, state_, terminated)
        
        agent.learn()

        # this loops allows for many rounds of training per step
        # for j in range(1):
            # agent.learn()        
   
    scores.append(score)
    
    avg_score = np.mean(scores[-100:])
    
    avg_reward = np.mean(rewards[-100:])
    
    tmp = time.localtime()
    current_time = time.strftime("%H:%M:%S", tmp)
    elapsed_time = time.time()-start_time
     
    if i % 100 == 1:
        print('samples %.2f,' % agent.mem_cntr, 'average reward %.2f,' % avg_reward ,'average score %.2f,' % avg_score, 'temp %.2f,' % temp,
              'elapsed time %.2f,' %(elapsed_time/3600), 'current time ', current_time)
        
    i=i+1

avg_cost_lstm, avg_cost_optimal, optimality_gap, cntr_infeasible = get_optimality_gap(agent,Env, N_iter, cbuy, csell, cprod, power_load,
                                                                                      power_res, state_min, state_max, state_opt, cost_lower_bound, cost_upper_bound)

parameter_str = 'N %.2d, job_idx %.3d, hidden_size %.3d, lr %.5f, state_opt %d, final_temp %.3d, seed %.2d, final_time %.2f, datapoints %d, n_test_iter %d, gamma %.2f, batch_size %d, num_layers %d, update_target %d, max_mem_size %d, DDQN %d, noisyNet %d' %(N, job_idx, hidden_size, lr, state_opt, final_temp, seed, elapsed_time/3600, n_samples, N_iter, gamma, batch_size, num_layers, update_target, max_mem_size, DDQN, noisyNet)
results_str = 'optimality_gap %.2f, cntr_infeasible %.2f, N_iter %d' %(optimality_gap*100, cntr_infeasible, N_iter)
with open('log_N%.2d.txt' %N, 'a') as f:
    f.write('\n' + parameter_str)
    f.write('\n' + results_str + ', date: ' + datetime_str)
    f.close()

with open('log_N%.2d_basic.txt' %N, 'a') as f:
    
    if os.stat('log_N%.2d_basic.txt' %N).st_size == 0:
        f.write('job_idx\thidden_size\tlr\tstate_opt\tfinal_temp\tseed\toptimality_gap*100\tinfeas_rate*1000\telapsed_time(h)\tdate\n')
    
    f.write('%.3d\t%d\t%.5f\t%d\t%.3d\t%.2d\t%.2f\t%.4f\t%.2f' %(job_idx, hidden_size, lr, state_opt, final_temp, seed, optimality_gap*100, cntr_infeasible/N_iter*1000, elapsed_time/3600) + '\t' + datetime_str + '\n')
    f.close()
    
print(parameter_str)
print('optimality_gap %.2f, ctr_infeasible %d, avg_cost_lstm %.2f, avg_cost_optimal %.2f' % (optimality_gap*100, cntr_infeasible, avg_cost_lstm, avg_cost_optimal) + ', date: ' + datetime_str)

torch.save(agent.Q_eval.state_dict(), 'weights//' + 'job_N%.2d_%.3d_' %(N, job_idx) + datetime_str)