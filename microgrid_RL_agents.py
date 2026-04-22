import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from bisect import bisect_left

class Network(nn.Module):
    """
    defines the LSTM architecture with one output per time step
    a dense layer is used to output the correct number of actions
    """
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
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
class Network2(nn.Module):
    """
    defines the LSTM architecture with one output per time step
    a dense layer is used to output the correct number of actions
    """
    def __init__(self, input_size, hidden_size, num_layers, lr, n_actions, batch_first=True):
        super(Network2, self).__init__()
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
        
        self.h0 = torch.randn(num_layers, 1, hidden_size)
        self.c0 = torch.randn(num_layers, 1, hidden_size)
        
    def forward(self, input):
        
        # input = torch.randn(batch_size, seq_len, input_size)

        output, (hn, cn) = self.lstm(input, (self.h0, self.c0))
        
        output = self.dense(output)

        return output
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
        
class DQN_Agent():
    def __init__(self, gamma: float, lr: float, input_size: int, seq_len: int, batch_size: int, hidden_size: int,
                 num_layers: int, n_actions: int, update_target: int, max_mem_size: int=10000, DDQN: bool=True):
        
        """
        arguments:
            gamma (float): discount factor
            lr (float) : learning rate
            input_size (int) : size of the system state (with the forecasts) or the LSTM input
            seq_len (int) : prediction horizon or sequence length for the LSTM
            batch_size (int) : batch size
            hidden_size (int) : number of neurons in the hidden layers
            num_layers (int) : number of layers for the LSTM (default is 1)
            n_actions (int) : number of actions per time step or size of the output of the LSTM
            update_target (int) : (only relevant if DDQN==True)
            max_mem_size (int) : maximum capacity of the replay buffer
            DDQN (bool): True if DDQN is used
        """
        
        self.gamma = gamma
        self.epsilon = 1 # starts out fully random
        self.lr = lr
        self.input_size = input_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_actions = n_actions
        self.mem_size = max_mem_size  
        self.DDQN = DDQN
                           
        self.mem_cntr = 0
        
        self.Q_eval = Network(input_size, hidden_size, num_layers, lr, n_actions)
        
        self.state_memory = np.zeros((self.mem_size, seq_len, input_size), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, seq_len, input_size), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, self.seq_len), dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        
        if self.DDQN == True:
            self.Q_target = Network(input_size, hidden_size, num_layers, lr, n_actions)       
            
            self.target_cntr = 0
            self.update_target = update_target
        
        
    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        #ACTION HAS TO COME IN A NP.ARRAY([x,y,z,w])
        
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        
        self.mem_cntr += 1
            
    @torch.no_grad()
    def choose_action(self, state, eps_greedy=False, softmax=False, greedy=False, temp = 0):
        
        #if no option is chosen, then the method defaults to the epsilon greedy policy
        if eps_greedy==False and softmax==False : greedy = True
                     
        #epsilon-greedy policy   
        if eps_greedy == True:
            if np.random.random() > self.epsilon:                
                state = torch.tensor(state, dtype=torch.float32)
                
                h0 = torch.zeros(self.num_layers, 1, self.hidden_size)
                c0 = torch.zeros(self.num_layers, 1, self.hidden_size)
                            
                actions = self.Q_eval(state, h0, c0)
                actions = torch.max(actions, dim=2)[1].squeeze(0)
                actions = actions.numpy()
                                
            else:                
                actions = np.random.randint(low=0,high=self.n_actions,size=(self.seq_len,))
                          
            return actions
        
        if softmax==True:
            actions, _ = self.sample_action_softmax(state, temp)
            
            return actions
        
        if greedy==True:
            state = torch.tensor(state, dtype=torch.float32)
                
            h0 = torch.zeros(self.num_layers, 1, self.hidden_size)
            c0 = torch.zeros(self.num_layers, 1, self.hidden_size)
                        
            actions = self.Q_eval(state, h0, c0)
            actions = torch.max(actions, dim=2)[1].squeeze(0)
            actions = actions.numpy()
            
            return actions
        
    @torch.no_grad()
    def sample_action_softmax(self, state, temp):
        state = torch.tensor(state, dtype=torch.float32)
        h0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        c0 = torch.zeros(self.num_layers, 1, self.hidden_size)

        Q_vals = self.Q_eval(state, h0, c0)

        Q_max = torch.max(Q_vals, dim=2)[0]
        Q_max_vec = torch.tile(Q_max, (self.n_actions,1)).unsqueeze(0)
        Q_max_vec = Q_max_vec.permute(0,2,1)

        Q_vals_norm = Q_vals - Q_max_vec
        Q_softmax = torch.exp(temp*Q_vals_norm)
        Q_softmax_sum = torch.tile(torch.sum(Q_softmax, dim=2), (self.n_actions,1)).unsqueeze(0)
        Q_softmax_sum = Q_softmax_sum.permute(0,2,1)
        Q_softmax_norm = Q_softmax/Q_softmax_sum

        Q_softmax_cumsum = torch.cumsum(Q_softmax_norm, dim=2)
        random_sample_vec = np.random.rand(self.seq_len,)

        sampled_actions = np.zeros(self.seq_len,)

        for i in range(self.seq_len):
            sampled_actions[i] = bisect_left(Q_softmax_cumsum[:,i,:].detach().numpy().reshape(self.n_actions,), random_sample_vec[i], lo=0, hi=self.n_actions-1)
            
        info = {
            'Q_vals': Q_vals,
            'Q_vals_norm': Q_vals_norm,
            'Q_softmax_norm': Q_softmax_norm,
            'Q_softmax_cumsum': Q_softmax_cumsum,
        }
            
        return sampled_actions.astype(int), info    
    
    def learn(self):
        if self.mem_cntr < self.batch_size:
            return
             
        max_mem = min(self.mem_cntr, self.mem_size)        
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        state_batch = torch.tensor(self.state_memory[batch], dtype=torch.float32)
        new_state_batch = torch.tensor(self.new_state_memory[batch], dtype=torch.float32)
        reward_batch = torch.tensor(self.reward_memory[batch], dtype=torch.float32)
        terminal_batch = torch.tensor(self.terminal_memory[batch], dtype=torch.bool)
        action_batch = self.action_memory[batch]

        h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
        c0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

        q_eval = self.Q_eval(state_batch, h0, c0)

        q_eval = q_eval[torch.arange(self.batch_size).unsqueeze(1), torch.arange(self.seq_len), action_batch] #.unsqueeze(-1)

        with torch.no_grad():
            
            if self.DDQN==True:
                q_next = self.Q_target(new_state_batch, h0, c0)
            else:
                q_next = self.Q_eval(new_state_batch, h0, c0)            
            
            q_next = torch.max(q_next, dim=2)[0]
            q_next[terminal_batch] = 0.0
            
            reward_batch = reward_batch.unsqueeze(-1).repeat(1,self.seq_len) #repeats the reward for each LSTM head            
            q_target = reward_batch + self.gamma * q_next # does not take the average of the LSTM heads
            
            # q_next_max_mean = torch.mean(q_next,dim=1).unsqueeze(-1).repeat(1,self.seq_len) # average Q value for all LSTM heads
            # q_target = reward_batch + self.gamma * q_next_max_mean # target with the average of the LSTM heads

        loss = self.Q_eval.loss(q_eval, q_target)

        self.Q_eval.zero_grad()        
        loss.backward()
        nn.utils.clip_grad_value_(self.Q_eval.parameters(), clip_value=1.0) # gradient clipping for learning stability
        self.Q_eval.optimizer.step()
        
        if self.DDQN == True:
            self.target_cntr += 1
            if self.target_cntr % self.update_target == 0:
                self.Q_target.load_state_dict(self.Q_eval.state_dict())
                self.target_cntr = 0