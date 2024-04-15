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
    
#### Noisy Net

#copied from the torchrl library
from typing import Optional, Union, Sequence
DEVICE_TYPING = Union[torch.device, str, int]
import math
class NoisyLinear(nn.Linear):
    """Noisy Linear Layer.

    Presented in "Noisy Networks for Exploration", https://arxiv.org/abs/1706.10295v3

    A Noisy Linear Layer is a linear layer with parametric noise added to the weights. This induced stochasticity can
    be used in RL networks for the agent's policy to aid efficient exploration. The parameters of the noise are learned
    with gradient descent along with any other remaining network weights. Factorized Gaussian
    noise is the type of noise usually employed.


    Args:
        in_features (int): input features dimension
        out_features (int): out features dimension
        bias (bool, optional): if ``True``, a bias term will be added to the matrix multiplication: Ax + b.
            Defaults to ``True``
        device (DEVICE_TYPING, optional): device of the layer.
            Defaults to ``"cpu"``
        dtype (torch.dtype, optional): dtype of the parameters.
            Defaults to ``None`` (default pytorch dtype)
        std_init (scalar, optional): initial value of the Gaussian standard deviation before optimization.
            Defaults to ``0.1``

    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[torch.dtype] = None,
        std_init: float = 0.1,
    ):
        nn.Module.__init__(self)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = std_init

        self.weight_mu = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.weight_sigma = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.register_buffer(
            "weight_epsilon",
            torch.empty(out_features, in_features, device=device, dtype=dtype),
        )
        if bias:
            self.bias_mu = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.bias_sigma = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.register_buffer(
                "bias_epsilon",
                torch.empty(out_features, device=device, dtype=dtype),
            )
        else:
            self.bias_mu = None
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        if self.bias_mu is not None:
            self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: Union[int, torch.Size, Sequence]) -> torch.Tensor:
        if isinstance(size, int):
            size = (size,)
        x = torch.randn(*size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    @property
    def weight(self) -> torch.Tensor:
        if self.training:
            return self.weight_mu + self.weight_sigma * self.weight_epsilon
        else:
            return self.weight_mu

    @property
    def bias(self) -> Optional[torch.Tensor]:
        if self.bias_mu is not None:
            if self.training:
                return self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                return self.bias_mu
        else:
            return None
        
class NoisyNetwork(nn.Module):
    """the same as Network class, but utilizes noisy linear layers"""
    
    def __init__(self, input_size, hidden_size, num_layers, lr, n_actions, batch_first=True):
        super(NoisyNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.n_actions = n_actions
        self.batch_first = batch_first        
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        self.noisydense = NoisyLinear(hidden_size, n_actions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
    def forward(self, input, h0, c0):
        
        # input = torch.randn(batch_size, seq_len, input_size)
        # h0 = torch.randn(num_layers, batch_size, hidden_size)
        # c0 = torch.randn(num_layers, batch_size, hidden_size)

        output, (hn, cn) = self.lstm(input, (h0, c0))
        
        output = self.noisydense(output)

        return output
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.noisydense.reset_noise()
        
class DQN_Agent():
    def __init__(self, gamma: float, lr: float, input_size: int, seq_len: int, batch_size: int, hidden_size: int,
                 num_layers: int, n_actions: int, update_target: int, max_mem_size: int=10000, DDQN: bool=True, noisyNet: bool = False):
        
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
            noisyNet: True if noisyNet is used
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
        self.noisyNet = noisyNet
                   
        self.mem_cntr = 0
        
        if self.noisyNet == True:
            self.Q_eval = NoisyNetwork(input_size, hidden_size, num_layers, lr, n_actions)
        else:
            self.Q_eval = Network(input_size, hidden_size, num_layers, lr, n_actions)
        
        self.state_memory = np.zeros((self.mem_size, seq_len, input_size), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, seq_len, input_size), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, self.seq_len), dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)
        
        if self.DDQN == True:
            if self.noisyNet == True:
                self.Q_target = NoisyNetwork(input_size, hidden_size, num_layers, lr, n_actions)
            else:
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
        
        #if noisyNet is True, then greedy=True
        if self.noisyNet==True:
            eps_greedy = False
            softmax = False
            greedy = True
                     
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
        terminal_batch = torch.tensor(self.terminal_memory[batch])
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
            q_next_max_mean = torch.mean(q_next,dim=1).unsqueeze(-1).repeat(1,self.seq_len) # average Q value for all heads
            
            reward_batch = reward_batch.unsqueeze(-1).repeat(1,self.seq_len) #repeats the reward for each head
            # q_target = reward_batch + self.gamma * q_next # does not take the average of the heads
            q_target = reward_batch + self.gamma * q_next_max_mean

        loss = self.Q_eval.loss(q_eval, q_target)

        self.Q_eval.zero_grad()        
        loss.backward()
        self.Q_eval.optimizer.step()
        
        if self.noisyNet == True:
            self.Q_eval.reset_noise()
        
        if self.DDQN == True:
            self.target_cntr += 1
            if self.target_cntr % self.update_target == 0:
                self.Q_target.load_state_dict(self.Q_eval.state_dict())
                self.target_cntr = 0