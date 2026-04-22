# Description

Repository containing the data and code of the paper entitled “Integrating Reinforcement Learning and Model Predictive Control for Mixed-Logical Dynamical Systems” published in the IEEE Open Journal of Control Systems, available [here](https://ieeexplore.ieee.org/document/11134093/). Corresponds to Chapter 2 of the thesis.

This repository only contains the Python scripts and notebooks. The full repository with the dataset is available as the file "chapter2.rar" [here](10.4121/277b2054-94e1-4d6a-b16c-db448bd8c4c5).

Acknowledgement: This research has received funding from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (Grant agreement No. 101018826 - CLariNet).
# Setup    
  
This project uses [`uv`] for reproducible dependency management.  
  
Developed and tested on Windows 11 with `uv` version `0.11.6` and `Python 3.11.15`.  
  
Please install `uv` first by following the official documentation.    
## Installation

Download "chapter2.rar" from [this link](10.4121/277b2054-94e1-4d6a-b16c-db448bd8c4c5), and extract the files on a folder.
  
```powershell  
cd <PATH:project_folder>  
uv python install  
uv sync --locked  
```

## Non-Python dependencies  
  
The following external dependencies are required:   
- GUROBI (version 12.0.3) with valid license
This dependency is not managed by `uv` and must be installed separately.
# Folders

analysis: contains the summarized results of the RL agents training for different hyperparameters in .xls files.

best_weights: first batch of saved trained weights in a .npy file. The weights of the SL neural networks are saved here.

data_entso: data extracted from the ENTSO platform. The data is condensed in the file "data_costs_loads_2021_2022.npy" which contains the buying price, selling price, production price, power load, renewables power generation for the whole year with sampling time of 30 minutes.
Example of loading:
"data_2021 = data[0]; data_2022 = data[1]
cbuy, csell, cprod, power_load, power_res = data_2022
cbuy_2021, csell_2021, cprod_2021, power_load_2021, power_res_2021 = data_2021"

data_milp: contains the dataset used for the training of the supervised learning approach.

new_best_weights: second batch of saved trained weights in .npy file. The weights of the RL neural networks are saved here. For each different experiment, the corresponding hyperparameters are saved as a dictionary in a file "info_exp_..." in a .npy file.

# Files

computation_time_tests.ipynb: it compares the computation time of all the approaches.

config.py: it defines several parameters which are used in the other scripts.

final_test.ipynb: it compares the optimal, supervised learning, and reinforcement learning solutions. The results shown in the tables come from this script (except for the computation time).

final_test.ipynb: generates some of the plots of the paper.

gurobi.env: defines settings for the GUROBI solver.

microgrid_env.py: defines the microgrid environment which the RL agents interacts with. It has a very similar architecture to environments in the library Gymnasium.

microgrid_fun.py: defines several functions related to optimization and data pre-processing.

microgrid_gen_profiles.ipynb: it loads the original ENTSO dataset, scales, applies down sampling, and handles missing values. The energy prices are generated as described in the paper.

microgrid_mpc.py: defines the class "MicrogridMPC" which implements the model predictive controller. The class methods allow the solution of the mixed-integer linear program or the solution of the linear program, if the discrete decision variables are given as extra inputs.

microgrid_RL_agents.py: defines several neural network architectures which are used during training and implements the deep Q-learning algorithm as a class "DQN_Agent".

microgrid_RL_main.py: trains the RL agent and saves the weights of the resulting neural network. The flag "testing=False" sets the script to be run in a computing cluster. Alternatively, "testing=True" is used for local small tests. The training progress is logged into .txt files and the corresponding hyperparameters are saved in a dictionary.

microgrid_supervised.ipynb: implements the supervised-learning-based policy to predict the discrete decision variables. The neural network's weights are trained and saved.

microgrid_sys.pdf: describes the system's equations.

mpc_hybrid_constrained_notes.pdf: notes on the matrix manipulations behind the MPC formulation.

utils.py: defines auxiliary functions to obtain the greedy action from the trained RL policy and to compute the open-loop and closed-loop optimality gaps.

The uv project is defined by the following files:
- .python-version
- pyproject.toml
- uv.lock
