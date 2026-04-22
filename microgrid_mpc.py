import numpy as np
# from qpsolvers import Problem, solve_problem, solve_qp # type: ignore
from cvxopt import matrix,solvers
import gurobipy as gp
from gurobipy import GRB

class MicrogridMPC:
    def __init__(self, N, mpc_param_dict, n_threads):
        
        #TODO: update function description
        
        """
        Initialize MPC controller for linear system x+ = Ax + B1u + B2delta + B3z
        s.t. E2 delta + E3z <= E1u + E4x + E5
        
        Parameters:
        A: System dynamics matrix (nx x nx)
        B: Input matrix (nx x nu)
        Q: State cost matrix (nx x nx)
        R: Input cost matrix (nu x nu)
        N: Prediction horizon
        x_min: State constraints (min) (nx x 1)
        x_max: State constraints (max) (nx x 1)
        u_min: Input constraints (min) (nu x 1)
        u_max: Input constraints (max) (nu x 1)
        H: Matrix for linear state constraints Hx <= b (nc x nx)
        b: Vector for linear state constraints Hx <= b (nc x 1)
        P: Terminal state cost matrix (if None, P = Q)
        """
                
        self.A = mpc_param_dict['A']
        self.B1 = mpc_param_dict['B1']
        self.B2 = mpc_param_dict['B2']
        self.B3 = mpc_param_dict['B3']
        self.E1 = mpc_param_dict['E1']
        self.E2 = mpc_param_dict['E2']
        self.E3 = mpc_param_dict['E3']
        self.E4 = mpc_param_dict['E4']
        self.E5 = mpc_param_dict['E5']
        self.N = N
        self.B_u_a = np.block([self.B1,self.B3])
        self.n_threads=n_threads
        
        # System dimensions
        self.nx = self.A.shape[0]  # number of states
        self.nu = self.B1.shape[1]  # number of continuous inputs
        self.n_delta = self.B2.shape[1] # number of discrete inputs
        self.nz = self.B3.shape[1] # number of continuous auxiliary variables
        
        # Pre-compute prediction matrices
        self._compute_prediction_matrices()
        
        # Pre-compute optimization problem matrices
        self._precompute_optimization_matrices()
        
        # for gurobi
        #for gurobi
        self.x_min = mpc_param_dict['x_min']
        self.x_max = mpc_param_dict['x_max']
        u_min = mpc_param_dict['u_min']
        u_max = mpc_param_dict['u_max']
        z_min = mpc_param_dict['z_min']
        z_max = mpc_param_dict['z_max']
        u_a_min = mpc_param_dict['u_a_min']
        u_a_max = mpc_param_dict['u_a_max']
        
        self.x_min_tile = np.tile(self.x_min, (self.N+1,))
        self.x_max_tile = np.tile(self.x_max, (self.N+1,))
        self.u_min_tile = np.tile(u_min, (self.N,))
        self.u_max_tile = np.tile(u_max, (self.N,))
        self.z_min_tile = np.tile(z_min, (self.N,))
        self.z_max_tile = np.tile(z_max, (self.N,))
        self.u_a_min_tile = np.tile(u_a_min, (self.N,))
        self.u_a_max_tile = np.tile(u_a_max, (self.N,))
        
    def _compute_prediction_matrices(self):
        """Compute matrices for prediction"""
        # Initialize prediction matrices
        self.Gamma_x = np.zeros((self.nx * (self.N + 1), self.nx))
        self.Gamma_u = np.zeros((self.nx * (self.N + 1), self.nu * self.N))
        self.Gamma_u_a = np.zeros((self.nx * (self.N + 1), (self.nu+self.nz) * self.N))
        self.Gamma_delta = np.zeros((self.nx * (self.N + 1), self.n_delta * self.N))
        self.Gamma_z = np.zeros((self.nx * (self.N + 1), self.nz * self.N))
        
        # Fill first block of Gamma_x
        self.Gamma_x[0:self.nx, :] = np.eye(self.nx)
        temp = np.eye(self.nx)        
        # Fill remaining blocks of Gamma_x
        for i in range(self.N):
            temp = np.dot(self.A, temp)
            self.Gamma_x[(i+1)*self.nx:(i+2)*self.nx, :] = temp

        # Fill blocks of Gamma_u
        list_coeff_Gamma_u = [self.B1]
        for i in range(1,self.N):
            list_coeff_Gamma_u.append(self.A@list_coeff_Gamma_u[-1])    
        # self.Gamma_u=np.zeros(((self.N+1)*self.nx, self.N*self.nu))
        # iterates per column then per row
        for j in range(self.N):
            for i in range(j,self.N):
                self.Gamma_u[self.nx*(i+1): self.nx*(i+2), self.nu*j: self.nu*(j+1)] = list_coeff_Gamma_u[i-j]
                       
        # Fill blocks of Gamma_u_a
        list_coeff_Gamma_u_a = [np.block([self.B1,self.B3])]
        for i in range(1,self.N):
            list_coeff_Gamma_u_a.append(self.A@list_coeff_Gamma_u_a[-1])    
        # self.Gamma_u=np.zeros(((self.N+1)*self.nx, self.N*self.nu))
        # iterates per column then per row
        for j in range(self.N):
            for i in range(j,self.N):
                self.Gamma_u_a[self.nx*(i+1): self.nx*(i+2), (self.nu+self.nz)*j: (self.nu+self.nz)*(j+1)] = list_coeff_Gamma_u_a[i-j]
                
        # Fill blocks of Gamma_delta
        list_coeff_Gamma_delta = [self.B2]
        for i in range(1,self.N):
            list_coeff_Gamma_delta.append(self.A@list_coeff_Gamma_delta[-1])
        # iterates per column then per row
        for j in range(self.N):
            for i in range(j,self.N):
                self.Gamma_delta[self.nx*(i+1): self.nx*(i+2), self.n_delta*j: self.n_delta*(j+1)] = list_coeff_Gamma_delta[i-j]
                
        # Fill blocks of Gamma_z
        list_coeff_Gamma_z = [self.B3]
        for i in range(1,self.N):
            list_coeff_Gamma_z.append(self.A@list_coeff_Gamma_z[-1])
        # iterates per column then per row
        for j in range(self.N):
            for i in range(j,self.N):
                self.Gamma_z[self.nx*(i+1): self.nx*(i+2), self.nz*j: self.nz*(j+1)] = list_coeff_Gamma_z[i-j]
                
    def _precompute_optimization_matrices(self):
        self.E1_a = np.block([
            [-self.E1, self.E3]
        ])    
        
        self.E1_a_blk = np.kron(np.eye(self.N), self.E1_a)
        self.E1_blk = np.kron(np.eye(self.N), self.E1)
        self.E2_blk = np.kron(np.eye(self.N), self.E2)    
        self.E3_blk = np.kron(np.eye(self.N), self.E3)    
        self.E4_blk = np.kron(np.eye(self.N), self.E4)    
        self.E5_conc = np.tile(self.E5, (self.N,1)).reshape(-1,)
        
        self.G_mld = (self.E1_a_blk - self.E4_blk@self.Gamma_u_a[:self.N,:])
        
        #power balance constraint
        G_eq = np.array([[1, -1, -1, -1, -1, 0, 0]])
        self.G_eq_blk = np.kron(np.eye(self.N), G_eq)
        
        #terminal constraint
        self.G_term = self.Gamma_u_a[self.N,:]
        
        # self.G = np.vstack([self.G, self.G_eq_blk, -self.G_eq_blk])
        self.G = np.vstack([self.G_mld, self.G_eq_blk, -self.G_eq_blk, self.G_term, -self.G_term])
        
        #for hybrid (power balance)
        G_eq_u = np.array([[1, -1, -1, -1, -1]])
        self.G_eq_u_blk = np.kron(np.eye(self.N), G_eq_u)
    
    def build_opt_matrices(self, x0, delta, power_res, power_load, cbuy, csell, cprod):
        #mld constraints
        self.h_mld = -self.E2_blk@delta + self.E4_blk@self.Gamma_x[:self.N,:]@x0 + self.E4_blk@self.Gamma_delta[:self.N,:]@delta + self.E5_conc

        #power balance
        self.h_eq = power_res-power_load
        
        #terminal constraint
        self.h_term_max = self.x_max - self.Gamma_x[self.N,:]@x0 - self.Gamma_delta[self.N,:]@delta
        self.h_term_min = -self.x_min + self.Gamma_x[self.N,:]@x0 + self.Gamma_delta[self.N,:]@delta
        
        self.h = np.concatenate([self.h_mld, self.h_eq, -self.h_eq, self.h_term_max, self.h_term_min])
        
        self.q = np.zeros((1,self.N*(self.nu+self.nz)))
        self.q[0, 0:(self.nu+self.nz)] = np.array([0, csell[0], cprod[0], cprod[0], cprod[0], 0, cbuy[0]-csell[0]])
        for i in range(1,self.N):
            self.q[0, i*(self.nu+self.nz):(i+1)*(self.nu+self.nz)] = np.array([[0, csell[i], cprod[i], cprod[i], cprod[i], 0, cbuy[i]-csell[i]]])
            # tmp = np.array([[0, csell[i], cprod[i], cprod[i], cprod[i], 0, cbuy[i]-csell[i]]])
            # self.q = np.hstack([self.q, tmp])
            
    # def solve_qpsolvers(self,solver='quadprog'):
        
    #     # 1/2 x'Px + q'u s.t. Gx <= h, Ax = b (qpsolvers standard)
               
    #     u_opt = solve_qp(np.eye((self.nu+self.nz)*self.N), self.q.T, G=self.G, h=self.h, solver=solver, verbose=False)
        
    #     #sequence of inputs
    #     return u_opt
        
    def solve_cvxopt(self,solver='glpk'):
        
        # G = np.vstack([self.G_mld, self.G_term, -self.G_term])
        # h = np.concatenate([self.h_mld, self.h_term_max, self.h_term_min])
        # A = matrix(self.G_eq_blk)
        # b = matrix(self.h_eq)
    
        # c = matrix(self.q.T)
        # G = matrix(G)
        # h = matrix(h.astype(np.double))
        
        # G = matrix(self.G)
        # h = matrix(self.h.astype(np.double))

        # sol = solvers.lp(c,G,h,A,b, solver=solver)
        
        # solvers.options['glpk'] = {'tolbnd' : 1e-3}
        # solvers.options['glpk'] = {'toldj' : 1e-3}
        
        G = matrix(self.G)
        h = matrix(self.h.astype(np.double))
        c = matrix(self.q.T)
        sol = solvers.lp(c,G,h, solver=solver)
        
        return np.array(sol['x']).reshape(-1)
    
    def solve_gurobi_lp(self):
      
        mdl = gp.Model("gurobilp")
        # mdl.Params.LogToConsole = 0
        mdl.Params.Threads = self.n_threads

        # x = mdl.addMVar(shape=((self.N+1)*self.nx,), lb=x_min_tile, ub=x_max_tile, name='x')
        u_a = mdl.addMVar(shape=(self.N*(self.nu+self.nz),), lb=self.u_a_min_tile, ub=self.u_a_max_tile, name='u_a')
        
        # mdl.addConstr(x == mpc.Gamma_x@x0 + mpc.Gamma_u_a@u_a + mpc.Gamma_delta@delta)
        # mdl.addConstr(mpc.E2_blk@delta + mpc.E1_a_blk@u_a <= mpc.E4_blk@x[:mpc.N] + mpc.E5_conc)
        # mdl.addConstr(mpc.G_eq_blk@u_a == mpc.h_eq)
        mdl.addConstr(self.G@u_a <= self.h)
            
        obj = self.q@u_a
        
        mdl.setObjective(obj, GRB.MINIMIZE)

        mdl.optimize()
    
        return mdl
    
    def build_opt_matrices_hybrid(self, power_res, power_load, cbuy, csell, cprod):   
        #power balance
        self.h_eq = power_res-power_load
                          
        self.q_u = np.zeros((1,self.N*(self.nu)))
        self.q_z = np.zeros((1,self.N*(self.nz)))
        for i in range(self.N):
            self.q_u[0, i*(self.nu):(i+1)*(self.nu)] = np.array([[0, csell[i], cprod[i], cprod[i], cprod[i]]])
            self.q_z[0, i*(self.nz):(i+1)*(self.nz)] = np.array([[0, cbuy[i]-csell[i]]])
    
    def solve_hybrid_gurobi(self, x0):
      
        mdl = gp.Model("hybrid")
        # mdl.Params.LogToConsole = 0
        mdl.Params.MIPGap = 1e-6
        mdl.Params.Threads = self.n_threads

        x = mdl.addMVar(shape=((self.N+1)*self.nx,), lb=self.x_min_tile, ub=self.x_max_tile, name='x')
        u = mdl.addMVar(shape=(self.N*self.nu,), lb=self.u_min_tile, ub=self.u_max_tile, name='u')
        z = mdl.addMVar(shape=(self.N*self.nz,), lb=self.z_min_tile, ub=self.z_max_tile, name='z')
        delta = mdl.addMVar(shape=(self.N*self.n_delta,), vtype=gp.GRB.BINARY, name='delta')
        
        mdl.addConstr(x == self.Gamma_x@x0 + self.Gamma_u@u + self.Gamma_delta@delta + self.Gamma_z@z)
        mdl.addConstr(self.E2_blk@delta + self.E3_blk@z <= self.E1_blk@u + self.E4_blk@x[:self.N] + self.E5_conc)
        mdl.addConstr(self.G_eq_u_blk@u == self.h_eq)
        
        obj = self.q_u@u + self.q_z@z
        mdl.setObjective(obj, GRB.MINIMIZE)

        mdl.optimize()
    
        return mdl