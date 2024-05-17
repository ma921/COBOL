import copy
import torch
import numpy as np
import casadi
import gpytorch
from ._solver_helper import set_likelihood_bounds, solve_opti


class SymbolicConstraints:
    def symbolic_KxX(self, var_x, hypers=None):
        if hypers is not None:
            lengthscales = hypers
        else:
            lengthscales = self.kernel.base_kernel.lengthscale.detach().squeeze().numpy()
        
        outputscale = self.kernel.outputscale.detach().numpy()
        diff_x_scaled = (var_x - self.train_X_agent.numpy().T) / lengthscales
        diff_x_scaled_inner_prod = diff_x_scaled[:, 0].T @ diff_x_scaled[:, 0]

        for i in range(1, self.n_data_agent):
            diff_x_scaled_inner_prod = casadi.vertcat(
                diff_x_scaled_inner_prod,
                diff_x_scaled[:, i].T @ diff_x_scaled[:, i]
            )

        if type(self.kernel.base_kernel) == gpytorch.kernels.rbf_kernel.RBFKernel:
            KxX = outputscale * casadi.exp(- 0.5 * diff_x_scaled_inner_prod)
        return KxX
    
    def log_likelihood(self, var_y):
        return sum(
            var_y[k] * self.train_Y_agent[k].item()
            for k in range(self.n_data_agent)
        ) - sum(
            casadi.log(1 + casadi.exp(var_y[k]))
            for k in range(self.n_data_agent)
        )
    
    def likelihood_constraint(self, var_y, beta_g):
        LL = self.log_likelihood(var_y)
        const = (LL >= self.LL_max - beta_g)
        return const
    
    def confindence_set_constraint(self, var_x, var_y):
        constr_KxX = self.symbolic_KxX(var_x)
        constr_input_cov = self.kernel(self.train_X_agent).to_dense() + \
            torch.eye(self.n_data_agent).to(self.dtype) * self.numerical_epsilon
        constr_input_cov_inv = (constr_input_cov).inverse().detach().numpy()
        constr_Kxx = self.kernel.outputscale.detach().numpy().item()
        covariance = constr_Kxx - constr_KxX.T @ constr_input_cov_inv @ constr_KxX
        Px = casadi.sqrt(casadi.fmax(covariance, 1e-10)) + 1e-15
        inv_1 = casadi.MX.zeros(self.n_data_agent+1, self.n_data_agent+1)
        inv_1[:self.n_data_agent, :self.n_data_agent] = constr_input_cov_inv
        inv_2_sqrt = casadi.MX.zeros(self.n_data_agent+1, 1)
        Kinv_k = constr_input_cov_inv @ constr_KxX
        inv_2_sqrt[:self.n_data_agent, :] = Kinv_k
        inv_2_sqrt[-1, :] = -1
        inv_2 = (inv_2_sqrt @ inv_2_sqrt.T) / (Px**2)
        const = (var_y.T @ (inv_1+inv_2) @ var_y <= self.g_norm_bound**2)
        return const
    
    def training_loop(self, X, hypers=None):
        N = len(X)
        opti = casadi.Opti()
        var_y = opti.variable(N)
        opti.set_initial(var_y, [0] * N)
        opti = set_likelihood_bounds(opti, var_y, N, self.g_ub, self.g_lb)
        LL = self.log_likelihood(var_y)
        if hypers is not None:
            hypers_opt = opti.variable(len(hypers))
            opti.set_initial(hypers_opt, hypers)
            cov = self.symbolic_KxX(X.numpy().T, hypers=hypers_opt)
            input_cov = cov + np.eye(N) * self.numerical_epsilon
            input_cov_inv = casadi.pinv(input_cov)
            const = var_y.T @ input_cov_inv @ var_y <= self.g_norm_bound**2
            opti.subject_to(const)
            return opti, var_y, LL, hypers_opt
        else:
            cov = self.kernel(X).to_dense()
            input_cov = cov + torch.eye(N).to(self.dtype) * self.numerical_epsilon
            input_cov_inv = (input_cov).inverse().detach().numpy()
            const = var_y.T @ input_cov_inv @ var_y <= self.g_norm_bound**2
            opti.subject_to(const)
            return opti, var_y, LL

    
class KernelManager:
    def kernel_initialisation(self, kernel):
        if kernel is not None:
            self.kernel = copy.deepcopy(kernel)
        else:
            self.kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=self.n_dims_agent, dtype=self.dtype),
            )
        self.hypers_init = self.recall_kernel_hypers()
            
    def recall_kernel_hypers(self):
        return self.kernel.base_kernel.lengthscale.detach()[0].numpy()
    
    def manual_set_kernel(self, hypers):
        hyperparameters = {
            'base_kernel.lengthscale': hypers,
        }
        self.kernel.initialize(**hyperparameters).eval()
        
    def set_kernel(self, hypers):
        try:
            self.manual_set_kernel(hypers)
        except:
            print("kernel training failed. Hyperparamters are initialised.")
            self.manual_set_kernel(self.hypers_init)
        
    def set_kernel_range(self, opti, hypers, rng=5):
        for k in range(self.n_dims_agent):
            opti.subject_to(
                hypers[k] <= self.hypers_init[k].item() * rng,
            )
            opti.subject_to(
                hypers[k] >= self.hypers_init[k].item() / rng,
            )
        return opti


class PredictiveDistribution:
    def check_x(self, x):
        if not x.shape[0] == 1:
            raise ValueError("The input x should be the shape [1, dim].")
        elif len(x.shape) == 1:
            x = x.unsqueeze(0)
        return x
    
    def lower_confidence_bound(self, x, likelihood=False):
        x = self.check_x(x)
        var_x = torch.cat([self.train_X_agent, x])
        opti, var_y, LL = self.training_loop(var_x)
        lcb = var_y[-1]
        const = (LL >= self.LL_max - self.beta_g_func(self.n_data_agent))
        opti.subject_to(const)
        y_lcb, _ = solve_opti(opti, var_y, lcb)
        if likelihood:
            return self.preferential_likelihood(y_lcb[-1])
        else:
            return y_lcb[-1]
    
    def upper_confidence_bound(self, x, likelihood=False):
        x = self.check_x(x)
        var_x = torch.cat([self.train_X_agent, x])
        opti, var_y, LL = self.training_loop(var_x)
        ucb = -var_y[-1]
        const = (LL >= self.LL_max - self.beta_g_func(self.n_data_agent))
        opti.subject_to(const)
        y_ucb, _ = solve_opti(opti, var_y, ucb)
        if likelihood:
            return self.preferential_likelihood(y_ucb[-1])
        else:
            return y_ucb[-1]
    
    def confidence_bound(self, X_next, likelihood=False):
        g_lcb = self.lower_confidence_bound(X_next, likelihood=likelihood)
        g_ucb = self.upper_confidence_bound(X_next, likelihood=likelihood)
        return g_lcb, g_ucb
    
    def predictive_mle(self, x, likelihood=False):
        x = self.check_x(x)
        var_x = torch.cat([self.train_X_agent, x])
        opti, var_y, LL = self.training_loop(var_x)
        y_mle, _ = solve_opti(opti, var_y, LL, maximize=True)
        if likelihood:
            return self.preferential_likelihood(y_mle[-1])
        else:
            return y_mle[-1]
    
    def preferential_likelihood(self, y):
        return 1 / (1 + np.exp(-y))
    
    def predictive_distribution(self, x, likelihood=False):
        if x.shape[0] == 1:
            y_lcb, y_ucb = self.confidence_bound(x, likelihood=likelihood)
            y_mle = self.predictive_mle(x, likelihood=likelihood)
            y_ucb_protected = max(y_ucb, y_mle)
            y_lcb_protected = min(y_lcb, y_mle)
        else:
            y_lcb = torch.tensor([self.lower_confidence_bound(X.unsqueeze(0), likelihood=likelihood) for X in x]).to(self.dtype)
            y_ucb = torch.tensor([self.upper_confidence_bound(X.unsqueeze(0), likelihood=likelihood) for X in x]).to(self.dtype)
            y_mle = torch.tensor([self.predictive_mle(X.unsqueeze(0), likelihood=likelihood) for X in x]).to(self.dtype)
            y_ucb_protected = torch.vstack([y_mle, y_ucb]).max(axis=0).values
            y_lcb_protected = torch.vstack([y_mle, y_lcb]).min(axis=0).values
        return y_mle, y_lcb_protected, y_ucb_protected


class HyperparameterManager:
    def update_beta(self, beta_coeff=None):
        if beta_coeff is not None:
            self.beta_coeff = beta_coeff
            self.beta_g_func = lambda t: beta_coeff * np.sqrt(t+1)
        else:
            self.beta_coeff = 2 * self.beta_coeff
            self.beta_g_func = lambda t: self.beta_coeff * np.sqrt(t+1)
            print(f'Updated. norm_{self.g_norm_bound} and alpha {self.beta_coeff}')
            
    def hypothesis_test(self, kernel=None):
        g_norm_bound_old = copy.deepcopy(self.g_norm_bound)
        LL_max_old = copy.deepcopy(self.LL_max)
        self.g_norm_bound = 2 * g_norm_bound_old
        LL_max_y_new, LL_max_new = self.mle(kernel=kernel)
        LL_thr = 2 * self.beta_g_func(self.n_data_agent)
        if LL_max_new - LL_max_old > LL_thr:
            self.update_beta()
            self.LL_max_y, self.LL_max = LL_max_y_new, LL_max_new
        else:
            self.g_norm_bound = g_norm_bound_old
            
    def initial_train(self, train_X_agent, train_Y_agent):
        for length in range(1, len(train_X_agent)+1):
            self.conditioning(train_X_agent[:length], train_Y_agent[:length])
            self.LL_max_y, self.LL_max = self.mle()
            self.hypothesis_test()

class LikelihoodRatioConfidenceSetModel(SymbolicConstraints, KernelManager, PredictiveDistribution, HyperparameterManager):
    def __init__(
        self,
        train_X_agent,
        train_Y_agent,
        domain,
        kernel=None,
        g_lb=-3,
        g_ub=3,
        numerical_epsilon=1e-6,
        beta_coeff=0.01,
        g_norm_bound=1,
        train_kernel=False,
        g_norm_bound_update=True,
        g_norm_bound_update_period=3,
    ):
        self.domain = domain
        self.numerical_epsilon = numerical_epsilon
        self.dtype = train_X_agent.dtype
        self.g_lb, self.g_ub = g_lb, g_ub
        self.g_norm_bound = g_norm_bound
        self.kernel_initialisation(kernel)
        self.train_kernel = train_kernel
        self.g_norm_bound_update = g_norm_bound_update
        self.g_norm_bound_update_period = g_norm_bound_update_period
        self.update_beta(beta_coeff)
        self.initial_train(train_X_agent, train_Y_agent)
    
    def conditioning(self, train_X_agent, train_Y_agent):
        self.train_X_agent = self.domain.transform_X(train_X_agent)
        self.train_Y_agent = train_Y_agent
        self.n_data_agent, self.n_dims_agent = train_X_agent.shape
        
    def mle(self, kernel=None, rng=5):
        if self.train_kernel and (kernel is not None):
            self.kernel_initialisation(kernel)
            opti, var_y, LL, hypers = self.training_loop(self.train_X_agent, hypers=self.hypers_init)
            opti = self.set_kernel_range(opti, hypers, rng=rng)
            LL_max_y, _, hypers = solve_opti(opti, var_y, LL, maximize=True, hypers=hypers)
            LL_max = self.log_likelihood(LL_max_y)
            self.set_kernel(hypers)
        else:
            opti, var_y, LL = self.training_loop(self.train_X_agent)
            LL_max_y, LL_max = solve_opti(opti, var_y, LL, maximize=True)
        return LL_max_y, LL_max
    
    def train(self, train_X_agent, train_Y_agent, kernel=None, rng=5):
        self.conditioning(train_X_agent, train_Y_agent)
        self.LL_max_y, self.LL_max = self.mle(kernel=kernel, rng=5)
        if self.g_norm_bound_update and (self.n_data_agent % self.g_norm_bound_update_period == 0):
            self.hypothesis_test(kernel=kernel)
            