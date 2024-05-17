import time
import torch
import casadi
import numpy as np
from ._solver_helper import set_domain_bounds, set_likelihood_bounds, solve_opti
from ._acquisition_function import AcquisitionFunction


class HaiboOpt:
    def minimise_unconstrained_lcb(self, t):
        acqf = AcquisitionFunction(t, self.model, self.n_iterations, is_LCB=True, tol=self.tol, gamma=self.gamma, R=self.R)
        X_LCB_unconst, LCB_unconst = acqf.optimize()
        return X_LCB_unconst, LCB_unconst.item(), acqf.beta
    
    def minimise_unconstrained_ucb(self, t, beta):
        acqf = AcquisitionFunction(t, self.model, self.n_iterations, is_LCB=False, beta=beta, tol=self.tol, gamma=self.gamma, R=self.R)
        X_UCB_unconst, UCB_unconst = acqf.optimize()
        return X_UCB_unconst, UCB_unconst.item()
    
    def single_minimise_constrained_lcb(self, x0, t, beta):
        # initial setting
        opti = casadi.Opti()
        var_x = opti.variable(self.model.n_dims)
        opti.set_initial(var_x, x0)
        opti = set_domain_bounds(opti, self.model, var_x)

        # symbolic objective setting
        acqf = AcquisitionFunction(t, self.model, self.n_iterations, is_LCB=True, beta=beta, tol=self.tol, gamma=self.gamma, R=self.R)
        lcb = acqf.symbolic_LCB(beta, var_x)

        # constraint setting
        var_y = opti.variable(self.model_const.n_data_agent + 1)
        opti.set_initial(var_y, [0]*(self.model_const.n_data_agent + 1))
        opti = set_likelihood_bounds(opti, var_y, self.model_const.n_data_agent+1, self.model_const.g_ub, self.model_const.g_lb)
        
        # loglikelihood of the binary inputs
        beta_g = self.model_const.beta_g_func(self.model_const.n_data_agent)
        const1 = self.model_const.likelihood_constraint(var_y, beta_g)
        opti.subject_to(const1)

        # likelihood ratio
        const2 = self.model_const.confindence_set_constraint(var_x, var_y)
        opti.subject_to(const2)

        # solve
        if self.use_primal_dual:
            obj = lcb + self.dual_var * var_y[-1]
            x_solution, val_solution, gubar_solution = solve_opti(opti, var_x, obj, var_y=var_y)
            return x_solution, val_solution, gubar_solution
        else:
            opti.subject_to(var_y[-1] <= 0)
            x_solution, val_solution = solve_opti(opti, var_x, lcb)
            return x_solution, val_solution
    
    def minimise_constrained_lcb(self, t, beta, n_restart=10, terminate=1):
        x_init = self.model.domain.sample(n_restart)
        x_init = x_init[torch.multinomial(torch.ones(n_restart), n_restart)].numpy()
        converge_sol_list = []
        converge_val_list = []
        converge_gubar_list = []
        for x0 in x_init:
            tik = time.monotonic()
            if self.use_primal_dual:
                x_solution, val_solution, gubar_solution = self.single_minimise_constrained_lcb(x0, t, beta)
            else:
                x_solution, val_solution = self.single_minimise_constrained_lcb(x0, t, beta)
            tok = time.monotonic()
            converge_sol_list.append(x_solution)
            converge_val_list.append(val_solution)
            if self.use_primal_dual:
                converge_gubar_list.append(gubar_solution)
            if (tok - tik) > terminate:
                break

        LCB_const = np.min(converge_val_list)
        best_conv_sol_id = np.argmin(converge_val_list)
        X_LCB_const_norm = torch.from_numpy(np.squeeze(converge_sol_list[best_conv_sol_id])).unsqueeze(0).to(self.dtype)
        X_LCB_const = self.model.domain.untransform_X(X_LCB_const_norm)
        
        if self.use_primal_dual:
            best_gubar = converge_gubar_list[best_conv_sol_id]
            self.dual_var = np.max(
                self.dual_var + self.dual_step_size * best_gubar, 
                0,
            )
        
        return X_LCB_const, LCB_const


class HAIBO(HaiboOpt):
    def __init__(
        self, 
        model, 
        model_const, 
        n_iterations, 
        trust_weight=3., 
        tol=1e-2, 
        gamma=20, 
        R=None,
        almost_equal_thr=1e-5,
        n_restart=10,
        terminate=5,
        g_thr = 0.08,
        use_primal_dual = True,
        dual_step_size = 0.02,
        dual_var = 1.,
    ):
        self.n_iterations = n_iterations
        self.tol=tol
        self.gamma=gamma
        self.R=R
        self.model = model
        self.model_const = model_const
        self.dtype = model_const.dtype
        self.trust_weight = trust_weight
        self.almost_equal_thr = almost_equal_thr
        self.n_restart = n_restart
        self.terminate = terminate
        self.g_thr = g_thr
        self.use_primal_dual = use_primal_dual
        self.dual_step_size = dual_step_size
        self.dual_var = dual_var
        
    def clean_X(self, X_next):
        X = X_next.nan_to_num().squeeze()
        X = torch.clamp(X, min=self.model.domain.bounds[0], max=self.model.domain.bounds[1])
        X_next = X.unsqueeze(0)
        return X_next
        
    def initialisation(self, train_X, train_Y, train_X_agent, train_Y_agent):
        self.model.train(train_X, train_Y)
        self.model_const.kernel_initialisation(self.model.gp.covar_module)
        self.model_const.train(train_X_agent, train_Y_agent)
        
    def judge(self, UCB_unconst, LCB_const, LCB_unconst, stddev_UCB_unconst, stddev_LCB_const):
        const1 = (UCB_unconst < LCB_const)
        const2 = (stddev_UCB_unconst > self.trust_weight *stddev_LCB_const)
        const3 = (abs(LCB_const - LCB_unconst) > self.almost_equal_thr)
        return const1 or const2 and const3
    
    def query(self, t):
        X_LCB_unconst, LCB_unconst, beta = self.minimise_unconstrained_lcb(t)
        X_UCB_unconst, UCB_unconst = self.minimise_unconstrained_ucb(t, beta)
        X_LCB_const, LCB_const = self.minimise_constrained_lcb(t, beta, n_restart=self.n_restart, terminate=self.terminate)
        _, stddev_UCB_unconst = self.model.predictive_mean_and_stddev(X_UCB_unconst)
        _, stddev_LCB_const = self.model.predictive_mean_and_stddev(X_LCB_const)
        flag_query_agent = False
        if self.judge(UCB_unconst, LCB_const, LCB_unconst, stddev_UCB_unconst, stddev_LCB_const):
            X_next = X_LCB_unconst
            print("Vanilla BO is selected")
        else:
            X_next = X_LCB_const
            print("Agmented BO is selected")
            g_mle, g_LCB, g_UCB = self.model_const.predictive_distribution(X_next, likelihood=False)
            if g_UCB - g_LCB > self.g_thr:
                flag_query_agent = True
        X_next = self.clean_X(X_next)
        return X_next, flag_query_agent
