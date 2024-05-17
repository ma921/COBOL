import os
import torch
import botorch
import pandas as pd
from ._domain import UniformDomain
from ._gaussian_process import SimpleGP
from ._synthetic_agent import SyntheticAgentResponse
from ._likelihood_ratio import LikelihoodRatioConfidenceSetModel
from ._haibo import HAIBO
from ._dataset import DatasetManager
from ._experiments import select_experiment

def initial_setting(
    experiment,
    method,
    n_init = 3,
    n_init_agent = 10,
    n_iterations = 50,
    trust_weight=3,
    beta_coeff=0.01,
    g_norm_bound=3,
    a=1,
    b=0,
    terminate=5,
    balanced=True,
    seed=0,
    use_primal_dual=False,
    dual_var = 1.,
    g_thr = 0.08,
):
    torch.manual_seed(seed)
    
    home_dir = "./results"
    if not os.path.exists(home_dir):
        os.makedirs(home_dir)
    
    save_path = home_dir +  f"/{experiment}_{method}_n_init{n_init}_n_init_agent{n_init_agent}_trust_weight{trust_weight}_beta_coeff{beta_coeff}_g_norm_bound{g_norm_bound}_a{a}_use_primal_dual{use_primal_dual}_dual_var{dual_var}_g_thr{g_thr}"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    test_function, domain, global_minimum = select_experiment(experiment)
    agent = SyntheticAgentResponse(test_function, a=a, b=b)

    train_X = domain.sample(n_init)
    train_Y = test_function(train_X)
    train_X_agent, train_Y_agent = agent.sample(domain, n_init_agent, balanced=balanced)
    dm = DatasetManager(train_Y, train_Y_agent, global_minimum, save_path, seed)

    model = SimpleGP(train_X, train_Y, domain)
    model_const = LikelihoodRatioConfidenceSetModel(
        train_X_agent, 
        train_Y_agent, 
        domain, 
        beta_coeff=beta_coeff,
        kernel=model.gp.covar_module, 
        g_norm_bound=g_norm_bound,
    )
    haibo = HAIBO(
        model, 
        model_const,
        n_iterations, 
        trust_weight=trust_weight, 
        terminate=terminate,
        use_primal_dual=use_primal_dual,
        dual_var = dual_var,
        g_thr = g_thr,
    )
    haibo.initialisation(train_X, train_Y, train_X_agent, train_Y_agent)
    return dm, haibo, agent, test_function, train_X, train_Y, train_X_agent, train_Y_agent

def HAIBO_loop(
    experiment, n_init, n_init_agent, n_iterations, trust_weight, beta_coeff, g_norm_bound, a, seed, use_primal_dual, dual_var, g_thr
):
    dm, haibo, agent, test_function, train_X, train_Y, train_X_agent, train_Y_agent = initial_setting(
        experiment,
        "HAIBO",
        n_init = n_init,
        n_init_agent = n_init_agent,
        n_iterations = n_iterations,
        trust_weight=trust_weight,
        beta_coeff=beta_coeff,
        g_norm_bound=g_norm_bound,
        a=a,
        b=0,
        terminate=5,
        balanced=True,
        seed=seed,
        use_primal_dual=use_primal_dual,
        dual_var=dual_var,
        g_thr=g_thr,
    )
    while dm.t < n_iterations:
        dm.tik(train_Y, train_Y_agent)
        X_next, flag_query_agent = haibo.query(dm.t_function)
        if flag_query_agent:
            print("Asking agents...")
            Y_agent_next = agent(X_next)
            train_X_agent, train_Y_agent = dm.cat_dataset(train_X_agent, train_Y_agent, X_next, Y_agent_next)
            haibo.model_const.train(train_X_agent, train_Y_agent)
            if Y_agent_next == 0:
                Y_next = test_function(X_next)
                train_X, train_Y, log = dm.tok(train_X, train_Y, X_next, Y_next, train_Y_agent)
                haibo.model.train(train_X, train_Y)
                haibo.model_const.kernel_initialisation(haibo.model.gp.covar_module)
        else:
            Y_next = test_function(X_next)
            train_X, train_Y, log = dm.tok(train_X, train_Y, X_next, Y_next, train_Y_agent)
            haibo.model.train(train_X, train_Y)
            haibo.model_const.kernel_initialisation(haibo.model.gp.covar_module)
    return log

def UCB_loop(
    experiment, n_init, n_init_agent, n_iterations, trust_weight, 
    beta_coeff, g_norm_bound, a, seed, use_primal_dual, dual_var, g_thr
):
    dm, haibo, agent, test_function, train_X, train_Y, train_X_agent, train_Y_agent = initial_setting(
        experiment,
        "UCB",
        n_init = n_init,
        n_init_agent = n_init_agent,
        n_iterations = n_iterations,
        trust_weight=trust_weight,
        beta_coeff=beta_coeff,
        g_norm_bound=g_norm_bound,
        a=a,
        b=0,
        terminate=5,
        balanced=True,
        seed=seed,
        use_primal_dual=use_primal_dual,
        dual_var=dual_var,
        g_thr=g_thr,
    )
    while dm.t < n_iterations:
        dm.tik(train_Y, train_Y_agent)
        haibo.model.train(train_X, train_Y)
        X_next, _, _ = haibo.minimise_unconstrained_lcb(dm.t_function)
        Y_next = test_function(X_next)
        train_X, train_Y, log = dm.tok(train_X, train_Y, X_next, Y_next, train_Y_agent)
    return log

def run_loop(
    experiment, method, n_init, n_init_agent, n_iterations, trust_weight, beta_coeff, g_norm_bound, a, seed, use_primal_dual, dual_var, g_thr
):
    if method == "HAIBO":
        log = HAIBO_loop(
            experiment, n_init, n_init_agent, n_iterations, trust_weight, beta_coeff, g_norm_bound, a, seed, use_primal_dual, dual_var, g_thr
        )
    elif method == "UCB":
        log = UCB_loop(
            experiment, n_init, n_init_agent, n_iterations, trust_weight, beta_coeff, g_norm_bound, a, seed, use_primal_dual, dual_var, g_thr
        )
    return log