import casadi
import numpy as np


def set_domain_bounds(opti, model, var_x):
    n_dims = model.domain.normed_bounds.shape[-1]
    for k in range(n_dims):
        opti.subject_to(
            var_x[k] <= model.domain.normed_bounds[1][k].item()
        )
        opti.subject_to(
            var_x[k] >= model.domain.normed_bounds[0][k].item()
        )
    return opti

def set_likelihood_bounds(opti, var_y, n_data, g_ub, g_lb):
    for k in range(n_data):
        opti.subject_to(
            var_y[k] <= g_ub
        )
        opti.subject_to(
            var_y[k] >= g_lb
        )
    return opti

def solve_opti(opti, var_x, obj, maximize=False, hypers=None, var_y=None):
    if maximize:
        opti.minimize(-obj)
    else:
        opti.minimize(obj)
    
    opti.solver(
        'ipopt',
        dict(print_time=False),
        dict(print_level=False, max_iter=50, tol=1e-2, constr_viol_tol=1e-20)
    )
    try:
        sol = opti.solve()
        x_solution = sol.value(var_x)
        val_solution = sol.value(obj)
        if var_y is not None:
            gubar_solution = sol.value(var_y[-1])
        if hypers is not None:
            hypers_solution = sol.value(hypers)
    except Exception as e:
        x_solution = opti.debug.value(var_x)
        val_solution = opti.debug.value(obj)
        if var_y is not None:
            gubar_solution = opti.debug.value(var_y[-1])
        if hypers is not None:
            hypers_solution = opti.debug.value(hypers)
            
    if hypers is not None:
        return x_solution, val_solution, hypers_solution
    elif var_y is not None:
        return x_solution, val_solution, gubar_solution
    else:
        return x_solution, val_solution

