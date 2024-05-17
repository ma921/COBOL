import torch
import botorch
from ._domain import UniformDomain, FiniteDomain


class FunctionWrapper:
    def __init__(self, test_function, global_minimum, global_maximum, dtype):
        self.test_function = test_function
        self.global_minimum = torch.tensor(global_minimum).to(dtype)
        self.global_maximum = torch.tensor(global_maximum).to(dtype)
    
    def bounded_return(self, x):
        y = self.test_function(x)
        y_normalised = (y - self.global_minimum) / (self.global_maximum - self.global_minimum) # [0, 1]
        y_transformed = 6 * y_normalised - 3 # [-3, 3]
        return y_transformed
    
    def __call__(self, x):
        y = self.test_function(x)
        if not torch.all(y > self.global_minimum):
            print(f"bug in test function. It returns {y.item(): .3e}")
            y[y < self.global_minimum] = self.global_minimum
        return y


def select_experiment(function, noise_std=None, dtype=torch.float64, seed=0):
    if function == "ackley":
        return setup_ackley(n_dims=4, noise_std=noise_std, dtype=dtype)
    elif function == "holder":
        return setup_holder(n_dims=2, noise_std=noise_std, dtype=dtype)
    elif function == "michalewicz":
        return setup_michalewicz(n_dims=4, noise_std=noise_std, dtype=dtype)
    elif function == "rosenbrock":
        return setup_rosenbrock(n_dims=3, noise_std=noise_std, dtype=dtype)
    elif function == "beale":
        return setup_beale(n_dims=2, noise_std=noise_std, dtype=dtype)
    elif function == "rastringin":
        return setup_rastringin(n_dims=2, noise_std=noise_std, dtype=dtype)
    else:
        raise NotImplementedError

def setup_ackley(n_dims=4, noise_std=None, dtype=torch.float64):
    bounds = torch.vstack([-1 * torch.ones(n_dims), 1 * torch.ones(n_dims)]).to(dtype)
    domain = UniformDomain(bounds)
    _test_function = botorch.test_functions.Ackley(
        dim=n_dims,          # number of dimensions
        noise_std=noise_std, # noiseless feedback
        negate=False,         # minimisation problem
    )
    global_minimum = 0
    global_maximum = 3.6253849384403627
    test_function = FunctionWrapper(_test_function, global_minimum, global_maximum, dtype)
    return test_function, domain, global_minimum

def setup_holder(n_dims=2, noise_std=None, dtype=torch.float64):
    bounds = torch.vstack([-10 * torch.ones(2), 10 * torch.ones(2)]).to(dtype)
    domain = UniformDomain(bounds)
    _test_function = botorch.test_functions.HolderTable(
        negate=False,         # minimisation problem
    )
    global_minimum = -19.2098
    global_maximum = 0
    test_function = FunctionWrapper(_test_function, global_minimum, global_maximum, dtype)
    return test_function, domain, global_minimum

def setup_michalewicz(n_dims=5, noise_std=None, dtype=torch.float64):
    bounds = torch.vstack([torch.zeros(n_dims), 3.14 * torch.ones(n_dims)]).to(dtype)
    domain = UniformDomain(bounds)
    _test_function = botorch.test_functions.Michalewicz(
        dim=n_dims,
        negate=False,         # minimisation problem
    )
    global_minimum = -4.687658
    global_maximum = 0
    test_function = FunctionWrapper(_test_function, global_minimum, global_maximum, dtype)
    return test_function, domain, global_minimum

def setup_rosenbrock(n_dims=3, noise_std=None, dtype=torch.float64):
    bounds = torch.vstack([-5*torch.ones(n_dims), 10 * torch.ones(n_dims)]).to(dtype)
    domain = UniformDomain(bounds)
    _test_function = botorch.test_functions.synthetic.Rosenbrock(
        dim=n_dims,
        negate=False,         # minimisation problem
    )
    global_minimum = 0
    global_maximum = 1.5e4
    test_function = FunctionWrapper(_test_function, global_minimum, global_maximum, dtype)
    return test_function, domain, global_minimum

def setup_beale(n_dims=2, noise_std=None, dtype=torch.float64):
    bounds = torch.vstack([-4.5*torch.ones(n_dims), 4.5 * torch.ones(n_dims)]).to(dtype)
    domain = UniformDomain(bounds)
    _test_function = botorch.test_functions.synthetic.Beale()
    global_minimum = 0
    global_maximum = 1.5e4
    test_function = FunctionWrapper(_test_function, global_minimum, global_maximum, dtype)
    return test_function, domain, global_minimum

def setup_rastringin(n_dims=2, noise_std=None, dtype=torch.float64):
    bounds = torch.vstack([-5.12*torch.ones(n_dims), 5.12 * torch.ones(n_dims)]).to(dtype)
    domain = UniformDomain(bounds)
    _test_function = botorch.test_functions.synthetic.Rastrigin()
    global_minimum = 0
    global_maximum = 80
    test_function = FunctionWrapper(_test_function, global_minimum, global_maximum, dtype)
    return test_function, domain, global_minimum
