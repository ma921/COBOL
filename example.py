import torch
import botorch
import matplotlib.pyplot as plt
from utils._experiments import FunctionWrapper
from utils._domain import UniformDomain
from utils._gaussian_process import SimpleGP
from utils._synthetic_agent import SyntheticAgentResponse
from utils._likelihood_ratio import LikelihoodRatioConfidenceSetModel


# 1.Problem definition
n_dims = 1
dtype = torch.float64
noise_std = None

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

# 2. Set up initial dataset
torch.manual_seed(0)

n_init = 7
n_init_agent = 10
g_norm_bound = 1
confidence_bound=1e-8

x_grid = torch.linspace(-1,1,100).unsqueeze(-1).to(dtype)
y_grid = test_function(x_grid)
agent = SyntheticAgentResponse(test_function, a=1, b=0)

train_X = domain.sample(n_init)
train_Y = test_function(train_X)
train_X_agent, train_Y_agent = agent.sample(domain, n_init_agent, balanced=True)
print(train_Y_agent.sum())


# 3. set up and train models
model = SimpleGP(train_X, train_Y, domain)
model.train(train_X, train_Y)
model_const = LikelihoodRatioConfidenceSetModel(
    train_X_agent, train_Y_agent, domain, 
    kernel=model.gp.covar_module, 
    g_norm_bound=g_norm_bound,
    confidence_bound=confidence_bound,
    train_kernel=False,
)
model_const.train(train_X_agent, train_Y_agent)

# 4. prediction at grid
x_grid_norm = domain.transform_X(x_grid)
y_grid_norm = domain.transform_Y(y_grid)

obj_mean, obj_stdddev = model.predictive_mean_and_stddev(x_grid_norm)
obj_mean, obj_stdddev = obj_mean.detach(), obj_stdddev.detach()

const_mean, const_lcb, const_ucb = model_const.predictive_distribution(x_grid_norm)

# 5. Visualise
plt.plot(x_grid_norm, y_grid_norm, color="k")
plt.scatter(domain.transform_X(train_X), domain.transform_Y(train_Y), color="k")
plt.plot(x_grid_norm, obj_mean, color="b")
plt.fill_between(x_grid_norm.squeeze(), obj_mean+obj_stdddev, obj_mean-obj_stdddev, color="b", alpha=0.2)

plt.scatter(domain.transform_X(train_X_agent), train_Y_agent, color="r")
plt.plot(x_grid_norm, const_mean, color="r")
plt.fill_between(x_grid_norm.squeeze(), const_ucb, const_lcb, color="r", alpha=0.2)