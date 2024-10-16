import copy
import torch
import gpytorch
import botorch
from abc import ABC, abstractmethod
from ._utils import cleansing_x


class TemplateGP(ABC):
    @abstractmethod
    def conditioning(self, train_X, train_Y):
        r"""Conditioning GP model on the data"""
        pass
    
    def train(self):
        r"""Training the hyperparamters of the GP model"""
        pass
    
    def predictive_mean_and_stddev(self, X):
        r"""Predictive mean and standard deviation at X"""
        pass


class SimpleGP(TemplateGP):
    def __init__(self, train_X, train_Y, domain, mean_module=None, covar_module=None):
        self.domain = domain
        if mean_module is None:
            mean_module = gpytorch.means.ZeroMean()
        self.mean_module = copy.deepcopy(mean_module)
        
        if covar_module is None:
            covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=train_X.shape[-1]),
            )
        self.covar_module = copy.deepcopy(covar_module)
        #self.train(train_X, train_Y)
        self.conditioning(train_X, train_Y)
        self.gp = self.set_model()
        self.gp.eval()
        self.dtype = train_Y.dtype
        self.num_outputs = 1
        self.is_fully_bayesian = False

    def set_model(self):
        gp = botorch.models.SingleTaskGP(
            self.train_X_norm,
            self.train_Y_norm,
            mean_module = copy.deepcopy(self.mean_module),
            covar_module = copy.deepcopy(self.covar_module)
        )
        # We fix the likelihood variance and outputscale.
        # We only learn lengthscale for stability
        hypers = {
            'likelihood.noise_covar.noise': torch.tensor(1e-4),
            'covar_module.outputscale': torch.tensor(1),
        }
        gp.initialize(**hypers)
        gp.likelihood.raw_noise.requires_grad = False
        gp.covar_module.raw_outputscale.requires_grad = False
        return gp
    
    def conditioning(self, train_X, train_Y):
        self.train_X_norm = self.domain.transform_X(train_X)
        self.train_Y_norm = self.domain.transform_Y(train_Y, update=True)
        self.gp = self.set_model()
        self.n_data, self.n_dims = train_X.shape
        
    def predictive_mean_and_stddev(self, X, transform=False):
        X = cleansing_x(X)
        if transform:
            X = self.domain.transform_X(X)
        pred = self.gp.likelihood(self.gp(X))
        return pred.mean, pred.stddev
        
    def train(self, train_X, train_Y):
        self.conditioning(train_X, train_Y)
        # Type-II MLE
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp.likelihood, self.gp)
        botorch.fit.fit_gpytorch_mll(mll)
        self.gp.eval()

    def get_cov_cache(self):
        """
        Kinv = K(Xobs, Xobs)^(-1)
        S @ S.T = Kinv

        Input:
            - model: gpytorch.models, function of GP model, typically self.wsabi.model in _basq.py

        Output:
            - woodbury_inv: torch.tensor, the inverse of Gram matrix K(Xobs, Xobs)^(-1)
            - Xobs: torch.tensor, the observed inputs X
            - lik_var: torch.tensor, the GP likelihood noise variance
        """
        Xobs = self.gp.train_inputs[0].clone()
        Yobs = self.gp.train_targets.clone().view(-1).unsqueeze(-1).numpy()
        KXX = self.gp.covar_module.forward(Xobs, Xobs).detach().numpy()
        try:
            S = self.gp.prediction_strategy.covar_cache
        except:
            self.gp.eval()
            mean = Xobs[0].unsqueeze(0)
            self.gp(mean)
            S = self.gp.prediction_strategy.covar_cache
        Kinv = S @ S.T
        return Xobs.numpy(), Yobs, Kinv.detach().numpy(), KXX
