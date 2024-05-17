import random
import torch
from abc import ABC, abstractmethod


class Transform:
    def verticalise(self, Y):
        return Y.view(-1).unsqueeze(-1)
    
    def normalise(self, X):
        return (X - self.bounds[0]) / (self.bounds[1] - self.bounds[0])
    
    def standardise(self, Y, scaler=None):
        if scaler==None:
            scaler = [Y.mean(), Y.std()]
        if scaler[1] == 0:
            Y = (Y - scaler[0])
        else:
            Y = (Y - scaler[0]) / scaler[1]
        return Y
    
    def transform_X(self, X):
        if self.is_X_transformed:
            return self.normalise(X)
        else:
            return X
    
    def transform_Y(self, Y, update=False):
        if update:
            self.scaler = [Y.mean(), Y.std()]
        if self.is_Y_transformed:
            return self.verticalise(
                self.standardise(Y, scaler=self.scaler)
            )
        else:
            return self.verticalise(Y)
    
    def untransform_X(self, X):
        if self.is_X_transformed:
            return X * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
        else:
            return X
    
    def untransform_Y(self, Y):
        if self.is_Y_transformed:
            return Y * self.scaler[1] + self.scaler[0]
        else:
            return Y

class BaseDomain(ABC, Transform):
    @abstractmethod
    def sample(self, X):
        r"""Sampling from the prior"""
        pass
    
    @abstractmethod
    def pdf(self, X):
        r"""Return the probability density function of the prior"""
        pass
    
    def rand(self, n_dims, n_samples, qmc=True):
        if qmc:
            random_samples = torch.quasirandom.SobolEngine(n_dims, scramble=True).draw(n_samples)
        else:
            random_samples = torch.rand(n_samples, n_dims)
        return random_samples.to(self.dtype)

class FiniteDomain(BaseDomain):
    def __init__(self, dataset, is_X_transformed=True, is_Y_transformed=True) -> None:
        super().__init__()
        self.dataset = dataset
        self.dtype = dataset.dtype
        self.n_dims = dataset.shape[1]
        self.is_X_transformed = is_X_transformed
        self.is_Y_transformed = is_Y_transformed
        self.bounds = torch.cat([torch.min(self.dataset, axis=0).values.unsqueeze(0), torch.max(self.dataset, axis=0).values.unsqueeze(0)])
        if is_X_transformed:
            self.normed_bounds = torch.vstack([torch.zeros(self.n_dims), torch.ones(self.n_dims)]).to(self.dtype)
        else:
            self.normed_bounds = self.bounds
        
        self.normed_dataset = self.transform_X(self.dataset)

    def sample(self, n_samples):
        return self.dataset[random.sample(range(self.dataset.shape[0]), n_samples),:]

    def pdf(self, X):
        raise ValueError()

class UniformDomain(BaseDomain):
    def __init__(self, bounds, is_X_transformed=True, is_Y_transformed=True):
        """
        Uniform domain class
        
        Args:
        - bounds: torch.tensor, the lower and upper bounds for each dimension
        """
        self.dtype = bounds.dtype
        self.bounds = bounds
        self.n_dims = bounds.shape[1]
        self.is_X_transformed = is_X_transformed
        self.is_Y_transformed = is_Y_transformed
        if is_X_transformed:
            self.normed_bounds = torch.vstack([torch.zeros(self.n_dims), torch.ones(self.n_dims)]).to(self.dtype)
        else:
            self.normed_bounds = bounds
        
    def sample(self, n_samples, qmc=True):
        """
        Sampling from Uniform domain
        
        Args:
        - n_samples: int, the number of initial samples
        - qmc: bool, sampling from Sobol sequence if True, otherwise simply Monte Carlo sampling.
        
        Return:
        - samples: torch.tensor, the samples from uniform prior
        """
        random_samples = self.rand(self.n_dims, n_samples, qmc=qmc)
        samples = self.bounds[0].unsqueeze(0) + (
            self.bounds[1] - self.bounds[0]
        ).unsqueeze(0) * random_samples
        return samples
    
    def pdf(self, samples):
        """
        The probability density function (PDF) over samples
        
        Args:
        - samples: torch.tensor, the input where to compute PDF
        
        Return:
        - pdfs: torch.tensor, the PDF over samples
        """
        _pdf = torch.ones(len(samples)).to(self.dtype) * (1/(self.bounds[1] - self.bounds[0])).prod()
        _ood = torch.logical_or(
            (samples >= self.bounds[1]).any(axis=1), 
            (samples <= self.bounds[0]).any(axis=1),
        ).logical_not().to(self.dtype)
        return _pdf * _ood
