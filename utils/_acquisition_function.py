import torch
import casadi
import gpytorch
import botorch
from botorch.acquisition import AnalyticAcquisitionFunction
from botorch.utils import t_batch_mode_transform
from ._information_gain import InformationGain
from ._domain import FiniteDomain


class NormalLCB(AnalyticAcquisitionFunction):
    def __init__(self, model, beta, is_LCB=True, minimise=True):
        AnalyticAcquisitionFunction.__init__(
            self, 
            model=model,
            posterior_transform=None,
        )
        self.register_buffer("beta", beta)
        self.sign_LCB = (-1 if is_LCB else 1)
        self.sign_min = (-1 if minimise else 1)
        
    @botorch.utils.t_batch_mode_transform(expected_q=1)
    def forward(self, X):
        pred_mean, pred_std = self.model.predictive_mean_and_stddev(X)
        return self.sign_min * (pred_mean + self.sign_LCB * self.beta * pred_std)
    

class BetaCalculator(InformationGain):
    def __init__(self, model, n_iterations, tol=1e-2, gamma=20, R=None, method="exact"):
        super().__init__(model.gp, n_iterations, model.dtype, gamma=gamma)
        self.update_model(model)
        self.method = method
        self.tol = torch.tensor(tol, dtype=model.dtype)
        self.B = (model.domain.normed_bounds[1] - model.domain.normed_bounds[0]).prod()
        if not R is None:
            self.R = torch.tensor(R, dtype=model.dtype)
        
    def update_model(self, model):
        self.model = model
        self.dtype = model.dtype
        
    def compute_beta(self, t, gp):
        if self.method == "bayesian":
            d = gp.train_inputs[0].shape[1]
            beta = (2 * d * torch.log(t**2 / self.tol)).sqrt()
        else:
            if hasattr(self, "R"):
                R = self.R
            else:
                R = gp.likelihood.noise.detach()
            
            try:
                information_gain = self.compute_information_gain(t, gp, method=self.method)
                beta = self.B + R * (
                    2 * (information_gain + 1 + (2 / self.tol).log())
                ).sqrt()
                
                if torch.isnan(beta) or torch.isinf(beta) or (beta < 0):
                    print("Beta computation fails, we set beta as 1.")
                    print(beta)
                    beta = torch.tensor(1, dtype=self.dtype)
            except:
                print("Beta computation fails, we set beta as 1.")
                beta = torch.tensor(1, dtype=self.dtype)
        return beta


class SymbolicLowerConfidenceBound:
    def symbolic_KxX(self, var_x):
        Xobs, Yobs, Kinv, KXX = self.model.get_cov_cache()
        n_data, n_dims = Xobs.shape

        lengthscales = self.model.gp.covar_module.base_kernel.lengthscale.detach().squeeze().numpy()
        outputscale = self.model.gp.covar_module.outputscale.detach().numpy()

        diff_x_scaled = (var_x - Xobs.T) / lengthscales
        diff_x_scaled_inner_prod = diff_x_scaled[:, 0].T @ diff_x_scaled[:, 0]

        for i in range(1, n_data):
            diff_x_scaled_inner_prod = casadi.vertcat(
                diff_x_scaled_inner_prod,
                diff_x_scaled[:, i].T @ diff_x_scaled[:, i]
            )

        if type(self.model.gp.covar_module.base_kernel) == gpytorch.kernels.rbf_kernel.RBFKernel:
            KxX = outputscale * casadi.exp(- 0.5 * diff_x_scaled_inner_prod)
        return KxX, Yobs, Kinv, n_data, n_dims

    def symbolic_LCB(self, beta, var_x):
        KxX, Yobs, Kinv, n_data, n_dims = self.symbolic_KxX(var_x)
        posterior_mean = KxX.T @ Kinv @ Yobs
        x0 = torch.zeros(n_dims).unsqueeze(0)
        posterior_var = self.model.gp.covar_module(x0, x0).numpy().item() - KxX.T @ Kinv @ KxX
        if type(beta) == torch.Tensor:
            beta = beta.item()
        lcb = posterior_mean - beta * casadi.sqrt(posterior_var)
        return lcb


class AcquisitionFunction(BetaCalculator, SymbolicLowerConfidenceBound):
    def __init__(
        self,
        t,
        model,
        n_iterations,
        is_LCB=True,
        minimise=True,
        beta=None,
        tol=1e-2,
        gamma=20,
        R=None,
        method="exact",
        bounds=None,
    ):
        BetaCalculator.__init__(self, model, n_iterations, tol=tol, gamma=gamma, R=R, method=method)
        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = model.domain.normed_bounds
        self.is_LCB = is_LCB
        self.minimise = minimise
        if beta is not None:
            self.beta = beta
        else:
            self.beta = self.compute_beta(t, self.model.gp)
    
    def optimize(self):
        acqf = NormalLCB(self.model, self.beta, is_LCB=self.is_LCB, minimise=self.minimise)
        if type(self.model.domain) == FiniteDomain:
            return self.model.domain.dataset[torch.argmax(acqf(self.model.domain.normed_dataset.unsqueeze(1))), :].unsqueeze(0)
        else:
            X_next_norm, LCB_value_negative = botorch.optim.optimize_acqf(
                acqf,
                bounds=self.bounds,
                q=1,
                num_restarts=10,
                raw_samples=512,
            )
            X_next = self.model.domain.untransform_X(X_next_norm)
            LCB_value = acqf.sign_min * LCB_value_negative
            return X_next, LCB_value
     
    def __call__(self, X):
        acqf = NormalLCB(self.model, self.beta, is_LCB=self.is_LCB, minimise=self.minimise)
        return acqf(X)
    