import copy
import scipy
import torch
import numpy as np

class InformationGain:
    def __init__(self, gp, n_iterations, dtype, gamma=20):
        self.T = n_iterations
        self.dtype = dtype
        self.gamma = gamma
        self.cache = {}
        self.update_gp(gp)
        
    def update_gp(self, gp):
        self.Xobs = gp.train_inputs[0]
        self.KXX = gp.covar_module.forward(self.Xobs, self.Xobs).to_dense().detach()
        self.L, _ = torch.linalg.cholesky_ex(self.KXX, upper=False)
        
    def compute_S(self, p, A, gamma=0):
        k, d = A.shape
        ret = torch.zeros((d, d), dtype=self.dtype)
        for i in range(k):
            ret += p[i] * A[[i], :].T @ A[[i], :]
        return ret + gamma * torch.eye(d, dtype=self.dtype)
    
    def compute_G(self, x, A, gamma):
        k, d = A.shape
        return (
            self.compute_S(x, A) + gamma * torch.eye(d, dtype=self.dtype)
        )
    
    def compute_phi(self, c, t, x, A, gamma):
        g = self.compute_G(x, A, gamma)
        _, logdet = torch.linalg.slogdet(g)
        return (
            t * ((c * x).sum() - logdet)
            - x.log().sum()
            - (1 - x.sum()).log()
        )
    
    def newton_direction(self, c, t, x, A, gamma):
        n, d = A.shape
        V = self.compute_S(x, A) + gamma * torch.eye(d, dtype=self.dtype)
        V_inv = V.inverse()
        vs = torch.zeros(n, dtype=self.dtype)
        for i in range(n):
            a = A[i, :]
            vs[i] = V_inv @ a @ a
        d1 = (
            t * (c - vs)
            - 1 / x
            + 1 / (1 - x.sum())
        )
        d2 = (
            t * (A @ V_inv @ A.T).pow(2)
            + 1 / ((1 - x.sum()) ** 2)
            + (1 / x.pow(2)).diag()
        )
        direction = -d2.inverse() @ d1
        v = direction.reshape((n, 1))
        l = (v.T @ d2 @ v).sqrt().item()
        return direction, l
    
    def generalized_eigenvalues(self, x, direction, A, gamma):
        k, d = A.shape
        left = self.compute_S(direction, A)
        right = self.compute_S(x, A) + gamma * torch.eye(d, dtype=self.dtype)
        eigen1 = torch.from_numpy(
            np.real(scipy.linalg.eigvals(left.numpy(), right.numpy()))
        ).to(self.dtype)
        eigen2 = torch.cat([
            direction / x,
            (-direction.sum() / (1 - x.sum())).unsqueeze(0),
        ])
        return (eigen1, eigen2)
    
    def line_search(self, c, t, x, direction, A, gamma):
        eigen1, eigen2 = self.generalized_eigenvalues(x, direction, A, gamma)
        h = 0
        for _ in range(30):
            d1 = (
                t * (c * direction).sum()
                - (t * eigen1 / (1 + h * eigen1)).sum()
                - (eigen2 / (1 + h * eigen2)).sum()
            )
            d2 = (
                (t / ((1 / eigen1 + h) ** 2)).sum()
                + (1 / ((1 / eigen2 + h) ** 2)).sum()
            )
            increment = -d1 / d2
            if (1 + (h + increment) * eigen1 <= 1e-8).any() or (1 + (h + increment) * eigen2 <= 1e-8).any():
                min_eigenvalue = min(eigen1.min(), eigen2.min())
                h = (h - 1 / min_eigenvalue) / 2
            else:
                h = h - d1 / d2
        return h
    
    def newton_optimize(self, c, t, A, gamma, x=None):
        n, d = A.shape
        if x is None:
            x = torch.ones(n, dtype=self.dtype) / (2 * n)
        l = 1
        J = torch.inf
        counter = 0

        save_x = copy.deepcopy(x)
        while l > 1e-6 / gamma * t:
            direction, l = self.newton_direction(c, t, x, A, gamma)
            h = 1
            if l > 0.5:
                h = self.line_search(c, t, x, direction, A, gamma)
            x = x + h * direction
            new_J = self.compute_phi(c, t, x, A, gamma)
            J = new_J
            counter += 1
            if counter > 10:
                break
        return x
    
    def central_path(self, c, A, gamma):
        n, d = A.shape
        x = torch.ones(n, dtype=self.dtype) / (2 * n)
        t = 1
        while t < 10 or (t < 1e7 and x.sum() < 0.8):
            x = self.newton_optimize(c, t, A, gamma, x=x)
            t = t * 1.1
        return x
    
    def compute_exact_information_gain(self, t):
        n, d = self.L.shape
        delta = torch.zeros(n, dtype=self.dtype)
        c = delta/2
        p = self.central_path(c, self.L, self.gamma / t)
        p /= p.sum()
        
        s = self.compute_S(p, self.L) * t / self.gamma + torch.eye(d, dtype=self.dtype)
        _, logdet = torch.linalg.slogdet(s)
        information_gain = logdet / 2
        return information_gain
    
    def compute_with_cache_check(self, t):
        if t in self.cache:
            return self.cache[t]
        else:
            return self.compute_exact_information_gain(t)
        
    def approximate_information_gain(self, t):
        if t in self.cache:
            return self.cache[t]
        else:
            t_left = 2 ** int(torch.tensor(t).log2().item())
            dimension_left = self.compute_with_cache_check(t_left)
            if t_left == t:
                return dimension_left
            t_right = min(self.T, t_left * 2)
            dimension_right = self.compute_with_cache_check(t_right)

            # interpolate
            p = (t - t_left) / (t_right - t_left)
            return dimension_left * (1 - p) + dimension_right * p
        
    def compute_information_gain(self, t, gp, method="exact"):
        self.update_gp(gp)
        if method == "exact":
            information_gain = self.compute_with_cache_check(t)
        elif method == "approx":
            information_gain = self.approximate_information_gain(t)
        else:
            raise NotImplementedError("The method should be either 'exact' or 'approx'.")
        return information_gain
