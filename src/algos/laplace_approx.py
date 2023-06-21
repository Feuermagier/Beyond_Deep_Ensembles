import torch
import torch.nn as nn
from laplace import Laplace
from laplace.curvature import BackPackGGN

from src.algos.pp import MAP

class LaplaceApprox(nn.Module):
    def __init__(self, map, regression=False, weights='last_layer', hessian='diag', out_activation=None, prior_precision=1.0, temperature=1.0):
        super().__init__()
        self.map = map
        self.laplace = Laplace(self.map,
                               'regression' if regression else 'classification', 
                               subset_of_weights=weights, 
                               hessian_structure=hessian, 
                               prior_precision=prior_precision, 
                               temperature=temperature)
                               #backend=BackPackGGN)
        self.out_activation = out_activation
        self.supports_multisample = True

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        return {
            "map": self.map.state_dict(destination, prefix, keep_vars),
            "laplace": self.laplace
        }

    def load_state_dict(self, dict):
        self.map.load_state_dict(dict["map"])
        self.laplace = dict["laplace"]

    def fit(self, loader):
        self.laplace.fit(loader)

    def optimize_prior_prec(self):
        self.laplace.optimize_prior_precision(method="marglik")

    def prior_precision(self):
        return self.laplace.prior_precision

    def forward(self, input, n_samples=1):
        #output = self.laplace(input, n_samples=n_samples, pred_type="glm", link_approx="mc")
        output = self.laplace._nn_predictive_samples(input, n_samples=n_samples)
        if self.out_activation is not None:
            output = self.out_activation(output)
        if n_samples == 1:
            output = output.squeeze(0)
        return output
