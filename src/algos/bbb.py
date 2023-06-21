import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from .algo import BayesianOptimizer
from .util import GaussianParameter
from .opt_util import apply_lr

class GaussianPrior:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma
        self.dist = torch.distributions.Normal(mu, sigma)

    def log_prob(self, x):
        return self.dist.log_prob(x)

    def kl_divergence(self, mu2, sigma2):
        #kl = 0.5 * (2 * torch.log(sigma2 / self.sigma) - 1 + (self.sigma / sigma2).pow(2) + ((mu2 - self.mu) / sigma2).pow(2))
        kl = 0.5 * (2 * torch.log(self.sigma / sigma2) - 1 + (sigma2 / self.sigma).pow(2) + ((self.mu - mu2) / self.sigma).pow(2))
        return kl.sum()

class MixturePrior:
    def __init__(self, pi, sigma1, sigma2, validate_args=None):
        self.pi = torch.tensor(pi)
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.dist1 = torch.distributions.Normal(0, sigma1, validate_args)
        self.dist2 = torch.distributions.Normal(0, sigma2, validate_args)

    def log_prob(self, value):
        prob1 = torch.log(self.pi) + torch.clamp(self.dist1.log_prob(value), -23, 0)
        prob2 = torch.log(1 - self.pi) + torch.clamp(self.dist2.log_prob(value), -23, 0)
        return torch.logaddexp(prob1, prob2)

    def kl_divergence(self, mu2, sigma2):
        return -self.log_prob(mu2).sum()

def collect_kl(model) -> torch.tensor:
    return sum(getattr(layer, "kl", 0) + collect_kl(layer) for layer in model.children())


class BBBOptimizer(BayesianOptimizer):
    '''
        Bayes By Backprop. You need to use Bayesian layers for the layers that should be treated as Bayesian.
    '''
    def __init__(self, params, base_optimizer, prior, dataset_size, mc_samples=1, kl_rescaling=1, components=1, l2_scale=0):
        defaults = {
            "prior": prior,
            "l2_scale": l2_scale
        }
        super().__init__(params, defaults)
        self.state["__base_optimizer"] = base_optimizer
        self.mc_samples = mc_samples
        self.kl_rescaling = kl_rescaling
        self.components = components
        self.dataset_size = dataset_size

    def step(self, forward_closure, backward_closure, grad_scaler=None):
        self.state["__base_optimizer"].zero_grad()
        
        total_data_loss = None
        for _ in range(self.mc_samples):
            if total_data_loss is None:
                total_data_loss = forward_closure()
            else:
                total_data_loss += forward_closure()

        # Collect KL loss & reg only once
        total_kl_loss = torch.tensor(0.0, device=self._params_device())
        for group in self.param_groups:
            for param in group["params"]:
                if hasattr(param, "get_parameter_kl"):
                    total_kl_loss += param.get_parameter_kl(group["prior"])
                elif not getattr(param, "_is_gaussian_mean", False) and not getattr(param, "_is_gaussian_rho", False):
                    total_kl_loss += group["l2_scale"] / 2 * param.pow(2).sum()

        pi = self.kl_rescaling / self.dataset_size
        # don't divide the kl loss by the mc sample count as we have only been collection it once
        loss = pi * total_kl_loss + total_data_loss / (self.mc_samples * self.components)
        if not loss.isnan().any():
            backward_closure(loss)

            if grad_scaler is not None:
                grad_scaler.step(self.state["__base_optimizer"])
            else:
                self.state["__base_optimizer"].step()

        return loss


    def sample_parameters(self):
        '''
            The parameters sample themself
        '''
        pass

    def get_base_optimizer(self):
        return self.state["__base_optimizer"]
