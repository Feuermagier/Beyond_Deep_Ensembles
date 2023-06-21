import torch
import torch.nn as nn
from torch.nn.functional import softplus

# Filter Response Normalization (https://arxiv.org/abs/1911.09737)
# Code adapted from https://github.com/izmailovpavel/neurips_bdl_starter_kit/blob/main/pytorch_models.py

class FilterResponseNorm(nn.Module):
    def __init__(self, num_filters, eps=1e-6):
        super().__init__()
        self.eps = eps
        par_shape = (1, num_filters, 1, 1)  # [1,C,1,1]
        self.tau = torch.nn.Parameter(torch.zeros(par_shape))
        self.beta = torch.nn.Parameter(torch.zeros(par_shape))
        self.gamma = torch.nn.Parameter(torch.ones(par_shape))

    def forward(self, x):
        nu2 = torch.mean(torch.square(x), dim=[2, 3], keepdim=True)
        x = x * torch.rsqrt(nu2 + self.eps)
        y = self.gamma * x + self.beta
        z = torch.max(y, self.tau)
        return z

class VariationalFilterResponseNorm(nn.Module):
    def __init__(self, num_filters, prior, eps=1e-6, rho_init=-3):
        super().__init__()
        self.eps = eps
        self.prior = prior
        par_shape = (1, num_filters, 1, 1)  # [1,C,1,1]

        self.tau_mu = torch.nn.Parameter(torch.zeros(par_shape))
        self.tau_rho = torch.nn.Parameter(torch.zeros(par_shape))
        nn.init.constant_(self.tau_rho, rho_init)

        self.beta_mu = torch.nn.Parameter(torch.zeros(par_shape))
        self.beta_rho = torch.nn.Parameter(torch.zeros(par_shape))
        nn.init.constant_(self.beta_rho, rho_init)

        self.gamma_mu = torch.nn.Parameter(torch.ones(par_shape))
        self.gamma_rho = torch.nn.Parameter(torch.ones(par_shape))
        nn.init.constant_(self.gamma_rho, rho_init)

    def forward(self, x):
        gamma_sigma = softplus(self.gamma_rho)
        gamma = self.gamma_mu + gamma_sigma * torch.rand_like(self.gamma_rho)

        beta_sigma = softplus(self.beta_rho)
        beta = self.beta_mu + beta_sigma * torch.rand_like(self.beta_rho)

        tau_sigma = softplus(self.tau_rho)
        tau = self.tau_mu + tau_sigma * torch.rand_like(self.tau_rho)

        nu2 = torch.mean(torch.square(x), dim=[2, 3], keepdim=True)
        x = x * 1 / torch.sqrt(nu2 + self.eps)
        y = gamma * x + beta
        z = torch.max(y, tau)

        self.kl = self.prior.kl_divergence(self.gamma_mu, gamma_sigma) \
            + self.prior.kl_divergence(self.beta_mu, gamma_sigma) \
            + self.prior.kl_divergence(self.tau_mu, tau_sigma)

        return z
