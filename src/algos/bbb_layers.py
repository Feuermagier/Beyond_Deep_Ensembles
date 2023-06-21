import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import math
import time
from .util import GaussianMixture, GaussianParameter, normal_like

class BBBLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, weight_prior, bias_prior, **kwargs):
        super().__init__()
        self.is_bayesian = True
        self.sampling = kwargs.get("sampling", "activations")
        self.mc_sample = kwargs.get("mc_sample", 1)
        self.freeze_on_eval = kwargs.get("freeze_on_eval", True)
        self.kl_on_eval = kwargs.get("kl_on_eval", False)
        self.in_features, self.out_features = in_features, out_features
        self.weight_prior, self.bias_prior = weight_prior, bias_prior
        self.rho_init = kwargs.get("rho_init", -3)
        self.use_bias = kwargs.get("bias", True)

        self.weight = GaussianParameter((self.out_features, self.in_features))
        if self.use_bias:
            self.bias = GaussianParameter((out_features,))

        self.reset_parameters()

        self.kl = 0

    def reset_parameters(self):
        self.weight.blundell_init()
        if self.use_bias:
            self.bias.blundell_init()

    def sample_parameters(self, mu, sigma):
        epsilon = torch.empty(sigma.shape, device=mu.device).normal_(0, 1)
        return mu + sigma * epsilon

    def forward(self, input: torch.Tensor):
        self.kl = 0

        if self.sampling == "parameters":
            output = torch.zeros((input.shape[0], self.out_features))

            for _ in range(self.mc_sample):
                weight = self.weight.sample()
                if self.use_bias:
                    bias = self.bias.sample()
                    output += F.linear(input, weight, bias)
                else:
                    output += F.linear(input, weight)

                if self.training or self.kl_on_eval:
                    log_prior = self.weight_prior.log_prob(weight).sum() + (self.bias_prior.log_prob(bias).sum() if self.use_bias else 0)
                    log_posterior = log_prob(self.weight.mean, self.weight.std, weight).sum() + (log_prob(self.bias.mean, self.bias.std, bias).sum() if self.use_bias else 0)
                    self.kl += log_posterior - log_prior

            self.kl /= self.mc_sample
            return output / self.mc_sample
        elif self.sampling == "activations":
            if input.is_cuda:
                # TODO handle missing bias
                # This is slightly faster than the sequential variant on GPUs
                batch_in = torch.stack((input, (input**2).clamp(min=1e-4)))
                batch_mat = torch.stack((self.weight.mean.transpose(0, 1), (self.weight.std.transpose(0, 1)**2).clamp(min=1e-4)))
                batch_add = torch.stack((self.bias.mean.expand((input.shape[0], self.out_features)), (self.bias.std**2).clamp(min=1e-4).expand((input.shape[0], self.out_features))))
                batch_out = torch.baddbmm(batch_add, batch_in, batch_mat)
                activation_mean = batch_out[0]
                activation_std = torch.sqrt(batch_out[1])
            else:
                activation_mean = F.linear(input, self.weight.mean, self.bias.mean)
                activation_var = F.linear((input**2).clamp(min=1e-4), (self.weight.std**2).clamp(min=1e-4), (self.bias.std**2).clamp(min=1e-4))
                activation_std = torch.sqrt(activation_var)

            if not self.training and self.freeze_on_eval:
                epsilon = torch.empty(activation_mean.shape[1:], device=activation_mean.device).normal_(0, 1).unsqueeze(0).expand((activation_mean.shape))
            else:
                epsilon = normal_like(activation_mean)
            output = activation_mean + activation_std * epsilon
            
            if self.training or self.kl_on_eval:
                #log_prior = self.weight_prior.log_prob(self.weight_mu).sum() + self.bias_prior.log_prob(self.bias_mu).sum() 
                #self.kl = -log_prior
                weight_kl = self.weight_prior.kl_divergence(self.weight.mean, self.weight.std)
                bias_kl = self.bias_prior.kl_divergence(self.bias.mean, self.bias.std)
                self.kl = weight_kl + bias_kl
            return output / self.mc_sample
        else:
            raise ValueError("Invalid value of sampling")

    def means(self):
        return torch.cat([self.weight.mean.flatten(), self.bias.mean.flatten()])

    def mean_grads(self):
        return torch.cat([self.weight.mean.grad.flatten(),self.bias.mean.grad.flatten()])

    def sigmas(self):
        return torch.cat([to_sigma(self.weight.rho).flatten(), to_sigma(self.bias.rho).flatten()])

    def rho_grads(self):
        return torch.cat([self.weight.rho.grad.flatten(),self.bias.rho.grad.flatten()])


class BBBConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, weight_prior, bias_prior, **kwargs):
        super().__init__()
        self.is_bayesian = True
        self.sampling = kwargs.get("sampling", "activations")
        self.stride = kwargs.get("stride", 1)
        self.freeze_on_eval = kwargs.get("freeze_on_eval", True)
        self.kl_on_eval = kwargs.get("kl_on_eval", False)
        self.use_bias = kwargs.get("bias", True)
        self.padding = kwargs.get("padding", 0)
        self.out_channels, self.in_channels = out_channels, in_channels
        self.kernel_size = kernel_size

        self.weight_prior, self.bias_prior = weight_prior, bias_prior

        # Weights
        #self.weight_mu = nn.Parameter(torch.empty((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)).normal_(0, 0.1))
        #self.weight_rho = nn.Parameter(torch.empty((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)).uniform_(-3, -3))
        self.weight = GaussianParameter((self.out_channels, self.in_channels, self.kernel_size, self.kernel_size))

        # Biases
        if self.use_bias:
            #self.bias_mu = nn.Parameter(torch.empty(self.out_channels).normal_(0, 0.1))
            #self.bias_rho = nn.Parameter(torch.empty(self.out_channels).uniform_(-3, -3))
            self.bias = GaussianParameter((self.out_channels,))

        self.reset_parameters()
        self.kl = 0

    def reset_parameters(self):
        self.weight.blundell_init()
        if self.use_bias:
            self.bias.blundell_init()

    def forward(self, input: torch.Tensor):
        self.kl = 0

        if self.sampling == "parameters":
            raise NotImplementedError()
        elif self.sampling == "activations":

            activation_mean = F.conv2d(input, self.weight.mean, self.bias.mean if self.use_bias else None, stride=self.stride, padding=self.padding)
            activation_var = F.conv2d((input**2).clamp(min=1e-4), (self.weight.std**2).clamp(min=1e-4), self.bias.std**2 if self.use_bias else None, stride=self.stride, padding=self.padding)
            activation_std = torch.sqrt(activation_var)

            if not self.training and self.freeze_on_eval:
                epsilon = torch.empty(activation_mean.shape[1:], device=activation_mean.device).normal_(0, 1).unsqueeze(0).expand((activation_mean.shape))
            else:
                epsilon = normal_like(activation_mean)
            output = activation_mean + activation_std * epsilon

            if self.training or self.kl_on_eval:
                self.kl = self.weight_prior.kl_divergence(self.weight.mean, self.weight.std)
                if self.use_bias:
                    self.kl += self.bias_prior.kl_divergence(self.weight.mean, self.weight.std)
                    
            return output
        else:
            raise ValueError("Invalid value of sampling")

class LowRankBBBLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, gamma, K, **kwargs):
        super().__init__()
        self.is_bayesian = True
        self.freeze_on_eval = kwargs.get("freeze_on_eval", True)
        self.kl_on_eval = kwargs.get("kl_on_eval", False)
        self.rho_init = kwargs.get("rho_init", -3)
        self.offdiag_init = kwargs.get("offdiag_init", 0)
        self.in_features, self.out_features = in_features, out_features
        self.gamma = gamma
        self.K = K
        self.alpha = 1 / math.sqrt(self.K) if K != 0 else 1

        self.params = (self.in_features + 1) * self.out_features
        self.param_mean = nn.Parameter(torch.empty(self.params))
        self.param_diag_rho = nn.Parameter(torch.empty(self.params))
        self.param_lr_vars = nn.Parameter(torch.empty((self.K, self.params)))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.normal_(self.param_mean, 0, 0.1)
        torch.nn.init.constant_(self.param_diag_rho, self.rho_init)
        torch.nn.init.constant_(self.param_lr_vars, self.offdiag_init)

    def forward(self, input: torch.Tensor):
        batch_size = input.shape[0]

        # Add 1 as an additional value for each input and expand the tensor to K + 2
        input_pad = torch.cat((input, torch.ones(batch_size, 1)), dim=-1)
        input_ext = torch.cat((input_pad.repeat(self.K + 1, 1, 1), input_pad.unsqueeze(0)**2))

        diag_vars = F.softplus(self.param_diag_rho).reshape(1, self.in_features + 1, self.out_features)**2
        weight_means = self.param_mean.reshape(1, self.in_features + 1, self.out_features)
        lr_vars = self.param_lr_vars.reshape(self.K, self.in_features + 1, self.out_features)
        matrices = torch.cat((weight_means, lr_vars, diag_vars), dim=0)

        # Batch matrix multiplication - handle diag separately to avoid repeating the input?
        results = torch.bmm(input_ext, matrices)

        # Undo batching
        activation_mean = results[0]
        activation_lr_std = results[1:-1]
        activation_diag_std = torch.sqrt(results[-1])

        # Perturb
        if not self.training and self.freeze_on_eval:
            epsilon_diag = torch.empty((1, self.out_features), device=activation_mean.device).normal_(0, 1).expand((activation_mean.shape))
            epsilon_lr = torch.empty((self.K, 1, 1), device=activation_mean.device).normal_(0, 1).expand((self.K, batch_size, self.out_features))
        else:
            epsilon_diag = torch.empty((batch_size, self.out_features), device=activation_mean.device).normal_(0, 1)
            epsilon_lr = torch.empty((self.K, batch_size, 1), device=activation_mean.device).normal_(0, 1).expand((self.K, batch_size, self.out_features))

        activation_diag_std_p = activation_diag_std * epsilon_diag
        activation_lr_std_p = activation_lr_std * epsilon_lr

        output = activation_mean + activation_diag_std_p + self.alpha * activation_lr_std_p.sum(dim=0)

        return output

        # Mean
        weight_mean, bias_mean = self.convert_params(self.param_mean)
        activation_mu = F.linear(input, weight_mean, bias_mean)

        # Diagonal
        param_diag_var = to_sigma(self.param_diag_rho)
        weight_diag_var, bias_diag_var = self.convert_params(param_diag_var)
        if not self.training and self.freeze_on_eval:
            epsilon = torch.empty(activation_mu.shape[1:]).normal_(0, 1).unsqueeze(0).expand((activation_mu.shape)).to(activation_mu.device)
        else:
            epsilon = torch.empty(activation_mu.shape).normal_(0, 1).to(activation_mu.device)
        activation_diag_var = torch.sqrt(F.linear(input**2, weight_diag_var**2, bias_diag_var**2)) * epsilon

        # Low Rank
        if not self.training and self.freeze_on_eval:
            epsilon = torch.normal(torch.zeros(input.shape[0]), 1).unsqueeze(dim=-1).expand((input.shape[0], self.out_features))
        else:
            epsilon = None
        activation_lr_var = torch.stack([self.low_rank_forward(k, input, epsilon=epsilon) for k in range(self.K)]).sum(dim=0) if self.K != 0 else torch.tensor(0)

        # Output
        output = activation_mu + activation_diag_var + self.alpha * activation_lr_var
        
         
        return output

    @property
    def kl(self):
        param_diag_var = to_sigma(self.param_diag_rho)
        capacitance = torch.eye(self.K) + self.param_lr_vars @ torch.diag(1 / param_diag_var) @ self.param_lr_vars.T
        return 0.5 * (
            (param_diag_var / self.gamma - torch.log(param_diag_var)).sum() \
            + self.alpha / self.gamma * (torch.linalg.vector_norm(self.param_lr_vars, dim=1)**2).sum() \
            - torch.log(torch.linalg.det(capacitance)) \
            + 1 / self.gamma * torch.linalg.vector_norm(self.param_mean)**2 \
            + self.params * (math.log(self.gamma) - 1)
        )

def to_sigma(rho) -> torch.tensor:
    return F.softplus(rho)

def log_prob(mu, sigma, value):
    #return torch.clamp(-((value - mu)**2) / (2 * sigma**2) - sigma.log() - math.log(math.sqrt(2 * math.pi)), -23, 0)
    return torch.clamp(torch.distributions.Normal(mu, sigma, False).log_prob(value), -23, 0)

def softplus_inverse(x):
    return torch.expm1(x).log()

# Closed form KL divergence for gaussians
# See https://github.com/kumar-shridhar/PyTorch-BayesianCNN/blob/master/metrics.py
def gauss_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl

def make_module_bbb(module, prior):
    '''
        Converts all nn.Linear and nn.Conv2d layers of the given model to their BBB counterparts.
        The mean values are initialized from the original weights of the layers.
        Note that the kernel of convolutional layers is assumed to be of quadratic shape.
    '''
    for name, m in list(module._modules.items()):
        if m._modules:
            make_module_bbb(m, prior)
        elif "Conv2d" in m.__class__.__name__:
            bbb_layer = BBBConv2d(in_channels=m.in_channels, out_channels=m.out_channels, kernel_size=m.kernel_size[0], weight_prior=prior, bias_prior=prior, stride=m.stride, bias=m.bias is not None, padding=m.padding)
            bbb_layer.weight.overwrite_mean(m.weight.data.clone())
            if m.bias is not None:
                bbb_layer.bias.overwrite_mean(m.bias.data.clone())
            setattr(module, name, bbb_layer)
        elif "Linear" in m.__class__.__name__:
            bbb_layer = BBBLinear(in_features=m.in_features, out_features=m.out_features, weight_prior=prior, bias_prior=prior, use_bias=m.bias is not None)
            bbb_layer.weight.overwrite_mean(m.weight.data.clone())
            if m.bias is not None:
                bbb_layer.bias.overwrite_mean(m.bias.data.clone())
            setattr(module, name, bbb_layer)
        else:
            pass
