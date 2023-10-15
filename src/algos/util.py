import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import itertools

def gauss_logprob(mean: torch.Tensor, variance: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return -((x - mean) ** 2) / (2 * variance) - torch.log(variance.sqrt()) - math.log(math.sqrt(2 * math.pi))

def sgd(lr, momentum=0, weight_decay=0, nesterov=False):
    return lambda parameters: torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)

def adam(lr, weight_decay=0):
    return lambda parameters: torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)

def nll_loss(output, target, eps: float = 1e-4):
    mean = output[...,0]
    var = output[...,1]**2
    var = var.clamp(min=eps)
    #return F.gaussian_nll_loss(mean, target, std**2, reduction)
    # Custom implementation without any() to support functorch
    loss = 0.5 * (torch.log(var) + (mean - target)**2 / var)
    return loss.mean()

def scheduler_factory(schedule):
    return lambda opt: torch.optim.lr_scheduler.LambdaLR(opt, schedule)

def step_scheduler(milestones, gamma):
    def schedule(epoch):
        lr = 1
        for milestone in milestones:
            if milestone <= epoch:
                lr *= gamma
            else:
                break
        return lr
    return schedule

def lr_scheduler(milestones, gamma):
    return lambda opt: torch.optim.lr_scheduler.MultiStepLR(opt, milestones, gamma)

def wilson_scheduler(pretrain_epochs, lr_init, swag_lr=None):
    def wilson_schedule(epoch):
        t = (epoch) / pretrain_epochs
        lr_ratio = swag_lr / lr_init if swag_lr is not None else 0.01
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return factor
    return wilson_schedule

# Weighted sum of two gaussian distributions
class GaussianMixture:
    def __init__(self, pi, sigma1, sigma2):
        self.log_pi, self.sigma1, self.sigma2 = torch.log(torch.tensor(pi)), sigma1, sigma2
        self.gaussian1 = torch.distributions.Normal(0, sigma1)
        self.gaussian2 = torch.distributions.Normal(0, sigma2)

    def log_prob(self, value):
        return torch.logaddexp(self.log_pi + self.gaussian1.log_prob(value), self.log_pi + self.gaussian2.log_prob(value))

class GaussLayer(nn.Module):
    def __init__(self, std_init: torch.Tensor, learn_var: bool = False):
        super().__init__()
        if learn_var:
            self.rho = nn.Parameter(torch.log(torch.exp(std_init) - 1))

            # Prevent rho from being optimized with e.g. VI. Used by algorithms that operate on all model parameters (e.g. VOGN, SVGD) rather than locally on individual layers (e.g. BBB)
            self.rho.use_mle_training = True 
        else:
            self.register_buffer("rho", torch.log(torch.exp(std_init) - 1))
        self.learn_var = learn_var

    def forward(self, input):
        out = torch.stack((input, self.std.expand(input.shape)), dim=-1)
        return out

    @property
    def std(self):
        return F.softplus(self.rho)

    @property
    def var(self):
        return self.std**2


def plot_losses(name, losses, ax, val_losses=[], text=True, y_name="Training Loss"):
    epochs = max([len(loss) for loss in losses])
    ax.set_xlabel("Epoch", fontsize=14)
    ax.set_xticks(np.arange(0, epochs + 1, epochs // 10 if epochs > 10 else 1))
    ax.set_ylabel(y_name, fontsize=14)
    if len(losses) > 1:
        for i, single_losses in enumerate(losses):
            ax.plot(np.arange(1, len(single_losses) + 1, 1), single_losses, label=f"{name} ({i})")
    else:
        ax.plot(np.arange(1, len(losses[0]) + 1, 1), losses[0], label=name)
    
    if len(val_losses) > 0:
        ax.plot(np.arange(1, len(val_losses) + 1, 1), val_losses, label=name + " (Validation)")

    if text:
        ax.legend(loc="upper right")
    return ax


class EarlyStopper:
    def __init__(self, evaluator, interval, delta, patience):
        self.evaluator = evaluator
        self.interval = interval
        self.delta = delta
        self.patience = patience

        self.losses = []
        self.best_loss = float("inf")
        self.epochs_since_best = 0

    def should_stop(self, model, epoch):
        if epoch % self.interval != 0:
            return False

        with torch.no_grad():
            loss = self.evaluator(model)
            self.losses.append(loss)

            if loss < self.best_loss - self.delta:
                self.best_loss = loss
                self.epochs_since_best = 0
            else:
                self.epochs_since_best += 1
            #print(f"val loss {loss}")
            #print(f"patience {self.epochs_since_best}")

            if self.epochs_since_best > self.patience:
                print(f"Stopping early")
                return True
            else:
                return False

class PrintLayer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, input):
        print(input.shape)
        return input

class GaussianParameter(nn.Module):
    '''
        Don't overwrite the rho *parameter*, or make sure to set the _is_gaussian_rho attribute on the parameter to True
    '''
    def __init__(self, size, device=None):
        super().__init__()
        self.overwrite_mean(torch.empty(size, device=device))
        self.rho = nn.Parameter(torch.empty(size, device=device))
        self.rho._is_gaussian_rho = True # Required e.g. by BBB

    def blundell_init(self, mean_std=0.1):
        torch.nn.init.normal_(self.mean, 0, mean_std)
        torch.nn.init.constant_(self.rho, -3)

    def sign_init(self):
        with torch.no_grad():
            self.mean.data = (torch.rand_like(self.mean) > 0.5).float() * 2 - 1
        torch.nn.init.constant_(self.rho, -3)

    def sample(self) -> torch.tensor:
        return self.mean + normal_like(self.std) * self.std

    def kl_divergence(self, prior):
        return prior.kl_divergence(self.mean, self.std)

    def overwrite_mean(self, mean):
        self.mean = nn.Parameter(mean)
        self.mean.get_parameter_kl = self.kl_divergence
        self.mean._is_gaussian_mean = True # Required e.g. by BBB

    @property
    def std(self) -> torch.tensor:
        return F.softplus(self.rho)

def normal_like(tensor) -> torch.tensor:
    return torch.empty_like(tensor).normal_(0, 1)

def non_mle_params(params):
    return filter(lambda p: getattr(p, "use_mle_training", False) is False, params)

def reset_model_params(model):
    '''
        Resets all parameters of the model (that can be reset, i.e. where the model implements a reset_parameters function)
    '''
    # From https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
    
    def weight_reset(m):
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    model.apply(weight_reset)

def patch_batchnorm(module, track_running_stats=True):
    '''
        Changes the properties of all BatchNormXd layers in the model. Useful for disabling BatchNorm's track_running_stats when evaluating Bayesian models.
    '''
    patched = 0
    for name, m in list(module._modules.items()):
        if m._modules:
            patched += patch_batchnorm(m, track_running_stats=track_running_stats)
        elif "BatchNorm" in m.__class__.__name__:
            patched += 1
            m.track_running_stats = track_running_stats
            if not track_running_stats:
                m.running_mean = None
                m.running_var = None
    return patched
