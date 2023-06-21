import torch
import torch.nn as nn
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
import torch.nn.functional as F
import math
import numpy as np
from .algo import BayesianOptimizer
from .opt_util import apply_lr

class SwagOptimizer(BayesianOptimizer):
    '''
        Stochastic Weight Averaging-Gaussian
    '''

    def __init__(self, params, base_optimizer, update_interval, start_epoch=0, deviation_samples=30):
        super().__init__(params, {})

        self.start_epoch = start_epoch
        self.update_interval = math.floor(update_interval)
        self.param_dist = None
        self.deviation_samples = deviation_samples

        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                state["original_param"] = param.data.detach().clone()

        self.state["__base_optimizer"] = base_optimizer
        self.state["__epoch"] = 0
        self.state["__steps_since_swag_start"] = 0
        self.state["__updates"] = 0
        self.state["__mean"] = parameters_to_vector(self._params()).cpu()
        self.state["__sq_weights"] = self.state["__mean"]**2
        self.state["__deviations"] = torch.zeros((self.state["__mean"].shape[0], self.deviation_samples))
        self.state["__params_dirty"] = False

    def step(self, forward_closure, backward_closure, grad_scaler=None):
        self._restore_original_params()
        self.state["__base_optimizer"].zero_grad()

        loss = forward_closure()
        backward_closure(loss)

        if grad_scaler is not None:
            grad_scaler.step(self.state["__base_optimizer"])
        else:
            self.state["__base_optimizer"].step()

        self._swag_update()

        return loss

    def sample_parameters(self):
        self._update_param_dist()
        self._save_original_params()
        self.state["__params_dirty"] = True
        new_params = self.param_dist.sample().to(self._params_device())
        vector_to_parameters(new_params, self._params())

    def complete_epoch(self):
        self.state["__epoch"] += 1

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)
    
    def get_base_optimizer(self):
        return self.state["__base_optimizer"]

        # Required as PyTorch's optimizer only casts per-param state in load_state_dict()
        # param_device = self._params_device()
        # if param_device != self.state["__mean"].device:
        #     self.state["__mean"] = self.state["__mean"].to(param_device)
        #     self.state["__sq_weights"] = self.state["__sq_weights"].to(param_device)
        #     self.state["__deviations"] = self.state["__deviations"].to(param_device)
    
    def _restore_original_params(self):
        if self.state["__params_dirty"]:
            for group in self.param_groups:
                for param in group["params"]:
                    state = self.state[param]
                    param.data = state["original_param"].detach().clone()
            self.state["__params_dirty"] = False

    def _save_original_params(self):
        if not self.state["__params_dirty"]:
            for group in self.param_groups:
                for param in group["params"]:
                    state = self.state[param]
                    state["original_param"] = param.data.detach().clone()

    def _swag_update(self):
        if self.state["__epoch"] >= self.start_epoch:
            self.state["__steps_since_swag_start"] += 1

            if self.state["__steps_since_swag_start"] % self.update_interval == 0:
                assert not self.state["__params_dirty"]
                with torch.no_grad():
                    self.state["__updates"] += 1
                    updates = self.state["__updates"]
                    params = parameters_to_vector(self._params()).cpu()
                    self.state["__mean"] = (updates * self.state["__mean"] + params) / (updates + 1)
                    self.state["__sq_weights"] = (updates * self.state["__sq_weights"] + params**2) / (updates + 1)
                    self.state["__deviations"] = torch.roll(self.state["__deviations"], -1, 1)
                    self.state["__deviations"][:,-1] = params - self.state["__mean"]
                    self.param_dist = None

    def _update_param_dist(self):
        if self.param_dist is None:
            device = self._params_device()
            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=False): # Disable autocast for LowRankMultivariateNormal's cholesky_solve
                    diag = 0.5 * (torch.relu(self.state["__sq_weights"].to(device).float() - self.state["__mean"].to(device).float()**2) + 1e-6) # Adding 1e-6 for numerical stability
                    cov_factor = self.state["__deviations"].to(device).float() / math.sqrt(2 * (self.deviation_samples - 1))
                    self.param_dist = torch.distributions.LowRankMultivariateNormal(self.state["__mean"].to(device).float(), cov_factor, diag)
