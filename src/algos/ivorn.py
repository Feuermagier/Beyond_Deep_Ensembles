import torch
import torch.nn as nn
import math

from src.algos.util import normal_like
from .algo import BayesianOptimizer

class iVONOptimizer(BayesianOptimizer):
    '''
        Improved Variational Online Newton

        This class implements hacky workarounds around the limitations of PyTorch's GradScaler. Therefore it depends heavily on the internals of the GradScaler and may stop working in future PyTorch releases. Tested with PyTorch 2.0
    '''

    def __init__(self, params, lr, prior_prec, dataset_size, betas=(0.9, 0.999), damping=0.0, tempering=1.0, augmentation=1.0, mc_samples=5, deterministic=False):
        defaults = {
            "lr": lr,
            "betas": betas,
            "prior_prec": prior_prec,
            "damping": damping,
            "tempering": tempering,
            "augmentation": augmentation,
            "N": dataset_size,
            "deterministic": deterministic,
            "step": 0
        }
        super().__init__(params, defaults)

        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                state["mean"] = param.data.detach().clone()
                state["momentum"] = torch.zeros_like(param)
                state["precision"] = torch.full_like(param, group["prior_prec"] / group["N"])
                state["delta"] = None
                state["acc_grad"] = None

        assert mc_samples > 0
        self.mc_samples = mc_samples

    def step(self, forward_closure, backward_closure, grad_scaler: torch.cuda.amp.GradScaler=None):
        self._reset_state()

        acc_loss = None
        for _ in range(self.mc_samples):
            # Set our state to READY so that the GradScaler does not complain when calling unscale_
            self._set_grad_scaler_state(grad_scaler, torch.cuda.amp.grad_scaler.OptState.READY)

            self.sample_parameters()
            with torch.enable_grad():
                self.zero_grad()
                loss = forward_closure()
                backward_closure(loss)

            if acc_loss is None:
                acc_loss = loss
            else:
                acc_loss += loss

            if not self._prepare_and_check_grads(grad_scaler):
                return None

            self._store_gradients()
        acc_loss /= self.mc_samples

        with torch.no_grad():
            for group in self.param_groups:
                group["step"] += 1
                t = group["step"]
                beta1, beta2 = group["betas"]
                N = group["N"] * group["augmentation"]
                #TODO fix tempering: is it required in the precision update? (see VOGN paper)
                delta = group["tempering"] * group["prior_prec"] / N
                lr = group["lr"]

                for param in group["params"]:
                    state = self.state[param]

                    gradient = state["acc_grad"] / self.mc_samples
                    g_mu = delta * state["mean"] + gradient
                    state["momentum"] = beta1 * state["momentum"] + (1 - beta1) * g_mu
                    g_s = delta - state["precision"] + (N * state["precision"] * state["delta"] / self.mc_samples) * gradient + group["damping"]

                    corrected_momentum = state["momentum"] / (1 - beta1**t)
                    corrected_precision = state["precision"] / (1 - beta2**t)
                    #state["mean"] -= lr * param.grad
                    #print(corrected_precision.mean())
                    state["mean"] -= lr * corrected_momentum / corrected_precision
                    state["precision"] += ((1 - beta2) + 0.5 * (1 - beta2)**2 * g_s / state["precision"]) * g_s

        self._set_grad_scaler_state(grad_scaler, torch.cuda.amp.grad_scaler.OptState.STEPPED)

        return acc_loss

    def _reset_state(self):
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                state["delta"] = None
                state["acc_grad"] = None

    def sample_parameters(self):
        for group in self.param_groups:
            N = group["N"] * group["augmentation"]
            for param in group["params"]:
                state = self.state[param]
                if not group["deterministic"]:
                    delta = 1 / (N * state["precision"].clamp(min=1e-4)).sqrt() * normal_like(state["precision"])
                else:
                    delta = torch.zeros_like(state["precision"])
                param.data = state["mean"] + delta
                if state["delta"] is None:
                    state["delta"] = delta
                else:
                    state["delta"] += delta

    def get_base_optimizer(self):
        return self

    def _store_gradients(self):
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                if state["acc_grad"] is None:
                    state["acc_grad"] = param.grad
                else:
                    state["acc_grad"] += param.grad
