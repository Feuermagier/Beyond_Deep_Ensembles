import torch
import torch.nn as nn
from .algo import BayesianOptimizer
from .opt_util import apply_lr

class MAPOptimizer(BayesianOptimizer):
    '''
        Maximum A Posteriori

        This simply optimizes a point estimate of the parameters with the given base_optimizer.
    '''

    def __init__(self, params, base_optimizer):
        super().__init__(params, {})
        self.state["__base_optimizer"] = base_optimizer

    def step(self, forward_closure, backward_closure, grad_scaler=None):
        self.zero_grad()

        loss = forward_closure()
        backward_closure(loss)

        if grad_scaler is not None:
            grad_scaler.step(self.state["__base_optimizer"])
        else:
            self.state["__base_optimizer"].step()

        return loss

    def sample_parameters(self):
        pass

    def get_base_optimizer(self):
        return self.state["__base_optimizer"]
