import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.convert_parameters import parameters_to_vector
from itertools import chain

from .util import non_mle_params
from .algo import BayesianOptimizer
from .opt_util import apply_lr

# Code is similar to https://github.com/activatedgeek/svgd and the original implementation at https://github.com/DartML/Stein-Variational-Gradient-Descent/tree/master/python
# Code for the RBF with the median heuristic is basically copied from the original implementation

def rbf(particles, h_override=None):
    distances = torch.cdist(particles, particles, p=2)**2

    if h_override is None:
        h = torch.sqrt(0.5 * torch.quantile(distances, 0.5) / np.log(particles.shape[0] + 1)) + 1e-8
    else:
        h = h_override
    kernel = torch.exp(-distances / (2 * h**2))

    grad_kernel = kernel.sum(dim=1).unsqueeze(-1) * particles - torch.matmul(kernel, particles)

    # Original implementation by Liu et al.:
    # grad_kernel = -torch.matmul(kernel, particles)
    # sums = torch.sum(kernel, dim=1)
    # for i in range(particles.shape[1]):
    #     grad_kernel[:, i] += particles[:, i] * sums

    grad_kernel /= h**2
    return kernel, grad_kernel

def filter_parameters(named_parameters, excluded_names):
    return map((lambda n, p: p), filter((lambda n, p: n not in excluded_names), named_parameters))

class SVGDOptimizer(BayesianOptimizer):
    '''
        Stein Variational Gradient Descent

        This optimizer does not support multiple parameter groups, as they are used to differentiate between the particles
    '''

    def __init__(self, params, reset_params_closure, base_optimizer, particle_count, dataset_size, l2_reg=0.0, kernel_grad_scale=1.0):
        '''
            reset_params_closure is a closure that re-initializes the parameters (could be as simple as calling module.reset_parameters() when all parameters are being optimized)

            base_optimizer must optimize all parameters that are optimized by this optimizier (i.e. use itertools.chain(*[particle.parameters() for particle in particles]))
        '''
        super().__init__(map(lambda p: {"params": p}, params), {})
        self.state["__base_optimizer"] = base_optimizer
        self.state["__l2_reg"] = l2_reg
        self.state["__dataset_size"] = dataset_size
        self.state["__current_particle"] = 0
        self.state["__particle_count"] = particle_count
        self.state["__kernel_grad_scale"] = kernel_grad_scale

        for particle_idx in range(particle_count):
            for group in self.param_groups:
                for param in group["params"]:
                    self.state[param][f"particle_{particle_idx}"] = param.detach().clone()
            if particle_idx < particle_count - 1:
                reset_params_closure()

    def step(self, forward_closure, backward_closure, grad_scaler=None):
        total_loss = torch.tensor(0.0, device=self._params_device())
        for particle_idx in range(self.state["__particle_count"]):
            self._set_grad_scaler_state(grad_scaler, torch.cuda.amp.grad_scaler.OptState.READY, self.state["__base_optimizer"])
            self._use_particle(particle_idx)
            self.state["__base_optimizer"].zero_grad()

            loss = forward_closure()
            total_loss += loss.detach()

            # It would be more performant to do one backward pass on the total loss, but we need to store the individual gradients
            # Sadly backward passes don't go through nn.Parameters, as their backward function is set to a NOOP (see torch.nn.Parameter: __torch_function__ = _disabled_torch_function_impl)
            backward_closure(loss)
            if not self._prepare_and_check_grads(grad_scaler, self.state["__base_optimizer"]):
                return None
            self._store_grads(particle_idx)

        with torch.no_grad():
            param_vecs = torch.stack([parameters_to_vector(self._params_for_particle(i)) for i in range(self.state["__particle_count"])])
            gradient_vecs = torch.stack([torch.cat([p.grad.view(-1) for p in self._params_for_particle(i)]) for i in range(self.state["__particle_count"])])

            gradient_vecs += self.state["__l2_reg"] / 2 * param_vecs # Prior (L2 reg)
            kernel, grad_kernel = rbf(param_vecs)

            phi = torch.matmul(kernel, -gradient_vecs) + self.state["__kernel_grad_scale"] * grad_kernel / self.state["__dataset_size"]

            # Write the modified gradients of each particle TO THE ORIGINAL PARAMETERS and call the optimizes on them
            for particle_idx in range(self.state["__particle_count"]):
                offset = 0
                for model_param, particle_param in zip(self._params(), self._params_for_particle(particle_idx)):
                    model_param.grad = -phi[particle_idx, offset:offset + model_param.numel()].view_as(model_param).clone()
                    model_param.data = particle_param # Connect the parameters so that the optimizer updates the particle's parameters
                    offset += model_param.numel()

                if grad_scaler is not None:
                    self._set_grad_scaler_state(grad_scaler, torch.cuda.amp.grad_scaler.OptState.UNSCALED, self.state["__base_optimizer"])
                    grad_scaler.step(self.state["__base_optimizer"])
                else:
                    self.state["__base_optimizer"].step()

        return total_loss / self.state["__particle_count"]

    def sample_parameters(self):
        '''
            Cycles through the particles
        '''
        self._use_particle(self.state["__current_particle"])
        self.state["__current_particle"] = (self.state["__current_particle"] + 1) % self.state["__particle_count"]

    def _params_for_particle(self, particle_idx):
        particle = f"particle_{particle_idx}"
        for group in self.param_groups:
            for param in group["params"]:
                yield self.state[param][particle]

    def _use_particle(self, particle_idx):
        '''
            Importantly, this does *not* clone the parameters, so that gradients of the model parameters are reflected in the respective particle parameters
        '''
        particle = f"particle_{particle_idx}"
        for group in self.param_groups:
            for param in group["params"]:
                param.data = self.state[param][particle]

    def _store_grads(self, particle_idx):
        particle = f"particle_{particle_idx}"
        for group in self.param_groups:
            for param in group["params"]:
                self.state[param][particle].grad = param.grad.clone()

    def get_base_optimizer(self):
        return self.state["__base_optimizer"]