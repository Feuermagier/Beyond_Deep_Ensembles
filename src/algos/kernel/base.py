import torch
import torch.nn as nn
import torch.nn.functional as F
from . import spectral_norm

def spectrally_normalize_module(module: nn.Module, norm_bound, power_iterations=1):
    for name, m in list(module._modules.items()):
        if m._modules:
            spectrally_normalize_module(m, norm_bound=norm_bound, power_iterations=power_iterations)
        elif "Linear" in m.__class__.__name__:
            normalized_layer = spectral_norm.spectral_norm(m, norm_bound=norm_bound, n_power_iterations=power_iterations)
            setattr(module, name, normalized_layer)
        elif "Conv2d" in m.__class__.__name__:
            normalized_layer = spectral_norm.spectral_norm(m, norm_bound=norm_bound, n_power_iterations=power_iterations)
            setattr(module, name, normalized_layer)
        else:
            pass
