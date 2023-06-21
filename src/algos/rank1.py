import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import parameters_to_vector
from src.algos.util import GaussianParameter

import math

class Rank1Linear(nn.Module):
    def __init__(self, in_features, out_features, prior, bias=True, components=1):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.components = components
        # self.prior = prior[0]
        # self.l2_scale = prior[1]
        self.layer = nn.Linear(in_features, out_features, bias=False)

        # self.s_mean = nn.Parameter(torch.empty(in_features))
        # self.s_rho = nn.Parameter(torch.empty(in_features))
        self.s = torch.nn.ModuleList([GaussianParameter(in_features) for _ in range(components)])

        # self.r_mean = nn.Parameter(torch.empty(out_features))
        # self.r_rho = nn.Parameter(torch.empty(out_features))
        self.r = torch.nn.ModuleList([GaussianParameter(out_features) for _ in range(components)])

        # Implement bias manually according to the paper
        if bias:
            self.bias = nn.Parameter(torch.empty((components, out_features)))
        else:
            self.bias = None

        self.component_counter = 0

        self.reset_parameters()

    def reset_parameters(self):
        self.layer.reset_parameters()

        for s in self.s:
            s.sign_init()
        for r in self.r:
            r.sign_init()

        if self.bias is not None:
            # From the linear layer of pytorch
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.layer.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        s = self.s[self.component_counter].sample()
        r = self.r[self.component_counter].sample()

        # s_kl = self.s[self.component_counter].kl_divergence(self.prior)
        # r_kl = self.r[self.component_counter].kl_divergence(self.prior)
        # l2 = parameters_to_vector(self.layer.parameters()).pow(2).sum()
        # self.kl = s_kl + r_kl + self.l2_scale / 2 * l2

        output = self.layer(input * s) * r
        if self.bias is not None:
            output += torch.unsqueeze(self.bias[self.component_counter], 0)

        self.component_counter = (self.component_counter + 1) % self.components
        return output

class Rank1Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, prior, stride=1, padding=0, bias=True, components=1):
        super().__init__()
        self.layer = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.components = components
        # self.prior = prior[0]
        # self.l2_scale = prior[1]

        self.s = torch.nn.ModuleList([GaussianParameter(in_channels) for _ in range(components)])
        # self.s_mean = nn.Parameter(torch.empty(in_channels))
        # self.s_rho = nn.Parameter(torch.empty(in_channels))

        self.r = torch.nn.ModuleList([GaussianParameter(out_channels) for _ in range(components)])
        # self.r_mean = nn.Parameter(torch.empty(out_channels))
        # self.r_rho = nn.Parameter(torch.empty(out_channels))

        # Implement bias manually according to the paper
        if bias:
            self.bias = nn.Parameter(torch.empty((components, out_channels)))
        else:
            self.bias = None

        self.component_counter = 0

        self.reset_parameters()

    def reset_parameters(self):
        self.layer.reset_parameters()

        for s in self.s:
            s.sign_init()
        for r in self.r:
            r.sign_init()

        if self.bias is not None:
            # From Pytorch's Conv2D
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.layer.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        s = self.s[self.component_counter].sample()
        r = self.r[self.component_counter].sample()

        # s_kl = self.s[self.component_counter].kl_divergence(self.prior)
        # r_kl = self.r[self.component_counter].kl_divergence(self.prior)
        # l2 = parameters_to_vector(self.layer.parameters()).pow(2).sum()
        # self.kl = s_kl + r_kl + self.l2_scale / 2 * l2

        s = s.unsqueeze(-1).unsqueeze(-1)
        r = r.unsqueeze(-1).unsqueeze(-1)
        output = self.layer(input * s) * r

        if self.bias is not None:
            output += torch.tile(self.bias[self.component_counter].unsqueeze(-1).unsqueeze(-1), (output.shape[-2], output.shape[-1]))

        self.component_counter = (self.component_counter + 1) % self.components
        return output

def make_module_rank1(module, prior, components):
    '''
        Converts all nn.Linear and nn.Conv2d layers of the given model to their Rank-1 VI counterparts.
        The mean values are initialized from the original weights of the layers.
        Note that the kernel of convolutional layers is assumed to be of quadratic shape.
    '''
    for name, m in list(module._modules.items()):
        if m._modules:
            make_module_rank1(m, prior, components)
        elif "Conv2d" in m.__class__.__name__:
            bbb_layer = Rank1Conv2D(in_channels=m.in_channels, out_channels=m.out_channels, kernel_size=m.kernel_size[0], prior=prior, stride=m.stride, bias=m.bias is not None, padding=m.padding, components=components)
            bbb_layer.layer = m
            if m.bias is not None:
                bbb_layer.bias.overwrite_mean(m.bias.data.clone())
            setattr(module, name, bbb_layer)
        elif "Linear" in m.__class__.__name__:
            bbb_layer = Rank1Linear(in_features=m.in_features, out_features=m.out_features, prior=prior, bias=m.bias is not None)
            bbb_layer.layer = m
            if m.bias is not None:
                bbb_layer.bias.overwrite_mean(m.bias.data.clone())
            setattr(module, name, bbb_layer)
        else:
            pass
