import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussLayer(nn.Module):
    def __init__(self, std_init: torch.Tensor, learn_var: bool = False):
        super().__init__()
        if learn_var:
            self.rho = nn.Parameter(torch.log(torch.exp(std_init) - 1))
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