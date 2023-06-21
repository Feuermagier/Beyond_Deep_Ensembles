import torch
import torch.nn as nn
from src.algos.dropout import FixableDropout
from src.algos.bbb_layers import BBBConv2d, BBBLinear
from src.algos.rank1 import Rank1Linear, Rank1Conv2D
from src.architectures.frn import FilterResponseNorm, VariationalFilterResponseNorm
from src.algos.util import PrintLayer

# Code is adapted from https://github.com/akamaster/pytorch_resnet_cifar10

def get_activation(activation):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "swish":
        return nn.SiLU()
    else:
        raise ValueError("Unknown activation function " + activation)

def get_norm_layer(norm, out_channels, prior=None):
    if norm == "batch_static":
        return nn.BatchNorm2d(out_channels, track_running_stats=False)
    elif norm == "frn":
        if prior is None or isinstance(prior, tuple): # check for tuple to use plain frn on rank1
            return FilterResponseNorm(out_channels)
        else:
            return VariationalFilterResponseNorm(out_channels, prior=prior)
    else:
        raise ValueError("Unknown renormalization layer " + norm)

def get_conv_layer(in_channels, out_channels, kernel_size, stride, padding, bias=True, variational=False, prior=None, rank1=False, components=1):
    if variational:
        if rank1:
            layer = Rank1Conv2D(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, prior=prior, components=components)
            #nn.init.kaiming_normal_(layer.layer.weight.data)
            return layer
        else:
            layer = BBBConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, weight_prior=prior, bias_prior=prior)
            nn.init.kaiming_normal_(layer.weight.mean)
            return layer
    else:
        layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal_(layer.weight.data)
        return layer

def get_linear_layer(in_features, out_features, variational, prior, rank1=False, components=1):
    if variational:
        if rank1:
            return Rank1Linear(in_features, out_features, prior, components=components)
        else:
            return BBBLinear(in_features, out_features, prior, prior)
    else:
        return nn.Linear(in_features, out_features)

########################### ResNet ###########################

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation="relu", norm="batch_static", dropout_p=None, variational=False, rank1=False, prior=None, components=1):
        super().__init__()

        self.main_path = nn.Sequential(
            get_conv_layer(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True, variational=variational, prior=prior, rank1=rank1, components=components),
            FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),
            get_norm_layer(norm, out_channels, prior=prior),
            get_activation(activation),
            
            get_conv_layer(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, variational=variational, prior=prior, rank1=rank1, components=components),
            FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),
            get_norm_layer(norm, out_channels, prior=prior),
            #get_activation(activation),
        )

        if stride != 1:
            self.skip_path = nn.Sequential(
                get_conv_layer(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False, variational=variational, prior=prior, rank1=rank1, components=components),
                FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),
                #get_norm_layer(norm, out_channels, prior=prior)
            )
        else:
            self.skip_path = nn.Identity()

        self.out_activation = get_activation(activation)

    def forward(self, input):
        return self.out_activation(self.main_path(input) + self.skip_path(input))

class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation="relu", norm="batch_static", dropout_p=None, variational=False, rank1=False, prior=None, components=1):
        super().__init__()

        self.main_path = nn.Sequential(
            get_conv_layer(in_channels, in_channels, kernel_size=1, stride=stride, padding=1, bias=True, variational=variational, prior=prior, rank1=rank1, components=components),
            FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),
            get_norm_layer(norm, in_channels, prior=prior),
            get_activation(activation),

            get_conv_layer(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, bias=True, variational=variational, prior=prior, rank1=rank1, components=components),
            FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),
            get_norm_layer(norm, in_channels, prior=prior),
            get_activation(activation),
            
            get_conv_layer(in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=True, variational=variational, prior=prior, rank1=rank1, components=components),
            FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),
            get_norm_layer(norm, out_channels, prior=prior),
            #get_activation(activation),
        )

        if stride != 1:
            self.skip_path = nn.Sequential(
                get_conv_layer(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False, variational=variational, prior=prior, rank1=rank1, components=components),
                FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),
                #get_norm_layer(norm, out_channels, prior=prior)
            )
        else:
            self.skip_path = nn.Identity()

        self.out_activation = get_activation(activation)

    def forward(self, input):
        return self.out_activation(self.main_path(input) + self.skip_path(input))


class ResNet20(nn.Module):
    def __init__(self, in_size, in_channels, classes, activation="relu", norm="batch_static", dropout_p=None, variational=False, prior=None, rank1=False, components=1):
        super().__init__()

        self.model = nn.Sequential(
            get_conv_layer(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=True, variational=variational, prior=prior, rank1=rank1, components=components),
            FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),

            BasicBlock(16, 16, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            BasicBlock(16, 16, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            BasicBlock(16, 16, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),

            BasicBlock(16, 32, 2, activation, norm, dropout_p, variational, rank1, prior, components=components),
            BasicBlock(32, 32, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            BasicBlock(32, 32, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),

            BasicBlock(32, 64, 2, activation, norm, dropout_p, variational, rank1, prior, components=components),
            BasicBlock(64, 64, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            BasicBlock(64, 64, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),

            nn.AvgPool2d(8) if in_size >= 32 else nn.Identity(),
            nn.Flatten(),
            get_linear_layer(64 * (in_size // (32 if in_size >= 32 else 4))**2, classes, variational, prior, rank1=rank1, components=components)
        )

    def forward(self, input):
        return self.model(input)

class ResNet18(nn.Module):
    def __init__(self, in_size, in_channels, classes, activation="relu", norm="batch_static", dropout_p=None, variational=False, rank1=False, prior=None, components=1):
        super().__init__()
        self.model = nn.Sequential(
            get_conv_layer(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=True, variational=variational, prior=prior, rank1=rank1, components=components),
            FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),

            nn.MaxPool2d(kernel_size=3, stride=2),
            BasicBlock(64, 64, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            BasicBlock(64, 64, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),

            BasicBlock(64, 128, 2, activation, norm, dropout_p, variational, rank1, prior, components=components),
            BasicBlock(128, 128, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),

            BasicBlock(128, 256, 2, activation, norm, dropout_p, variational, rank1, prior, components=components),
            BasicBlock(256, 256, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),

            BasicBlock(256, 512, 2, activation, norm, dropout_p, variational, rank1, prior, components=components),
            BasicBlock(512, 512, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            get_linear_layer(512, classes, variational, prior, rank1=rank1, components=components)
        )

    def forward(self, input):
        return self.model(input)

class ResNet50(nn.Module):
    def __init__(self, in_size, in_channels, classes, activation="relu", norm="batch_static", dropout_p=None, variational=False, rank1=False, prior=None, components=1):
        super().__init__()

        self.model = nn.Sequential(
            get_conv_layer(in_channels, 64, kernel_size=7, stride=1, padding=1, bias=True, variational=variational, prior=prior, rank1=rank1, components=components),
            FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),

            nn.MaxPool2d(kernel_size=3, stride=2),
            Bottleneck(64, 256, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            Bottleneck(64, 256, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            Bottleneck(64, 256, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),

            Bottleneck(128, 512, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            Bottleneck(128, 512, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            Bottleneck(128, 512, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            Bottleneck(128, 512, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),

            Bottleneck(256, 1024, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            Bottleneck(256, 1024, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            Bottleneck(256, 1024, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            Bottleneck(256, 1024, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            Bottleneck(256, 1024, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            Bottleneck(256, 1024, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),

            Bottleneck(512, 2048, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            Bottleneck(512, 2048, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            Bottleneck(512, 2048, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            get_linear_layer(2048, classes, variational, prior, rank1=rank1, components=components)
        )

    def forward(self, input):
        return self.model(input)

########################### PreResNet ###########################

class PreBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, activation="relu", norm="batch_static", dropout_p=None, variational=False, prior=None, rank1=False, components=1):
        super().__init__()

        self.main_path = nn.Sequential(
            get_norm_layer(norm, in_channels, prior=prior),
            get_activation(activation),
            FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),
            get_conv_layer(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True, variational=variational, prior=prior, rank1=rank1, components=components),
            
            get_norm_layer(norm, out_channels, prior=prior),
            get_activation(activation),
            FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),
            get_conv_layer(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, variational=variational, prior=prior, rank1=rank1, components=components),
        )

        if stride != 1:
            self.skip_path = nn.Sequential(
                FixableDropout(dropout_p) if dropout_p != None else nn.Identity(),
                get_conv_layer(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False, variational=variational, prior=prior, rank1=rank1, components=components)
            )
        else:
            self.skip_path = nn.Identity()

    def forward(self, input):
        return self.main_path(input) + self.skip_path(input)

class PreResNet20(nn.Module):
    def __init__(self, in_size, in_channels, classes, activation="relu", norm="batch_static", dropout_p=None, variational=False, prior=None, rank1=False, components=1):
        super().__init__()

        self.model = nn.Sequential(
            get_conv_layer(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=True, variational=variational, prior=prior, rank1=rank1, components=components),

            PreBasicBlock(16, 16, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            PreBasicBlock(16, 16, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            PreBasicBlock(16, 16, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),

            PreBasicBlock(16, 32, 2, activation, norm, dropout_p, variational, rank1, prior, components=components),
            PreBasicBlock(32, 32, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            PreBasicBlock(32, 32, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),

            PreBasicBlock(32, 64, 2, activation, norm, dropout_p, variational, rank1, prior, components=components),
            PreBasicBlock(64, 64, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),
            PreBasicBlock(64, 64, 1, activation, norm, dropout_p, variational, rank1, prior, components=components),

            get_norm_layer(norm, 64, prior=prior),
            get_activation(activation),
            nn.AvgPool2d(8) if in_size >= 32 else nn.Identity(),

            nn.Flatten(),
            get_linear_layer(64 * (in_size // (32 if in_size >= 32 else 4))**2, classes, variational, prior, rank1=rank1, components=components)
        )

    def forward(self, input):
        return self.model(input)
