import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import itertools
from src.resnet import PreResNet20, ResNet20, ResNet18
from src.unet import UNet
from src.densenet import MidasTiramisu
from src.bert import BertClassifier

from src.algos.util import GaussLayer
from src.bbb_layers import BBBLinear, LowRankBBBLinear, BBBConvolution
from src.dropout import FixableDropout
from src.rank1 import Rank1Linear, Rank1Conv2D


def map_activation(name):
    if name == "relu":
        return nn.ReLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "logsoftmax":
        return nn.LogSoftmax(dim=1)
    else:
        raise ValueError(f"Unknown activation function {name}")


def generate_model(architecture, print_summary=False, parallel=False, device=None):
    layers = []
    for i, (ty, size) in enumerate(architecture):
        if ty == "pool":
            layers.append(nn.MaxPool2d(size))
        elif ty == "flatten":
            layers.append(nn.Flatten())
        elif ty == "relu":
            layers.append(nn.ReLU())
        elif ty == "sigmoid":
            layers.append(nn.Sigmoid())
        elif ty == "logsoftmax":
            layers.append(nn.LogSoftmax(dim=1))
        elif ty == "fc":
            (in_features, out_features) = size
            layers.append(nn.Linear(in_features, out_features))
        elif ty == "v_fc":
            (in_features, out_features, prior, args) = size
            layers.append(BBBLinear(in_features, out_features, prior, prior, **args))
        elif ty == "vlr_fc":
            (in_features, out_features, K, gamma, args) = size
            layers.append(LowRankBBBLinear(in_features, out_features, gamma, K, **args))
        elif ty == "rank1_fc":
            (in_features, out_features, prior, args) = size
            layers.append(Rank1Linear(in_features, out_features, prior, **args))
        elif ty == "conv":
            (in_channels, out_channels, kernel_size) = size
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size))
        elif ty == "v_conv":
            (in_channels, out_channels, kernel_size, prior, args) = size
            layers.append(
                BBBConvolution(
                    in_channels, out_channels, kernel_size, prior, prior, **args
                )
            )
        elif ty == "rank1_conv":
            (in_channels, out_channels, kernel_size, prior, args) = size
            layers.append(
                Rank1Conv2D(in_channels, out_channels, kernel_size, prior, **args)
            )
        elif ty == "dropout":
            (p,) = size
            layers.append(FixableDropout(p))
        elif ty == "gauss":
            std, learn_std = size
            layers.append(GaussLayer(std, learn_std))
        elif ty == "unet":
            layers.append(UNet())
        elif ty == "midas":
            layers.append(MidasTiramisu())
        elif "resnet" in ty:
            preresnet = try_build_resnet(ty, size)
            if preresnet is not None:
                layers.append(preresnet)
            else:
                raise ValueError(f"Unknown resnet type '{ty}'")
        elif "bert" in ty:
            bert = try_build_bert(ty, size)
            if bert is not None:
                layers.append(bert)
            else:
                raise ValueError(f"Unknown bert type '{ty}'")
        else:
            raise ValueError(f"Unknown layer type '{ty}'")

    model = nn.Sequential(*layers)

    if device is not None:
        model.to(device)

    if print_summary:
        print(f"Generated model: {model}")
        print(
            f"{sum([p.numel() for p in model.parameters() if p.requires_grad])} trainable parameters"
        )

    if parallel:
        return nn.DistributedDataParallel(model)
    else:
        return model


def try_build_resnet(layer: str, args):
    if "preresnet-20" in layer:
        net = PreResNet20
    elif "resnet-20" in layer:
        net = ResNet20
    elif "resnet-18" in layer:
        net = ResNet18
    else:
        return None

    if len(args) == 4:
        in_size, in_channels, classes, extra_arg = args
    elif len(args) == 3:
        in_size, in_channels, classes = args
    else:
        raise ValueError("Wrong number of arguments for a resnet")

    p = None
    prior = None
    rank1 = False
    components = 1
    if layer.startswith("variational"):
        prior = extra_arg
    elif layer.startswith("drop"):
        p = extra_arg
    elif layer.startswith("rank1"):
        prior, components = extra_arg
        rank1 = True

    if layer.endswith("-swish-frn"):
        activation = "swish"
        norm = "frn"
    else:
        activation = "relu"
        norm = "batch_static"

    return net(
        in_size,
        in_channels,
        classes,
        activation=activation,
        norm=norm,
        dropout_p=p,
        variational=prior is not None,
        prior=prior,
        rank1=rank1,
        components=components,
    )


def try_build_bert(layer: str, args):
    classes, extra_arg = args
    prior = None
    drop_p = None
    components = None

    if layer.startswith("drop"):
        ty = "drop"
        drop_p = extra_arg
    elif layer.startswith("bbb"):
        ty = "bbb"
        prior = extra_arg
    elif layer.startswith("rank1"):
        ty = "rank1"
        prior, components = extra_arg
    else:
        ty = "map"

    return BertClassifier(
        ty=ty, classes=classes, prior=prior, drop_p=drop_p, components=components
    )
