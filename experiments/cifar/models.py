import torch
import torch.nn as nn

from src.architectures.resnet import ResNet20

from src.algos.swag import SwagOptimizer
from src.algos.bbb import GaussianPrior, BBBOptimizer
from src.algos.bbb_layers import BBBLinear
from src.algos.rank1 import Rank1Linear
from src.algos.ensemble import DeepEnsemble
from src.algos.pp import MAPOptimizer
from src.algos.ivorn import iVONOptimizer
from src.algos.svgd import SVGDOptimizer
from src.algos.dropout import patch_dropout, FixableDropout
from src.architectures.bert import BertClassifier
from src.algos.util import reset_model_params

def get_model(model, config, device):
    if model == "map":
        model_fn = build_map
    if model == "laplace":
        model_fn = build_map # Fitting Laplace post-hoc
    elif model == "swag":
        model_fn = build_swag
    elif model == "mcd":
        model_fn = build_mcd
    elif model == "bbb":
        model_fn = build_bbb
    elif model == "ivon":
        model_fn = build_ivon
    elif model == "rank1":
        model_fn = build_rank1
    elif model == "svgd":
        model_fn = build_svgd
    else:
        raise ValueError(f"Unknown model type '{model}'")

    return build_ensemble(model_fn, config, device)

def build_ensemble(fn, config, device):
    return DeepEnsemble([fn(config, device) for _ in range(config["members"])])

def build_map(config, device):
    model = nn.Sequential(
        ResNet20(32, 3, 10, "swish", "frn"),
        nn.LogSoftmax(dim=1)
    ).to(device)
    optimizer = MAPOptimizer(model.parameters(), torch.optim.SGD(model.parameters(), **config["base_optimizer"]))
    return model, optimizer

def build_swag(config, device):
    model = nn.Sequential(
        ResNet20(32, 3, 10, "swish", "frn"),
        nn.LogSoftmax(dim=1)
    ).to(device)
    optimizer = SwagOptimizer(model.parameters(), torch.optim.SGD(model.parameters(), **config["base_optimizer"]), **config["swag"])
    return model, optimizer

def build_mcd(config, device):
    model = nn.Sequential(
        ResNet20(32, 3, 10, "swish", "frn", dropout_p=config["p"]),
        nn.LogSoftmax(dim=1)
    ).to(device)
    optimizer = MAPOptimizer(model.parameters(), torch.optim.SGD(model.parameters(), **config["base_optimizer"]))
    return model, optimizer

def build_bbb(config, device):
    prior = GaussianPrior(0, config["prior_std"])
    model = nn.Sequential(
        ResNet20(32, 3, 10, "swish", "frn", variational=True, prior=prior),
        nn.LogSoftmax(dim=1)
    ).to(device)
    optimizer = BBBOptimizer(model.parameters(), torch.optim.SGD(model.parameters(), **config["base_optimizer"]), prior=prior, **config["bbb"])
    return model, optimizer

def build_ivon(config, device):
    model = nn.Sequential(
        ResNet20(32, 3, 10, "swish", "frn"),
        nn.LogSoftmax(dim=1)
    ).to(device)
    optimizer = iVONOptimizer(model.parameters(), **config["ivon"])
    return model, optimizer

def build_rank1(config, device):
    prior = GaussianPrior(0, config["prior_std"])
    model = nn.Sequential(
        ResNet20(32, 3, 10, "swish", "frn", variational=True, prior=prior, rank1=True, components=config["components"]),
        nn.LogSoftmax(dim=1)
    ).to(device)
    optimizer = BBBOptimizer(model.parameters(), torch.optim.SGD(model.parameters(), **config["base_optimizer"]), prior=prior, **config["bbb"])
    return model, optimizer

def build_svgd(config, device):
    model = nn.Sequential(
        ResNet20(32, 3, 10, "swish", "frn"),
        nn.LogSoftmax(dim=1)
    ).to(device)
    def reset_model():
        reset_model_params(model)
    optimizer = SVGDOptimizer(model.parameters(), reset_model, torch.optim.SGD(model.parameters(), **config["base_optimizer"]), **config["svgd"])
    return model, optimizer
