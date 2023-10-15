import torch
import torch.nn as nn

from src.algos.util import GaussLayer, reset_model_params
from src.algos.swag import SwagOptimizer
from src.algos.bbb import BBBOptimizer, GaussianPrior
from src.algos.ensemble import DeepEnsemble
from src.algos.pp import MAPOptimizer
from src.algos.svgd import SVGDOptimizer
from src.algos.ivorn import iVONOptimizer
from src.algos.kernel.sngp import SNGPWrapper, SNGPOptimizer
from src.algos.kernel.base import spectrally_normalize_module

from src.architectures.resnet import ResNet18

RESNET_OUT_DIMS = 512

def get_var_optimizer(model, config):
    return torch.optim.SGD(model[-1].parameters(), **config["var_optimizer"])

def get_model(model, device, config):
    if model == "map":
        model_fn = build_map
    elif model == "swag":
        model_fn = build_swag
    elif model == "mcd":
        model_fn = build_mcd
    elif model == "bbb":
        model_fn = build_bbb
    elif model == "rank1":
        model_fn = build_rank1
    elif model == "laplace_base":
        model_fn = build_laplace_base
    elif model == "svgd":
        model_fn = build_svgd
    elif model == "ivon":
        model_fn = build_ivon
    elif model == "sngp":
        model_fn = build_sngp
    else:
        raise ValueError(f"Unknown model type '{model}'")
    
    return DeepEnsemble([model_fn(config, device) for _ in range(config["members"])])

def build_map(config, device):
    model = nn.Sequential(
        ResNet18(224, 8, 1),
        GaussLayer(torch.tensor(config["init_std"]), config["learn_var"])
    ).to(device)
    optimizer = MAPOptimizer(model[0].parameters(), torch.optim.Adam(model[0].parameters(), **config["base_optimizer"]))
    return model, optimizer

def build_laplace_base(config, device):
    model = nn.Sequential(
        ResNet18(224, 8, 1),
        #GaussLayer(torch.tensor(config["init_std"]), config["learn_var"])
    ).to(device)
    optimizer = MAPOptimizer(model[0].parameters(), torch.optim.Adam(model[0].parameters(), **config["base_optimizer"]))
    return model, optimizer

def build_mcd(config, device):
    model = nn.Sequential(
        ResNet18(224, 8, 1, dropout_p=config["dropout_p"]),
        GaussLayer(torch.tensor(config["init_std"]), config["learn_var"])
    ).to(device)
    optimizer = MAPOptimizer(model[0].parameters(), torch.optim.Adam(model[0].parameters(), **config["base_optimizer"]))
    return model, optimizer

def build_swag(config, device):
    model = nn.Sequential(
        ResNet18(224, 8, 1),
        GaussLayer(torch.tensor(config["init_std"]), config["learn_var"])
    ).to(device)
    optimizer = SwagOptimizer(model[0].parameters(), torch.optim.Adam(model[0].parameters(), **config["base_optimizer"]), **config["swag"])
    return model, optimizer

def build_bbb(config, device):
    prior = GaussianPrior(0, config["prior_std"])
    model = nn.Sequential(
        ResNet18(224, 8, 1, variational=True, prior=prior),
        GaussLayer(torch.tensor(config["init_std"]), config["learn_var"])
    ).to(device)
    optimizer = BBBOptimizer(model[0].parameters(), torch.optim.Adam(model[0].parameters(), **config["base_optimizer"]), prior=prior, **config["bbb"])
    return model, optimizer

def build_rank1(config, device):
    prior = GaussianPrior(0, config["prior_std"])
    model = nn.Sequential(
        ResNet18(224, 8, 1, variational=True, rank1=True, prior=prior, components=config["rank1"]["components"]),
        GaussLayer(torch.tensor(config["init_std"]), config["learn_var"])
    ).to(device)
    optimizer = BBBOptimizer(model[0].parameters(), torch.optim.Adam(model[0].parameters(), **config["base_optimizer"]), prior=prior, **config["rank1"])
    return model, optimizer

def build_svgd(config, device):
    model = nn.Sequential(
        ResNet18(224, 8, 1),
        GaussLayer(torch.tensor(config["init_std"]), config["learn_var"])
    ).to(device)
    def reset():
        reset_model_params(model)
    optimizer = SVGDOptimizer(model[0].parameters(), reset, torch.optim.Adam(model[0].parameters(), **config["base_optimizer"]), **config["svgd"])
    return model, optimizer

def build_ivon(config, device):
    model = nn.Sequential(
        ResNet18(224, 8, 1),
        GaussLayer(torch.tensor(config["init_std"]), config["learn_var"])
    ).to(device)
    optimizer = iVONOptimizer(model[0].parameters(), **config["ivon"])
    return model, optimizer

def build_sngp(config, device):
    model = ResNet18(224, 8, -1) # Set classes to -1 to omit the final linear layer
    spectrally_normalize_module(model, norm_bound=config["spectral"]["norm_bound"])
    sngp = SNGPWrapper(
        model,
        GaussLayer(torch.tensor(config["init_std"]), config["learn_var"]),
        outputs=1,
        num_deep_features=RESNET_OUT_DIMS,
        **config["sngp"]
    ).to(device)
    optimizer = SNGPOptimizer(sngp, torch.optim.Adam(sngp.parameters(), **config["base_optimizer"]))
    return sngp, optimizer
