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
from src.algos.kernel.sngp import SNGPWrapper, SNGPOptimizer
from src.algos.kernel.base import spectrally_normalize_module
from src.algos.util import reset_model_params

def get_model(model, config, device):
    if model == "map":
        model_fn = build_map
    elif model == "laplace":
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
    elif model == "sngp":
        model_fn = build_sngp
    else:
        raise ValueError(f"Unknown model type '{model}'")

    return build_ensemble(model_fn, config, device)

def build_ensemble(fn, config, device):
    return DeepEnsemble([fn(config, device) for _ in range(config["members"])])

def build_map(config, device):
    model = _get_model(dict(), config, device)
    optimizer = MAPOptimizer(model.parameters(), torch.optim.SGD(model.parameters(), **config["base_optimizer"]))
    return model, optimizer

def build_swag(config, device):
    model = _get_model(dict(), config, device)
    optimizer = SwagOptimizer(model.parameters(), torch.optim.SGD(model.parameters(), **config["base_optimizer"]), **config["swag"])
    return model, optimizer

def build_mcd(config, device):
    model = _get_model(dict(dropout_p=config["p"]), config, device)
    optimizer = MAPOptimizer(model.parameters(), torch.optim.SGD(model.parameters(), **config["base_optimizer"]))
    return model, optimizer

def build_bbb(config, device):
    prior = GaussianPrior(0, config["prior_std"])
    model = _get_model(dict(variational=True, prior=prior), config, device)
    optimizer = BBBOptimizer(model.parameters(), torch.optim.SGD(model.parameters(), **config["base_optimizer"]), prior=prior, **config["bbb"])
    return model, optimizer

def build_ivon(config, device):
    model = _get_model({}, config, device)
    optimizer = iVONOptimizer(model.parameters(), **config["ivon"])
    return model, optimizer

def build_rank1(config, device):
    prior = GaussianPrior(0, config["prior_std"])
    model = _get_model(dict(variational=True, prior=prior, rank1=True, components=config["components"]), config, device)
    optimizer = BBBOptimizer(model.parameters(), torch.optim.SGD(model.parameters(), **config["base_optimizer"]), prior=prior, **config["bbb"])
    return model, optimizer

def build_svgd(config, device):
    model = _get_model({}, config, device)
    def reset_model():
        reset_model_params(model)
    optimizer = SVGDOptimizer(model.parameters(), reset_model, torch.optim.SGD(model.parameters(), **config["base_optimizer"]), **config["svgd"])
    return model, optimizer

def build_sngp(config, device):
    model = ResNet20(32, 3, 10, "swish", "frn")
    featurizer_out_size = model.model[-1].in_features
    model.model[-1] = nn.Identity()
    spectrally_normalize_module(model, norm_bound=config["spectral"]["norm_bound"])
    print(model)
    sngp = SNGPWrapper(
        model,
        nn.LogSoftmax(dim=1),
        outputs=10,
        num_deep_features=featurizer_out_size,
        **config["sngp"]
    ).to(device)
    optimizer = SNGPOptimizer(sngp, torch.optim.SGD(model.parameters(), **config["base_optimizer"]))
    return sngp, optimizer

def _get_model(resnet_params, config, device) -> nn.Module:
    return torch.compile(nn.Sequential(
        ResNet20(32, 3, 10, "swish", "frn", **resnet_params),
        nn.LogSoftmax(dim=1)
    ).to(device), disable=not config["use_compile"])
