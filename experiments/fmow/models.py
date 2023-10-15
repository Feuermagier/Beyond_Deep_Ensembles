import torch
import torch.nn as nn
from torchvision.models import densenet121

from src.algos.algo import LastLayerBayesianOptimizer
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
from src.algos.util import reset_model_params, patch_batchnorm

DENSENET_OUT_DIMS = 1024
N_CLASSES = 62

def get_model(model, config, device):
    if model == "map":
        model_fn = build_map
    elif model == "swag":
        model_fn = build_swag
    elif model == "swag":
        model_fn = build_swag
    elif model == "swag_ll":
        model_fn = build_ll_swag
    elif model == "mcd":
        model_fn = build_mcd
    elif model == "bbb":
        model_fn = build_bbb
    elif model == "rank1":
        model_fn = build_rank1
    elif model == "svgd":
        model_fn = build_svgd
    elif model == "ivon":
        model_fn = build_ivon
    elif model == "ll_ivon":
        model_fn = build_ll_ivon
    elif model == "laplace_base":
        model_fn = build_laplace_base
    elif model == "sngp":
        model_fn = build_sngp
    else:
        raise ValueError(f"Unknown model type '{model}'")
    
    ensemble = build_ensemble(model_fn, config, device)
    if config["static_bn"]:
        patch_batchnorm(ensemble, track_running_stats=False)
    return ensemble

def build_ensemble(fn, config, device):
    return DeepEnsemble([fn(config, device) for _ in range(config["members"])])

def build_map(config, device):
    model = get_base_model(nn.Linear(DENSENET_OUT_DIMS, N_CLASSES)).to(device)
    optimizer = MAPOptimizer(model.parameters(), torch.optim.Adam(model.parameters(), **config["base_optimizer"]))
    return model, optimizer

def build_laplace_base(config, device):
    model = densenet121(pretrained=True)
    model.classifier = nn.Linear(DENSENET_OUT_DIMS, N_CLASSES)
    model = nn.Sequential(model) # required for state_dict keys to match
    model.to(device)
    optimizer = MAPOptimizer(model.parameters(), torch.optim.Adam(model.parameters(), **config["base_optimizer"]))
    return model, optimizer

def build_mcd(config, device):
    head = nn.Sequential(
        FixableDropout(config["ll_dropout_p"], freeze_on_eval=False),
        nn.Linear(DENSENET_OUT_DIMS, N_CLASSES)
    )
    model = get_base_model(head).to(device)
    optimizer = MAPOptimizer(model.parameters(), torch.optim.Adam(model.parameters(), **config["base_optimizer"]))
    return model, optimizer

def build_swag(config, device):
    model = get_base_model(nn.Linear(DENSENET_OUT_DIMS, N_CLASSES)).to(device)
    optimizer = SwagOptimizer(model.parameters(), torch.optim.Adam(model.parameters(), **config["base_optimizer"]), **config["swag"])
    return model, optimizer

def build_ll_swag(config, device):
    model = get_base_model(nn.Linear(DENSENET_OUT_DIMS, N_CLASSES)).to(device)
    ll_optimizer = SwagOptimizer(params=model[0].classifier.parameters(), base_optimizer=torch.optim.Adam(model[0].classifier.parameters(), **config["base_optimizer"]), **config["swag"])
    det_optimizer = torch.optim.Adam(model[0].features.parameters(), **config["deterministic_optimizer"])
    optimizer = LastLayerBayesianOptimizer(ll_optimizer, det_optimizer)
    return model, optimizer

def build_bbb(config, device):
    prior = GaussianPrior(0, config["prior_std"])
    model = get_base_model(BBBLinear(DENSENET_OUT_DIMS, N_CLASSES, prior, prior)).to(device)
    optimizer = BBBOptimizer(model.parameters(), torch.optim.Adam(model.parameters(), **config["base_optimizer"]), prior=prior, **config["bbb"])
    return model, optimizer

def build_rank1(config, device):
    prior = GaussianPrior(0, config["prior_std"])
    model = get_base_model(Rank1Linear(DENSENET_OUT_DIMS, N_CLASSES, prior, components=config["rank1"]["components"])).to(device)
    optimizer = BBBOptimizer(model.parameters(), torch.optim.Adam(model.parameters(), **config["base_optimizer"]), prior=prior, **config["rank1"])
    return model, optimizer

def build_svgd(config, device):
    model = get_base_model(nn.Linear(DENSENET_OUT_DIMS, N_CLASSES)).to(device)
    def reset_model():
        reset_model_params(model[0].classifier)
    optimizer = SVGDOptimizer(model.parameters(), reset_model, torch.optim.Adam(model.parameters(), **config["base_optimizer"]), **config["svgd"])
    return model, optimizer

def build_ivon(config, device):
    model = get_base_model(nn.Linear(DENSENET_OUT_DIMS, N_CLASSES)).to(device)
    optimizer = iVONOptimizer(model.parameters(), **config["ivon"])
    return model, optimizer

def build_ll_ivon(config, device):
    model = get_base_model(nn.Linear(DENSENET_OUT_DIMS, N_CLASSES)).to(device)
    ll_optimizer = iVONOptimizer(model[0].classifier.parameters(), **config["ivon"])
    det_optimizer = torch.optim.Adam(model[0].features.parameters(), **config["deterministic_optimizer"])
    optimizer = LastLayerBayesianOptimizer(ll_optimizer, det_optimizer)
    return model, optimizer

def build_sngp(config, device):
    model = get_base_model(nn.Identity())
    spectrally_normalize_module(model[0], norm_bound=config["spectral"]["norm_bound"])
    sngp = SNGPWrapper(
        model,
        nn.LogSoftmax(dim=1),
        outputs=N_CLASSES,
        num_deep_features=DENSENET_OUT_DIMS,
        **config["sngp"]
    ).to(device)
    optimizer = SNGPOptimizer(sngp, torch.optim.Adam(sngp.parameters(), **config["base_optimizer"]))
    return sngp, optimizer

def get_base_model(classification_head):
    model = densenet121(pretrained=True)
    model.classifier = classification_head
    return torch.compile(nn.Sequential(
        model,
        nn.LogSoftmax(dim=1)
    ))
