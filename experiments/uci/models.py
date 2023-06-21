import torch
import torch.nn as nn

from src.algos.util import GaussLayer

from src.algos.ensemble import DeepEnsemble
from src.algos.pp import MAPOptimizer
from src.algos.dropout import FixableDropout
from src.algos.bbb import BBBOptimizer, GaussianPrior
from src.algos.bbb_layers import BBBLinear
from src.algos.rank1 import Rank1Linear
from src.algos.swag import SwagOptimizer
from src.algos.svgd import SVGDOptimizer
from src.algos.util import reset_model_params
from src.algos.ivorn import iVONOptimizer

def get_var_optimizer(model: torch.nn.Sequential, config):
    '''
        Assumes that the GaussLayer is the last layer of the model
    '''
    return torch.optim.SGD(model[-1].parameters(), **config["variance_optimizer"])

def get_model(model_name, config, device):
    if model_name == "map":
        return build_map(config, device)
    if model_name == "laplace":
        return build_map(config, device) # Same model as MAP
    elif model_name == "mcd":
        return build_mcd(config, device)
    elif model_name == "swag":
        return build_swag(config, device)
    elif model_name == "bbb" or model_name == "bbb_fixed_kl":
        return build_bbb(config, device)
    elif model_name == "rank1":
        return build_rank1(config, device)
    elif model_name == "svgd":
        return build_svgd(config, device)
    elif model_name == "ivon":
        return build_ivon(config, device)
    else:
        raise ValueError(f"Unknown model {model_name}")

def _build_map(config, device):
    model = nn.Sequential(
        nn.Sequential(
            nn.Linear(config["in_dim"], 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        ),
        GaussLayer(torch.tensor(config["std_init"]), config["learn_var"])
    ).to(device)
    optimizer = MAPOptimizer(model[0].parameters(), torch.optim.Adam(model[0].parameters(), **config["optimizer"]["base"]))
    return model, optimizer

def build_map(config, device):
    return DeepEnsemble([_build_map(config, device) for _ in range(config["members"])])

def _build_mcd(config, device):
    model = nn.Sequential(
        nn.Sequential(
            nn.Linear(config["in_dim"], 50),
            FixableDropout(config["dropout_p"]),
            nn.ReLU(),
            nn.Linear(50, 1)
        ),
        GaussLayer(torch.tensor(config["std_init"]), config["learn_var"])
    ).to(device)
    optimizer = MAPOptimizer(model[0].parameters(), torch.optim.Adam(model[0].parameters(), **config["optimizer"]["base"]))
    return model, optimizer

def build_mcd(config, device):
    return DeepEnsemble([_build_mcd(config, device) for _ in range(config["members"])])

def _build_swag(config, device):
    model = nn.Sequential(
        nn.Sequential(
            nn.Linear(config["in_dim"], 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        ),
        GaussLayer(torch.tensor(config["std_init"]), config["learn_var"])
    ).to(device)
    optimizer = SwagOptimizer(model[0].parameters(), torch.optim.Adam(model[0].parameters(), **config["optimizer"]["base"]), **config["optimizer"]["swag"])
    return model, optimizer

def build_swag(config, device):
    return DeepEnsemble([_build_swag(config, device) for _ in range(config["members"])])

def _build_bbb(config, device):
    prior = GaussianPrior(0, config["prior_std"])
    model = nn.Sequential(
        nn.Sequential(
            BBBLinear(config["in_dim"], 50, prior, prior),
            nn.ReLU(),
            BBBLinear(50, 1, prior, prior)
        ),
        GaussLayer(torch.tensor(config["std_init"]), config["learn_var"])
    ).to(device)
    optimizer = BBBOptimizer(model[0].parameters(), torch.optim.Adam(model[0].parameters(), **config["optimizer"]["base"]), prior, **config["optimizer"]["bbb"])
    return model, optimizer

def build_bbb(config, device):
    return DeepEnsemble([_build_bbb(config, device) for _ in range(config["members"])])

def _build_rank1(config, device):
    prior = GaussianPrior(0, config["prior_std"])
    model = nn.Sequential(
        nn.Sequential(
            Rank1Linear(config["in_dim"], 50, prior, prior),
            nn.ReLU(),
            Rank1Linear(50, 1, prior, prior)
        ),
        GaussLayer(torch.tensor(config["std_init"]), config["learn_var"])
    ).to(device)
    optimizer = BBBOptimizer(model[0].parameters(), torch.optim.Adam(model[0].parameters(), **config["optimizer"]["base"]), prior, **config["optimizer"]["rank1"])
    return model, optimizer

def build_rank1(config, device):
    return DeepEnsemble([_build_rank1(config, device) for _ in range(config["members"])])

def _build_svgd(config, device):
    model = nn.Sequential(
        nn.Sequential(
            nn.Linear(config["in_dim"], 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        ),
        GaussLayer(torch.tensor(config["std_init"]), config["learn_var"])
    ).to(device)
    
    def reset_params():
        reset_model_params(model)

    optimizer = SVGDOptimizer(model[0].parameters(), reset_params, torch.optim.Adam(model[0].parameters(), **config["optimizer"]["base"]), **config["optimizer"]["svgd"])
    return model, optimizer

def build_svgd(config, device):
    return DeepEnsemble([_build_svgd(config, device) for _ in range(config["members"])])

def _build_ivon(config, device):
    model = nn.Sequential(
        nn.Sequential(
            nn.Linear(config["in_dim"], 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        ),
        GaussLayer(torch.tensor(config["std_init"]), config["learn_var"])
    ).to(device)
    optimizer = iVONOptimizer(model[0].parameters(), **config["optimizer"]["ivon"])
    return model, optimizer

def build_ivon(config, device):
    return DeepEnsemble([_build_ivon(config, device) for _ in range(config["members"])])

