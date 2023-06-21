import torch
import torch.nn as nn

from src.architectures.densenet import DenseNet, ClassificationHead

from src.algos.ensemble import DeepEnsemble
from src.algos.pp import MAPOptimizer
from src.algos.swag import SwagOptimizer
from src.algos.bbb import BBBOptimizer, GaussianPrior
from src.algos.ivorn import iVONOptimizer
from src.algos.svgd import SVGDOptimizer
from src.algos.util import reset_model_params

def get_model(model, config, device):
    if model == "map":
        return build_map(config, device)
    elif model == "mcd":
        return build_mcd(config, device)
    elif model == "swag":
        return build_swag(config, device)
    elif model == "bbb":
        return build_bbb(config, device)
    elif model == "rank1":
        return build_rank1(config, device)
    elif model == "ivon":
        return build_ivon(config, device)
    elif model == "svgd":
        return build_svgd(config, device)
    else:
        raise ValueError(f"Unknown model {model}")

def _build_map(config, device):
    net_config = {
        "linear": {
            "type": "plain"
        }, 
        "conv": {
            "type": "plain"
        }
    }
    densenet = DenseNet(32, (6, 12, 24, 16), 3, 64, 4, net_config)
    model = nn.Sequential(
        densenet,
        ClassificationHead(densenet.out_features, 2),
        nn.LogSoftmax(dim=1)
    ).to(device)
    optimizer = MAPOptimizer(model.parameters(), torch.optim.SGD(model.parameters(), **config["optimizer"]["base"]))
    return model, optimizer

def build_map(config, device):
    return DeepEnsemble([_build_map(config, device) for _ in range(config["members"])])

def _build_mcd(config, device):
    net_config = {
        "linear": {
            "type": "plain"
        }, 
        "conv": {
            "type": "plain"
        },
        "dropout_p": config["dropout_p"]
    }
    densenet = DenseNet(32, (6, 12, 24, 16), 3, 64, 4, net_config)
    model = nn.Sequential(
        densenet,
        # Don't add dropout here as the last layer of the DenseNet already includes dropout
        ClassificationHead(densenet.out_features, 2, net_config),
        nn.LogSoftmax(dim=1)
    ).to(device)
    optimizer = MAPOptimizer(model.parameters(), torch.optim.SGD(model.parameters(), **config["optimizer"]["base"]))
    return model, optimizer

def build_mcd(config, device):
    return DeepEnsemble([_build_mcd(config, device) for _ in range(config["members"])])

def _build_swag(config, device):
    net_config = {
        "linear": {
            "type": "plain"
        }, 
        "conv": {
            "type": "plain"
        }
    }
    densenet = DenseNet(32, (6, 12, 24, 16), 3, 64, 4, net_config)
    model = nn.Sequential(
        densenet,
        ClassificationHead(densenet.out_features, 2, net_config),
        nn.LogSoftmax(dim=1)
    ).to(device)
    optimizer = SwagOptimizer(model.parameters(), torch.optim.SGD(model.parameters(), **config["optimizer"]["base"]), **config["optimizer"]["swag"])
    return model, optimizer

def build_swag(config, device):
    return DeepEnsemble([_build_swag(config, device) for _ in range(config["members"])])

def _build_bbb(config, device):
    prior = GaussianPrior(0, config["prior_std"])
    net_config = {
        "linear": {
            "type": "variational",
            "prior": prior
        },
        "conv": {
            "type": "variational",
            "prior": prior
        }
    }
    densenet = DenseNet(32, (6, 12, 24, 16), 3, 64, 4, net_config)
    model = nn.Sequential(
        densenet,
        ClassificationHead(densenet.out_features, 2, net_config),
        nn.LogSoftmax(dim=1)
    ).to(device)
    optimizer = BBBOptimizer(model.parameters(), torch.optim.SGD(model.parameters(), **config["optimizer"]["base"]), prior=prior, **config["optimizer"]["bbb"])
    return model, optimizer

def build_bbb(config, device):
    return DeepEnsemble([_build_bbb(config, device) for _ in range(config["members"])])

def _build_rank1(config, device):
    prior = GaussianPrior(0, config["prior_std"])
    net_config = {
        "linear": {
            "type": "rank1",
            "prior": prior,
            "components": config["optimizer"]["rank1"]["components"]
        },
        "conv": {
            "type": "rank1",
            "prior": prior,
            "components": config["optimizer"]["rank1"]["components"]
        }
    }
    densenet = DenseNet(32, (6, 12, 24, 16), 3, 64, 4, net_config)
    model = nn.Sequential(
        densenet,
        ClassificationHead(densenet.out_features, 2, net_config),
        nn.LogSoftmax(dim=1)
    ).to(device)
    optimizer = BBBOptimizer(model.parameters(), torch.optim.SGD(model.parameters(), **config["optimizer"]["base"]), prior=prior, **config["optimizer"]["rank1"])
    return model, optimizer

def build_rank1(config, device):
    return DeepEnsemble([_build_rank1(config, device) for _ in range(config["members"])])

def _build_ivon(config, device):
    net_config = {
        "linear": {
            "type": "plain"
        }, 
        "conv": {
            "type": "plain"
        }
    }
    densenet = DenseNet(32, (6, 12, 24, 16), 3, 64, 4, net_config)
    model = nn.Sequential(
        densenet,
        ClassificationHead(densenet.out_features, 2),
        nn.LogSoftmax(dim=1)
    ).to(device)
    optimizer = iVONOptimizer(model.parameters(), **config["ivon"])
    return model, optimizer

def build_ivon(config, device):
    return DeepEnsemble([_build_ivon(config, device) for _ in range(config["members"])])

def _build_svgd(config, device):
    net_config = {
        "linear": {
            "type": "plain"
        }, 
        "conv": {
            "type": "plain"
        }
    }
    densenet = DenseNet(32, (6, 12, 24, 16), 3, 64, 4, net_config)
    model = nn.Sequential(
        densenet,
        ClassificationHead(densenet.out_features, 2),
        nn.LogSoftmax(dim=1)
    ).to(device)
    def reset_model():
        reset_model_params(model)
    optimizer = SVGDOptimizer(model.parameters(), reset_model, torch.optim.SGD(model.parameters(), **config["optimizer"]["base"]), **config["optimizer"]["svgd"])
    return model, optimizer

def build_svgd(config, device):
    return DeepEnsemble([_build_svgd(config, device) for _ in range(config["members"])])
