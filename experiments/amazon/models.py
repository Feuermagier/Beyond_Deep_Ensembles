import torch
import torch.nn as nn

from src.algos.algo import LastLayerBayesianOptimizer
from src.algos.swag import SwagOptimizer
from src.algos.bbb import GaussianPrior, BBBOptimizer
from src.algos.ensemble import DeepEnsemble
from src.algos.pp import MAPOptimizer
from src.algos.ivorn import iVONOptimizer
from src.algos.svgd import SVGDOptimizer
from src.algos.dropout import patch_dropout
from src.algos.kernel.sngp import SNGPWrapper, SNGPOptimizer
from src.algos.kernel.base import spectrally_normalize_module
from src.architectures.bert import BertClassifier
from src.algos.util import reset_model_params

N_CLASSES = 5

def get_model(model, config, device):
    if model == "map":
        model_fn = build_map
    elif model == "swag":
        model_fn = build_swag
    elif model == "ll_swag":
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
    
    return build_ensemble(model_fn, config, device)

def build_ensemble(fn, config, device):
    return DeepEnsemble([fn(config, device) for _ in range(config["members"])])

def build_map(config, device):
    model = nn.Sequential(
        BertClassifier("map", N_CLASSES),
        nn.LogSoftmax(dim=1)
    ).to(device)
    optimizer = MAPOptimizer(get_params(model, config), torch.optim.Adam(get_params(model, config), **config["base_optimizer"]))
    return model, optimizer

def build_laplace_base(config, device):
    model = nn.Sequential(
        BertClassifier("map", N_CLASSES)
    ).to(device)
    optimizer = MAPOptimizer(get_params(model, config), torch.optim.Adam(get_params(model, config), **config["base_optimizer"]))
    return model, optimizer

def build_mcd(config, device):
    model = nn.Sequential(
        BertClassifier("drop", N_CLASSES, drop_p=config["ll_dropout_p"]),
        nn.LogSoftmax(dim=1)
    ).to(device)
    if config.get("last_layer_mcd", False):
        patch_dropout(model[0].classifier, False)
    else:
        patch_dropout(model, False)
        
    optimizer = MAPOptimizer(get_params(model, config), torch.optim.Adam(get_params(model, config), **config["base_optimizer"]))
    return model, optimizer

def build_swag(config, device):
    model = nn.Sequential(
        BertClassifier("map", N_CLASSES),
        nn.LogSoftmax(dim=1)
    ).to(device)
    optimizer = SwagOptimizer(get_params(model, config), torch.optim.Adam(get_params(model, config), **config["base_optimizer"]), **config["swag"])
    return model, optimizer

def build_ll_swag(config, device):
    model = nn.Sequential(
        BertClassifier("map", N_CLASSES),
        nn.LogSoftmax(dim=1)
    ).to(device)
    head_optimizer = SwagOptimizer(get_params(model, config, "head"), torch.optim.Adam(get_params(model, config, "head"), **config["base_optimizer"]), **config["swag"])
    transformer_optimizer = torch.optim.Adam(get_params(model, config, "transformer"), **config["transformer_optimizer"])
    optimizer = LastLayerBayesianOptimizer(head_optimizer, transformer_optimizer)
    return model, optimizer

def build_bbb(config, device):
    prior = GaussianPrior(0, config["prior_std"])
    model = nn.Sequential(
        BertClassifier("bbb", N_CLASSES, prior=prior),
        nn.LogSoftmax(dim=1)
    ).to(device)
    optimizer = BBBOptimizer(get_params(model, config), torch.optim.Adam(get_params(model, config), **config["base_optimizer"]), prior=prior, **config["bbb"])
    return model, optimizer

def build_rank1(config, device):
    prior = GaussianPrior(0, config["prior_std"])
    model = nn.Sequential(
        BertClassifier("rank1", N_CLASSES, prior=prior, components=config["rank1"]["components"]),
        nn.LogSoftmax(dim=1)
    ).to(device)
    optimizer = BBBOptimizer(get_params(model, config), torch.optim.Adam(get_params(model, config), **config["base_optimizer"]), prior=prior, **config["rank1"])
    return model, optimizer

def build_svgd(config, device):
    model = nn.Sequential(
        BertClassifier("map", N_CLASSES),
        nn.LogSoftmax(dim=1)
    ).to(device)
    def reset_model():
        reset_model_params(model[0].classifier)
    optimizer = SVGDOptimizer(get_params(model, config), reset_model, torch.optim.Adam(get_params(model, config), **config["base_optimizer"]), **config["svgd"])
    return model, optimizer

def build_ivon(config, device):
    model = nn.Sequential(
        BertClassifier("map", N_CLASSES),
        nn.LogSoftmax(dim=1)
    ).to(device)
    optimizer = iVONOptimizer(get_params(model, config), **config["ivon"])
    return model, optimizer

def build_ll_ivon(config, device):
    model = nn.Sequential(
        BertClassifier("map", N_CLASSES),
        nn.LogSoftmax(dim=1)
    ).to(device)
    head_optimizer = iVONOptimizer(get_params(model, config, "head"), **config["ivon"])
    transformer_optimizer = torch.optim.Adam(get_params(model, config, "transformer"), **config["transformer_optimizer"])
    optimizer = LastLayerBayesianOptimizer(head_optimizer, transformer_optimizer)
    return model, optimizer

def build_sngp(config, device):
    if config["with_head"]:
        model = nn.Sequential(
            BertClassifier("no_out_projection", N_CLASSES),
        )
        spectrally_normalize_module(model[0].classifier, norm_bound=config["spectral"]["norm_bound"])
    else:
        model = nn.Sequential(
            BertClassifier("no_classifier", N_CLASSES),
        )
    if config["regularize_all"]:
        spectrally_normalize_module(model, norm_bound=config["spectral"]["norm_bound"])
    elif config["with_head"]:
        spectrally_normalize_module(model[0].classifier, norm_bound=config["spectral"]["norm_bound"])

    sngp = SNGPWrapper(model, nn.LogSoftmax(dim=1), outputs=N_CLASSES, num_deep_features=768, **config["sngp"]).to(device)
    optimizer = SNGPOptimizer(sngp, torch.optim.Adam(sngp.parameters(), **config["base_optimizer"]))
    return sngp, optimizer

def get_params(model, config, subset=None):
    if subset is None or subset == "all":
        if config["train_all_layers"]:
            return model.parameters()
        else:
            return model[0].classifier.parameters()
    elif subset == "head":
        return model[0].classifier.parameters()
    elif subset == "transformer":
        return model[0].bert.parameters()
    else:
        raise ValueError(f"Unknown param subset {subset}")
