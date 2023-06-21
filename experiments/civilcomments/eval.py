import sys
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

from experiments.base import wilds1
from experiments.civilcomments.models import get_model
from experiments.civilcomments.civil import eval_all_groups
from experiments.base.multiclass_classification import _analyze_output
from src.eval.calibration import ClassificationCalibrationResults
from src.algos.util import adam
from src.algos.dropout import patch_dropout
import math

import wandb

import os

def test_ensemble():
    config = {
        "batch_size": 64,
        "data_path": "../../data/",
        "use_amp": True,
        "train_all_layers": True,
        "eval_samples": 10,
        "ece_bins": 10,
        "disable_wandb": True,
        "test_subsample": 0,
        "members": 5,
        "base_optimizer": {
            "lr": 0.00001,
            "weight_decay": 0.01
        }
    }

    member_config = {
        "batch_size": 64,
        "data_path": "../../data/",
        "use_amp": True,
        "train_all_layers": True,
        "eval_samples": 10,
        "ece_bins": 10,
        "disable_wandb": True,
        "test_subsample": 10,
        "members": 1,
        "base_optimizer": {
            "lr": 0.00001,
            "weight_decay": 0.01
        }
    }

    print("Creating ensemble...")
    ensemble = get_model("map", config, device)

    for i in range(5):
        print(f"Loading model {i}...")
        member = get_model("map", member_config, device)
        member.load_state_dict(torch.load(f"results/MAP/log/rep_0{i}map_final.tar"))
        ensemble.models[i] = member.models[0]
        ensemble.optimizers[i] = member.optimizers[0]
    
    print("Evaluating...")
    testloader = wilds1.civil_comments_testloader(config["data_path"], config["batch_size"], subsample=config["test_subsample"])
    results = eval_all_groups(ensemble, testloader, config, device)
    print(results)


def test_mcd():
    config = {
        "batch_size": 64,
        "data_path": "../../data/",
        "use_amp": True,
        "train_all_layers": True,
        "eval_samples": 10,
        "ece_bins": 10,
        "disable_wandb": True,
        "test_subsample": 0,
        "members": 1,
        "ll_dropout_p": 0.2,
        "base_optimizer": {
            "lr": 0.00001,
            "weight_decay": 0.01
        }
    }

    print("Creating...")
    model = get_model("mcd", config, device)

    print("Loading...")
    model.load_state_dict(torch.load("results/MCD/log/rep_00mcd_final.tar"))
    patch_dropout(model, override_p=0.05, patch_fixable=True)

    print("Evaluating...")
    testloader = wilds1.civil_comments_testloader(config["data_path"], config["batch_size"], subsample=config["test_subsample"])
    results = eval_all_groups(model, testloader, config, device)
    print(results)

device = torch.device("cuda")
test_ensemble()
