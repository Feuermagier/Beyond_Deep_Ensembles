import sys
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

from experiments.base import wilds1
from experiments.camelyon.models import get_model
from experiments.camelyon.camelyon import eval_model
from experiments.base.multiclass_classification import _analyze_output
from src.eval.calibration import ClassificationCalibrationResults
from src.algos.util import adam
import math

import wandb

import os

from experiments.base import wilds1

if torch.cuda.is_available():
    print("Using CUDA")
    device = torch.device("cuda")
else:
    print("Using the CPU")
    device = torch.device("cpu")

testloader = wilds1.camelyon_loader(wilds1.camelyon_split("../../data/", "test"), 1024)
print(len(testloader))
test_means = []
for input, output, _ in testloader:
    print("Batch")
    test_means.append(input.to(device).mean(dim=1))

print(torch.cat(test_means).mean())

trainloader = wilds1.camelyon_loader(wilds1.camelyon_split("../../data/", "train"), 1024)
print(len(trainloader))
train_means = []
for input, output, _ in trainloader:
    print("Batch")
    train_means.append(input.to(device).mean(dim=1))

print(torch.cat(train_means).mean())

def test_eval():
    config = {
        "data_path": "../../data/",
        "batch_size": 32,
        "use_amp": False,
        "eval_samples": 10,
        "ece_bins": 10,
        "train_on_val": False,
        "disable_wandb": False,
        "eval_while_train": True,
        "subsample": 0,
        "test_subsample": 0,
        "members": 1,
        "optimizer": {
            "base": {
                "lr": 0.001,
                "momentum": 0.9,
                "weight_decay": 0.1
            },
        
            "swag": {
                "update_interval": 630,
                "start_epoch": 3,
                "deviation_samples": 30
            }
        }
    }

    model = get_model("swag", config, device)
    print("Loading...")
    model.load_state_dict(torch.load("results/SWAG/log/rep_00swag_final.tar"))

    print("Evaluating...")
    results = eval_model(model, config, device, split="test", subsample=10)
    print(results)
