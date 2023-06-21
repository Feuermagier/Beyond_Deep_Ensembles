import sys
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import copy

from experiments.base import wilds1
from experiments.iwildcam.models import get_model
from experiments.iwildcam.iwildcam import eval_model
from src.algos.ensemble import DeepEnsemble

from cw2.cw_data import cw_logging
from cw2 import experiment, cw_error, cluster_work

import wandb

import os

def run(device, name, config, out_path, log, rep):
    if config.get("share_file_system", False):
        torch.multiprocessing.set_sharing_strategy('file_system')
    wandb.init(
        name=f"{config['model']}_{config['members']}-({rep})", 
        project="iwildcam",
        group=f"{config['model']}-{config['members']}",
        config=config,
        tags=[f"{config['model']}-{config['members']}", "ensembles"],
        mode=("disabled" if config["disable_wandb"] else "online"))

    models_and_optimizer = []
    
    log.info(f"===================== Rep {rep} =====================")
    single_config = copy.deepcopy(config)
    single_config["members"] = 1

    if config.get("split_directories", True):
        for i in range(3):
            if i == rep:
                continue
            single_model = get_model(config["model"], single_config, device)
            path = f"./results/{config['run_name']}/log/rep_0{i}{config['model']}_final.tar"
            single_model.load_state_dict(torch.load(path))
            models_and_optimizer.append(single_model.models_and_optimizers[0])
        
        for i in range(3):
            single_model = get_model(config["model"], single_config, device)
            path = f"./results/{config['run_name']}_more/log/rep_0{i}{config['model']}_final.tar"
            single_model.load_state_dict(torch.load(path))
            models_and_optimizer.append(single_model.models_and_optimizers[0])
    else:
        for i in range(6):
            if i == rep:
                continue
            single_model = get_model(config["model"], single_config, device)
            path = f"./results/{config['run_name']}/log/rep_0{i}{config['model']}_final.tar"
            single_model.load_state_dict(torch.load(path))
            models_and_optimizer.append(single_model.models_and_optimizers[0])
    
    model = DeepEnsemble(models_and_optimizer)

    eval_time = time.time()

    test_results = eval_model(model, config, device, split="test", subsample=config["test_subsample"])
    log.info(f"Test: {test_results}")

    val_results = eval_model(model, config, device, split="test", subsample=config["test_subsample"])
    log.info(f"OOD Validation: {val_results}")

    id_val_results = eval_model(model, config, device, split="test", subsample=config["test_subsample"])
    log.info(f"ID Validation: {id_val_results}")

    wandb.log({
        "test_results": test_results,
        "val_results": val_results,
        "id_val_results": id_val_results
    })
    log.info(f"Eval time: {time.time() - eval_time}s")


####################### CW2 #####################################

class WildsExperiment(experiment.AbstractExperiment):
    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        l = cw_logging.getLogger()
        l.info(config["params"])
        if torch.cuda.is_available():
            l.info("Using the GPU")
            device = torch.device("cuda")
        else:
            l.info("Using the CPU")
            device = torch.device("cpu")

        torch.manual_seed(rep + 1)

        run(device, config["name"], config["params"], config["_rep_log_path"], l, rep)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(WildsExperiment)
    cw.run()
