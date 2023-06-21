import sys
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

from experiments.base import wilds1
from experiments.civilcomments.models import get_model
from experiments.civilcomments.civil import eval_all_groups
from src.algos.dropout import patch_dropout

from cw2.cw_data import cw_logging
from cw2 import experiment, cw_error, cluster_work

import wandb

import os

def run(device, name, config, out_path, log, rep):
    if config.get("share_file_system", False):
        torch.multiprocessing.set_sharing_strategy('file_system')
    wandb.init(
        name=f"{config['model']}-{rep}-p{config['ll_dropout_p']}", 
        project="civil", 
        group=config["model"],
        config=config,
        tags=["drop-rates"],
        mode=("disabled" if config["disable_wandb"] else "online"))

    model = get_model(config["model"], config, device)
    
    log.info(f"===================== Rep {rep} =====================")
    path = f"./results/MCD/log/rep_0{rep}{config['model']}_final.tar"
    model.load_state_dict(torch.load(path))

    print(f"============= {config['ll_dropout_p']} ==================")
    patch_dropout(model, freeze_on_eval=False, override_p=config["ll_dropout_p"], patch_fixable=True)
    
    eval_time = time.time()
    testloader = wilds1.civil_comments_testloader(config["data_path"], config["batch_size"], subsample=config["test_subsample"])
    test_results = eval_all_groups(model, testloader, config, device)
    log.info({
        "test_results": test_results
    })
    wandb.log({
        "test_results": test_results
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
