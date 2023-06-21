import sys
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

from experiments.base import wilds1
from experiments.camelyon.models import get_model
from experiments.camelyon.camelyon import eval_model
from src.algos.ensemble import DeepEnsemble
from src.algos.util import patch_batchnorm

from cw2.cw_data import cw_logging
from cw2 import experiment, cw_error, cluster_work

import wandb

def run(device, name, config, out_path, log, rep):
    if config.get("share_file_system", False):
        torch.multiprocessing.set_sharing_strategy('file_system')
    wandb_mode = "disabled" if config.get("disable_wandb") else "online"
    wandb.init(
        name=f"{config['model']}-{config['members']}-({rep})", 
        project="camelyon17", 
        group=config["model"],
        config=config,
        tags=[config['model'], "bn-static"],
        mode=wandb_mode)

    model = get_model(config["model"], config, device)

    log.info(f"===================== Rep {rep} =====================")
    # path = f"./results/{name}/log/rep_0{rep}{config['model']}_final.tar"
    path = f"./results/MAP/log/rep_0{rep}{config['model']}_final.tar"
    log.info(f"Loading from '{path}'")
    model.load_state_dict(torch.load(path))

    if config["static_bn"]:
        patch_batchnorm(model, track_running_stats=False)

    log.info("Evaluating...")
    eval_time = time.time()
    test_results = eval_model(model, config, device, split="test", subsample=config["test_subsample"])
    log.info(f"Eval time: {time.time() - eval_time}s")

    log.info(f"Test: {test_results}")
    wandb.log({
        "test_results": test_results
    })

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

        torch.manual_seed(rep)

        run(device, config["name"], config["params"], config["_rep_log_path"], l, rep)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(WildsExperiment)
    cw.run()
