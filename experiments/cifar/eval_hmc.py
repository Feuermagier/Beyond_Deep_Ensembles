import sys
sys.path.append("../../")
sys.path.append("../../google-bnn-hmc/")

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import math
import itertools

from experiments.base import cifar
from experiments.cifar.cifar import eval_model
from experiments.base.multiclass_classification import _analyze_output
from src.eval.calibration import ClassificationCalibrationResults
from src.algos.ensemble import DeepEnsemble
from src.wilson import WilsonHMC

from cw2.cw_data import cw_logging
from cw2 import experiment, cw_error, cluster_work

import wandb

def run(device, config, out_path, log, rep):
    if config.get("share_file_system", False):
        torch.multiprocessing.set_sharing_strategy('file_system')
    wandb_mode = "disabled" if config.get("disable_wandb") else "online"

    wandb.init(
        name=f"hmc-({rep})", 
        project="cifar_10", 
        group="hmc",
        config=config,
        tags=["hmc"],
        mode=wandb_mode)

    log.info("Loading HMC samples...")
    hmc = WilsonHMC(config["data_path"] + "Wilson", "cifar10")
    log.info("HMC samples loaded")

    class HMCModelWrapper:
        def predict(self, model_fn, samples):
            return model_fn(lambda input: hmc.infer(input, samples))

        def eval(self):
            hmc.eval()
        
        def train(self):
            hmc.train()

    model = HMCModelWrapper()

    eval_time = time.time()

    test_results = eval_model(model, config, device, hmc, split="test", subsample=None)
    log.info(f"Test: {test_results}")

    c1_results = eval_model(model, config, device, hmc, split=0, subsample=10)
    log.info(f"Corrupted (1): {c1_results}")

    c3_results = eval_model(model, config, device, hmc, split=2, subsample=10)
    log.info(f"Corrupted (3): {c3_results}")

    c5_results = eval_model(model, config, device, hmc, split=4, subsample=10)
    log.info(f"Corrupted (5): {c5_results}")

    wandb.log({
        "test_results": test_results,
        "c1_results": c1_results,
        "c3_results": c3_results,
        "c5_results": c5_results
    })
    log.info(f"Eval time: {time.time() - eval_time}s")

####################### CW2 #####################################

class CifarExperiment(experiment.AbstractExperiment):
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

        torch.manual_seed(rep + config["params"]["seed_offset"])

        run(device, config["params"], config["_rep_log_path"], l, rep)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(CifarExperiment)
    cw.run()
