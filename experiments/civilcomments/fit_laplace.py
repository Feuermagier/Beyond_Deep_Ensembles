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
from src.algos.laplace_approx import LaplaceApprox

from cw2.cw_data import cw_logging
from cw2 import experiment, cw_error, cluster_work

import wandb
import copy

def run(device, config, out_path, log, rep):
    if config.get("share_file_system", False):
        torch.multiprocessing.set_sharing_strategy('file_system')

    ensemble = get_model("laplace_base", config, device)

    single_config = copy.deepcopy(config)
    single_config["members"] = 1
    single_model = get_model("laplace_base", single_config, device)
    
    all_models_and_optimizers = load_all(log, config, device)

    # Train
    laplace_models = []
    for i in range(5):
        laplace_models.append(fit_laplace(all_models_and_optimizers[i][0], config, log, device))

    # Eval
    eval_time = time.time()
    testloader = wilds1.civil_comments_testloader(config["data_path"], 64, subsample=config["test_subsample"])

    for i in range(5):
        eval_rep_single(i, laplace_models, single_model, testloader, config, log, device)
        eval_rep_ensemble(i, laplace_models, ensemble, testloader, config, log, device)

    log.info(f"Eval time: {time.time() - eval_time}s")

def load_all(log, config, device):
    models_and_optimizer = []
    single_config = copy.deepcopy(config)
    single_config["members"] = 1

    for i in range(5):
        single_model = get_model(config["model"], single_config, device)
        path = f"./results/{config['run_name']}/log/rep_0{i}map_final.tar"
        single_model.load_state_dict(torch.load(path))
        models_and_optimizer.append(single_model.models_and_optimizers[0])
    
    return models_and_optimizer

def fit_laplace(model, config, log, device):
    trainloader = wilds1.civil_comments_trainloader(config["data_path"], config["batch_size"], val=False, subsample=config["subsample"])

    class DatasetMock:
        def __len__(self):
            return len(trainloader.dataset)

    class LoaderMock:
        def __init__(self):
            self.dataset = DatasetMock()

        def __iter__(self):
            return iter(map(lambda x: (x[0], x[1]), iter(trainloader)))

    log.info("Fitting laplace...")
    laplace = LaplaceApprox(model, regression=False, out_activation=torch.log, hessian="diag")
    laplace.fit(LoaderMock())

    print("Optimizing prior prec...")
    laplace.optimize_prior_prec()
    log.info("Done")
    return laplace

def eval_rep_single(rep, all_models, single_model, testloader, config, log, device):
    wandb.init(
        name=f"laplace-{rep}", 
        project="civil", 
        group="laplace",
        config=config,
        tags=[],
        mode=("disabled" if config["disable_wandb"] else "online"))

    log.info(f"===================== Rep {rep} (single) =====================")
    single_model.models[0] = all_models[rep]

    test_results = eval_all_groups(single_model, testloader, config, device)
    log.info({
        "test_results": test_results
    })
    wandb.log({
        "test_results": test_results
    })

    wandb.finish()

def eval_rep_ensemble(rep, all_models, ensemble, testloader, config, log, device):
    wandb.init(
        name=f"laplace_4-{rep}", 
        project="civil", 
        group="laplace-4",
        config=config,
        tags=[],
        mode=("disabled" if config["disable_wandb"] else "online"))

    log.info(f"===================== Rep {rep} (ensemble) =====================")
    models = []
    for i in range(5):
        if i == rep:
            continue
        models.append(all_models[i])

    for i in range(4):
        ensemble.models[i] = models[i]

    test_results = eval_all_groups(ensemble, testloader, config, device)
    log.info({
        "test_results": test_results
    })
    wandb.log({
        "test_results": test_results
    })

    wandb.finish()

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

        rep = rep + 0

        torch.manual_seed(rep + config["params"]["seed_offset"])

        run(device, config["params"], config["_rep_log_path"], l, rep)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(WildsExperiment)
    cw.run()
