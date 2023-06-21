import sys
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.nn.functional as F

import time
import math
import copy

from experiments.base import wilds1
from experiments.poverty.models import get_model
from experiments.poverty.poverty import eval_model_id_ood
from experiments.base.multiclass_classification import _analyze_output
from src.eval.calibration import ClassificationCalibrationResults
from src.algos.ensemble import DeepEnsemble
from src.algos.laplace_approx import LaplaceApprox
from src.algos.util import GaussLayer

from cw2.cw_data import cw_logging
from cw2 import experiment, cw_error, cluster_work

import wandb

def run(device, config, out_path, log, rep):
    assert not config["learn_var"]

    if config.get("share_file_system", False):
        torch.multiprocessing.set_sharing_strategy('file_system')
    wandb_mode = "disabled" if config.get("disable_wandb") else "online"

    wandb_tags = [config["model"]]

    wandb.init(
        name=f"laplace-{config['members']}-({config['fold']})", 
        project="poverty", 
        group="laplace",
        config=config,
        tags=wandb_tags,
        mode=wandb_mode)

    model = get_model("laplace_base", device, config)
    model.load_state_dict(torch.load(f"results/{config['base_model']}/{config['base_model']}__f{config['fold']}/log/rep_0{rep}map_final.tar"), strict=False)

    single_config = copy.deepcopy(config)
    single_config["members"] = 1
    for i in range(len(model.models_and_optimizers)):
        trainloader = wilds1.poverty_loader(wilds1.poverty_split(config["data_path"], "train", config["fold"]), config["batch_size"], subsample=config["subsample"])

        class DatasetMock:
            def __len__(self):
                return len(trainloader.dataset)

        class LoaderMock:
            def __init__(self):
                self.dataset = DatasetMock()

            def __iter__(self):
                return iter(map(lambda x: (x[0], x[1]), iter(trainloader)))

        log.info("Fitting laplace...")
        laplace = LaplaceApprox(model.models[i], regression=True, out_activation=None, hessian="kron")
        laplace.fit(LoaderMock())

        print("Optimizing prior prec...")
        laplace.optimize_prior_prec()

        laplace.out_activation = GaussLayer(torch.tensor(config["init_std"]), config["learn_var"]).to(device)

        model.models[i] = laplace

    torch.save(model.state_dict(), out_path + f"laplace_final.tar")
    log.info("Done")

    eval_time = time.time()
    result = eval_model_id_ood(model, device, config, log)
    log.info(result)
    log.info(f"Eval time: {time.time() - eval_time}s")
    wandb.log(result)

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

        torch.manual_seed(rep + config["params"]["seed_offset"])

        run(device, config["params"], config["_rep_log_path"], l, rep)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(WildsExperiment)
    cw.run()
