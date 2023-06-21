import sys
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb
import time
import itertools

from cw2.cw_data import cw_logging
from cw2 import experiment, cw_error, cluster_work

import wandb

from src.algos.ensemble import DeepEnsemble
from src.eval.regresssion import RegressionResults
from src.algos.util import nll_loss
from src.algos.laplace_approx import LaplaceApprox
from src.log_mock import VoidLog

from experiments.uci2.models import get_model, get_var_optimizer
from experiments.uci2.data import UCIDataset, get_loader

def eval_model(model, config, device, split, gap_split):
    torch.manual_seed(42)

    dataset = get_dataset(config)    
    testloader = get_loader(dataset.get_dataset(split, gap_split, device=device), batch_size=config["batch_size"], shuffle=False)

    model.eval()
    with torch.no_grad():
        outputs = []
        targets = []

        for input, target in testloader:
            input = input.to(device)
            output = model.predict(lambda m: m(input), samples=config["eval_samples"])
            outputs.append(output.cpu())
            targets.append(target.cpu())

        outputs = torch.cat(outputs, dim=1)
        targets = torch.cat(targets)
        results = RegressionResults(outputs, targets, target_mean=dataset.y_mean, target_std=dataset.y_std)
        
        return results

def run(device, config, out_path, log):
    # torch.multiprocessing.set_sharing_strategy('file_system')
    wandb_mode = "disabled" if config.get("disable_wandb") else "online"
    wandb.init(
        name=f"{config['model']}-{config['members']}", 
        project=f"uci-{config['dataset']}", 
        group=config["model"],
        config=config,
        tags=[config['model']],
        mode=wandb_mode)

    all_results = {}

    if config["plain"]:
        plain_results = test_gap_split(None, config, log, device, repetitions=config["standard_split_reps"])
        log.info(f"Plain: {plain_results}")
        all_results["plain"] = plain_results

    if config["gap"]:
        gap_results = []
        for gap_split in range(config["in_dim"]):
            gap_results.append({
                "gap_split": gap_split,
                "result": test_gap_split(gap_split, config, log, device, 1)[0]
            })
        log.info(f"Gap: {gap_results}")
        all_results["gap_results"] = gap_results

    log.info(all_results)
    wandb.log(all_results)

    wandb.finish()

def train_model(ensemble: DeepEnsemble, config, device, log, split, gap_split):
    before_all = time.time()

    for model_idx, (model, optimizer) in enumerate(ensemble.models_and_optimizers):
        log.info(f"==================================================")
        log.info(f"Training model {model_idx}")
        log.info(f"==================================================")

        if config["learn_var"]:
            var_optimizer = get_var_optimizer(model, config)
        else:
            var_optimizer = None

        trainloader = get_loader(get_dataset(config).get_dataset(split, gap_split, device), config["batch_size"], True)

        log.info(f"Training on {len(trainloader)} minibatches")
        before = time.time()
        for epoch in range(config["epochs"]):
            ensemble.train()
            epoch_loss = torch.tensor(0.0, device=device)
            for input, target in trainloader:
                input, target = input.to(device), target.to(device)

                if var_optimizer is not None:
                    var_optimizer.zero_grad()

                def forward():
                    return nll_loss(model(input), target)

                def backward(loss):
                    loss.backward()

                loss = optimizer.step(forward, backward)
                
                if var_optimizer is not None:
                    var_optimizer.step()

                epoch_loss += loss.detach()
            optimizer.complete_epoch()

            if epoch % 10 == 0:
                log.info(f"Epoch {epoch}: train loss {(epoch_loss / len(trainloader)):.5f}")
        log.info(f"Final loss: {(epoch_loss / len(trainloader)):.5f}")
        log.info(f"Training time: {time.time() - before}s")
    
    # Laplace
    if "laplace" in config["model"]:
        log.info(f"--------------------------------------------------")
        log.info("Fitting laplace")
        for i in range(len(ensemble.models)):
            laplace = LaplaceApprox(ensemble.models[i][0], regression=True, hessian=config["ll_hessian"])
            laplace.fit(get_loader(get_dataset(config).get_dataset("train", gap_split, device), config["batch_size"], True))
            laplace.optimize_prior_prec()
            laplace.out_activation = ensemble.models[i][1]
            ensemble.models[i] = laplace

    log.info(f"==================================================")
    log.info(f"Finished training")
    log.info(f"Total training time {time.time() - before_all}s")
    log.info(f"==================================================")

def get_dataset(config):
    return UCIDataset(config["dataset"], normalize=config["normalize"],val_percentage=config["val_percentage"])

def run_trial(config, gap_split, device):
    model = get_model(config["model"], config, device)
    train_model(model, config, device, VoidLog(), "val_train", gap_split)
    return eval_model(model, config, device, "val_test", gap_split)

def tune_hyperparams(config, hp_ranges, use_hyperparams, gap_split, device, log):
    best_ll = -1000000
    best_config = None

    values = [[(key, value) for value in hp_ranges[key]] for key in hp_ranges.keys()]
    for combination in itertools.product(*values):
        hp_conf = {key:value for key, value in combination}
        log.info(f"Testing {hp_conf}")
        use_hyperparams(hp_conf, config)
        ll = run_trial(config, gap_split, device).average_log_likelihood
        if ll > best_ll:
            best_ll = ll
            best_config = hp_conf

    log.info(f"Selected {best_config} with ll {best_ll}")
    use_hyperparams(best_config, config)

def map_hyperparams():
    ranges = {
        "epochs": [40, 100], 
        "lr": [0.01, 0.001], 
        "weight_decay": [1e-4, 1e-5]
    }

    def use_hyperparams(params, config):
        config["epochs"] = params["epochs"]
        config["optimizer"]["base"]["lr"] = params["lr"]
        config["optimizer"]["base"]["weight_decay"] = params["weight_decay"]
    return ranges, use_hyperparams

def mcd_hyperparams():
    ranges = {
        "epochs": [40, 100], 
        "lr": [0.01, 0.001], 
        "weight_decay": [1e-4, 1e-5],
        "drop_rate": [0.2, 0.1, 0.05]
    }

    def use_hyperparams(params, config):
        config["epochs"] = params["epochs"]
        config["optimizer"]["base"]["lr"] = params["lr"]
        config["optimizer"]["base"]["weight_decay"] = params["weight_decay"]
        config["dropout_p"] = params["drop_rate"]
    return ranges, use_hyperparams

def swag_hyperparams():
    ranges = {
        "epochs": [60, 100, 150],
        "lr": [0.01, 0.001], 
        "weight_decay": [1e-4, 1e-5],
        "start": [0.5, 0.75, 0.9]
    }

    def use_hyperparams(params, config):
        config["epochs"] = params["epochs"]
        config["optimizer"]["base"]["lr"] = params["lr"]
        config["optimizer"]["base"]["weight_decay"] = params["weight_decay"]
        swag_epochs = int(params["start"] * params["epochs"])
        config["optimizer"]["swag"]["start_epoch"] = swag_epochs
        config["optimizer"]["swag"]["update_interval"] = int(config["train_set_size"] * swag_epochs / 30)
        config["optimizer"]["swag"]["deviation_samples"] = 30
    return ranges, use_hyperparams

def bbb_hyperparams():
    ranges = {
        #"epochs": [40, 100, 200],
        "epochs": [200],
        "lr": [0.01, 0.001],
        "prior_std": [0.1, 1.0, 10.0],
        #"kl_rescaling": [0.2, 0.5, 1.0]
        "kl_rescaling": [0.2, 0.5]
    }

    def use_hyperparams(params, config):
        config["epochs"] = params["epochs"]
        config["prior_std"] = params["prior_std"]
        config["optimizer"]["base"]["lr"] = params["lr"]
        config["optimizer"]["bbb"]["kl_rescaling"] = params["kl_rescaling"]
        config["optimizer"]["bbb"]["dataset_size"] = config["train_set_size"]

    return ranges, use_hyperparams

def bbb_fixed_kl_hyperparams():
    ranges = {
        "epochs": [200], # BBB chose *always* 200 epochs
        "lr": [0.01, 0.001],
        "prior_std": [0.1, 1.0, 10.0]
    }

    def use_hyperparams(params, config):
        config["epochs"] = params["epochs"]
        config["prior_std"] = params["prior_std"]
        config["optimizer"]["base"]["lr"] = params["lr"]
        config["optimizer"]["bbb"]["dataset_size"] = config["train_set_size"]

    return ranges, use_hyperparams

def rank1_hyperparams():
    ranges = {
        "epochs": [100, 200],
        "lr": [0.01, 0.001],
        "l2_scale": [1e-4, 1e-5]
    }

    def use_hyperparams(params, config):
        config["epochs"] = params["epochs"]
        config["optimizer"]["base"]["lr"] = params["lr"]
        config["optimizer"]["rank1"]["l2_scale"] = params["l2_scale"]
        config["optimizer"]["rank1"]["dataset_size"] = config["train_set_size"]

    return ranges, use_hyperparams

def svgd_hyperparams():
    ranges = {
        "epochs": [40, 100],
        "lr": [0.01, 0.001],
        "l2_reg": [1e-4, 1e-5]
    }

    def use_hyperparams(params, config):
        config["epochs"] = params["epochs"]
        config["optimizer"]["base"]["lr"] = params["lr"]
        config["optimizer"]["svgd"]["l2_reg"] = params["l2_reg"]
        config["optimizer"]["svgd"]["dataset_size"] = config["train_set_size"]

    return ranges, use_hyperparams

def ivon_hyperparams():
    ranges = {
        "epochs": [40, 100, 200],
        "lr": [0.01],
        "prior_prec": [10.0, 100.0, 200.0]
    }

    def use_hyperparams(params, config):
        config["epochs"] = params["epochs"]
        config["optimizer"]["ivon"]["lr"] = params["lr"]
        config["optimizer"]["ivon"]["prior_prec"] = params["prior_prec"]
        config["optimizer"]["ivon"]["dataset_size"] = config["train_set_size"]

    return ranges, use_hyperparams

def tune(gap_split, config, log, device):
    config["train_set_size"] = len(get_dataset(config).get_dataset("val_train", gap_split, device))

    if config["model"] == "map":
        ranges, use = map_hyperparams()
    elif config["model"] == "laplace":
        ranges, use = map_hyperparams() # Same as MAP
    elif config["model"] == "mcd":
        ranges, use = mcd_hyperparams()
    elif config["model"] == "swag":
        ranges, use = swag_hyperparams()
    elif config["model"] == "bbb":
        ranges, use = bbb_hyperparams()
    elif config["model"] == "bbb_fixed_kl":
        ranges, use = bbb_fixed_kl_hyperparams()
    elif config["model"] == "rank1":
        ranges, use = rank1_hyperparams()
    elif config["model"] == "svgd":
        ranges, use = svgd_hyperparams()
    elif config["model"] == "ivon":
        ranges, use = ivon_hyperparams()
    else:
        raise ValueError(f"Unknown model '{config['model']}'")
    
    tune_hyperparams(config, ranges, use, gap_split, device, log)


def test_gap_split(gap_split, config, log, device, repetitions=1):
    log.info(f"================= Gap split {gap_split} ================")
    if config["hpo"]:
        tune(gap_split, config, log, device)
    else:
        log.info("Skipping HPO")

    results = []
    for rep in range(repetitions):
        torch.manual_seed(rep + (gap_split * repetitions if gap_split is not None else 0))
        model = get_model(config["model"], config, device)
        train_model(model, config,device, log, split="train", gap_split=gap_split)

        result = eval_model(model, config, device, split="test", gap_split=gap_split)
        results.append({
            "avg ll": result.average_log_likelihood,
            "avg lml": result.average_lml,
            "mse": result.mse,
            "qce": result.qce,
            "sqce": result.sqce
        })
    return results

####################### CW2 #####################################

class UCIExperiment(experiment.AbstractExperiment):
    def initialize(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        pass

    def run(self, config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        l = cw_logging.getLogger()
        l.info(config["params"])
        if torch.cuda.is_available() and config["params"]["allow_cuda"]:
            l.info("Using the GPU")
            device = torch.device("cuda")
        else:
            l.info("Using the CPU")
            device = torch.device("cpu")

        torch.manual_seed(rep + 1)

        run(device, config["params"], config["_rep_log_path"], l)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(UCIExperiment)
    cw.run()
