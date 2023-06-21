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
from experiments.cifar.models import get_model
from experiments.base.multiclass_classification import _analyze_output
from src.algos.laplace_approx import LaplaceApprox
from src.eval.calibration import ClassificationCalibrationResults
from src.algos.ensemble import DeepEnsemble
from src.wilson import WilsonHMC

from cw2.cw_data import cw_logging
from cw2 import experiment, cw_error, cluster_work

import wandb

def eval_model_on_dataset(model, device, config, loader, hmc):
    outputs = []
    hmc_outputs = []
    targets = []
    with torch.no_grad():
        for input, target in loader:
            input = input.to(device)
            with torch.autocast(device_type="cuda", enabled=config["use_amp"]):
                output = model.predict(lambda m: m(input), samples=config["eval_samples"])
            output = torch.logsumexp(output.float(), dim=0).cpu() - math.log(output.shape[0])
            outputs.append(output)
            targets.append(target)

            hmc_output = hmc.infer(input, config["eval_samples"])
            hmc_output = torch.logsumexp(hmc_output, dim=0).cpu() - torch.log(torch.tensor(hmc_output.shape[0]))
            hmc_outputs.append(hmc_output)
    return torch.cat(outputs), torch.cat(targets), torch.cat(hmc_outputs)

def eval_model(model, config, device, hmc, split="test", subsample=None):
    model.eval()

    if split == "test":
        loader = cifar.cifar10_testloader(config["data_path"], config["eval_batch_size"], True)
    else:
        loader = cifar.cifar10_corrupted_testloader(config["data_path"], split, config["batch_size"], True) # Shuffle as datapoints are sorted by corruption type

    if subsample is not None:
        loader = itertools.islice(loader, subsample)
    
    outputs, targets, hmc_outputs = eval_model_on_dataset(model, device, config, loader, hmc)
    errors, confidences, log_likelihoods, hmc_agreement, hmc_tv = _analyze_output(outputs, targets, hmc_outputs)

    calibration = ClassificationCalibrationResults(config["ece_bins"], errors, confidences)
    return {
        "accuracy": errors.sum() / len(errors),
        "log_likelihood": log_likelihoods.mean(),
        "ece": calibration.ece,
        "sece": calibration.signed_ece,
        "agreement": hmc_agreement.mean(),
        "tv": hmc_tv.mean(),
        "bin_accuracies": calibration.bin_accuracys,
        "bin_confidences": calibration.bin_confidences,
        "bin_counts": calibration.bin_counts
    }

def run(device, config, out_path, log, rep):
    if config.get("share_file_system", False):
        torch.multiprocessing.set_sharing_strategy('file_system')
    wandb_mode = "disabled" if config.get("disable_wandb") else "online"

    wandb_tags = [config["model"]]

    if config["model"] == "mcd":
        wandb_tags.append(f"p-{config['dropout_p']}")

    wandb.init(
        name=f"{config['model']}_{config['members']}-({rep})", 
        project="cifar_10", 
        group=config["model"],
        config=config,
        tags=wandb_tags,
        mode=wandb_mode)

    model = get_model(config["model"], config, device)

    if config.get("use_checkpoint", None):
        model.load_state_dict(torch.load(out_path + f"{config['model']}_chkpt_{config['use_checkpoint']}.pth"))
        start_epoch = config["checkpoint_epochs"]
    else:
        start_epoch = 0

    train_model(model, device, config, log, out_path, start_epoch=start_epoch)
    torch.save(model.state_dict(), out_path + f"{config['model']}_final.tar")

    if "laplace" in config["model"]:
        fit_laplace(model, config, log)

    #model.load_state_dict(torch.load(out_path + f"{config['model']}_final.tar"))

    log.info("Loading HMC samples...")
    hmc = WilsonHMC(config["data_path"] + "Wilson", "cifar10")
    log.info("HMC samples loaded")

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

def train_model(ensemble: DeepEnsemble, device, config, log, out_path, start_epoch=0):
    ensemble.to(device)

    before_all = time.time()

    for model_idx, (model, optimizer) in enumerate(ensemble.models_and_optimizers):
        log.info(f"==================================================")
        log.info(f"Training model {model_idx}")
        log.info(f"==================================================")

        scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])
        optimizer.init_grad_scaler(scaler)
        
        if config["lr_schedule"]:
            scheduler = wilson_scheduler(optimizer.get_base_optimizer(), config["epochs"], config["lr"], None)
        else:
            scheduler = None

        trainloader = cifar.cifar10_trainloader(config["data_path"], config["batch_size"])

        log.info(f"Training on {len(trainloader)} minibatches")
        before = time.time()
        for epoch in range(start_epoch, config["epochs"]):
            ensemble.train()
            epoch_loss = torch.tensor(0.0, device=device)
            for input, target in trainloader:
                input, target = input.to(device), target.to(device)

                def forward():
                    with torch.autocast(device_type="cuda", enabled=config["use_amp"]):
                        return F.nll_loss(model(input), target)

                def backward(loss):
                    scaler.scale(loss).backward()

                loss = optimizer.step(forward, backward, grad_scaler=scaler)
                #print(loss.detach())
                scaler.update()
                epoch_loss += loss.detach()
            optimizer.complete_epoch()
            if scheduler is not None:
                scheduler.step()
            epoch_loss /= len(trainloader)

            if epoch % 20 == 0:
                torch.save(ensemble.state_dict(), out_path + f"{config['model']}_chkpt_{model_idx}_{epoch}.pth")
            log.info(f"Epoch {epoch}: train loss {(epoch_loss):.5f}")
            wandb.log({"train_loss": epoch_loss}, step=(epoch + model_idx * config["epochs"]))

        log.info(f"Final loss: {epoch_loss:.5f}")
        log.info(f"Training time: {time.time() - before}s")
    
    log.info(f"==================================================")
    log.info(f"Finished training")
    log.info(f"Total training time {time.time() - before_all}s")
    log.info(f"==================================================")

def fit_laplace(ensemble, config, log):
    for i in range(len(ensemble.models)):
        trainloader = cifar.cifar10_trainloader(config["data_path"], config["batch_size"])
        class DatasetMock:
            def __len__(self):
                return len(trainloader.dataset)

        class LoaderMock:
            def __init__(self):
                self.dataset = DatasetMock()

            def __iter__(self):
                return iter(map(lambda x: (x[0], x[1]), iter(trainloader)))

        log.info("Fitting laplace...")
        laplace = LaplaceApprox(ensemble.models[i][0], regression=False, out_activation=torch.log, hessian="full")
        laplace.fit(LoaderMock())

        log.info("Optimizing prior prec...")
        laplace.optimize_prior_prec()
        log.info("Done")

        ensemble.models[i] = laplace


def wilson_scheduler(optimizer, pretrain_epochs, lr_init, swag_lr=None):
    def wilson_schedule(epoch):
        t = (epoch) / pretrain_epochs
        lr_ratio = swag_lr / lr_init if swag_lr is not None else 0.01
        if t <= 0.5:
            factor = 1.0
        elif t <= 0.9:
            factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
        else:
            factor = lr_ratio
        return factor
    return torch.optim.lr_scheduler.LambdaLR(optimizer, wilson_schedule)

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
