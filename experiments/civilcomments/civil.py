import sys
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from experiments.base import wilds1
from experiments.civilcomments.models import get_model
from experiments.base.multiclass_classification import _analyze_output
from src.eval.calibration import ClassificationCalibrationResults
import math

from cw2.cw_data import cw_logging
from cw2 import experiment, cw_error, cluster_work

import wandb

import os

GROUP_INDICES = {
    "male": 0,
    "female": 1,
    "lgbtq": 2,
    "christian": 3,
    "muslim": 4,
    "other_religion": 5,
    "black": 6,
    "white": 7,
}

def get_group(metadata, targets, toxic, group):
    group = metadata[:,GROUP_INDICES[group]].nonzero()
    toxics = targets.nonzero() if toxic else (1 - targets).nonzero()
    uniques, counts = torch.cat((group, toxics)).unique(return_counts=True)
    return uniques[counts > 1]

def eval_group(group, output, metadata, targets, errors, confidences, lls, config):
    accuracy = errors[group].sum() / len(group)
    calibration = ClassificationCalibrationResults(config["ece_bins"], errors[group], confidences[group])
    return {
        "accuracy": accuracy,
        "log_likelihood": lls[group].mean(),
        "ece": calibration.ece,
        "sece": calibration.signed_ece,
        "bin_accuracies": calibration.bin_accuracys,
        "bin_confidences": calibration.bin_confidences,
        "bin_counts": calibration.bin_counts,
        "count": len(group)
    }

def eval_model(model, device, config, loader, multisample=False):
    model.eval()

    outputs = []
    targets = []
    metadata = []
    with torch.no_grad():
        for input, target, meta in loader:
            input = input.to(device)
            with torch.autocast(device_type="cuda", enabled=config["use_amp"]):
                if multisample:
                    output = model.predict(lambda m, n_samples: m(input, n_samples), samples=config["eval_samples"], multisample=True)
                else:
                    output = model.predict(lambda m: m(input), samples=config["eval_samples"], multisample=False)
            output = torch.logsumexp(output, dim=0).float().cpu() - math.log(output.shape[0])
            outputs.append(output)
            targets.append(target)
            metadata.append(meta)
    return torch.cat(outputs), torch.cat(targets), torch.cat(metadata)

def eval_all_groups(model, loader, config, device, multisample=False):
    outputs, targets, metadata = eval_model(model, device, config, loader, multisample)
    errors, confidences, log_likelihoods, _, _ = _analyze_output(outputs, targets, None)

    results = {}

    results["all"] = eval_group(torch.arange(0, outputs.shape[0]), outputs, metadata, targets, errors, confidences, log_likelihoods, config)
    results["all-non-toxic"] = eval_group((1 - targets).nonzero(), outputs, metadata, targets, errors, confidences, log_likelihoods, config)
    results["all-toxic"] = eval_group(targets.nonzero(), outputs, metadata, targets, errors, confidences, log_likelihoods, config)
    
    for group in ["male", "female", "lgbtq", "christian", "muslim", "other_religion", "black", "white"]:
        results[f"{group}-non-toxic"] = eval_group(get_group(metadata, targets, False, group), outputs, metadata, targets, errors, confidences, log_likelihoods, config)
        results[f"{group}-toxic"] = eval_group(get_group(metadata, targets, True, group), outputs, metadata, targets, errors, confidences, log_likelihoods, config)
    
    results["worst group accuracy"] = torch.stack([g["accuracy"] for _, g in results.items()]).min()

    return results

def run(device, config, out_path, log, rep):
    if config.get("share_file_system", False):
        torch.multiprocessing.set_sharing_strategy('file_system')
    wandb.init(
        name=f"{config['model']}-{rep}", 
        project="civil", 
        group=config["model"],
        config=config,
        tags=["scout"],
        mode=("disabled" if config["disable_wandb"] else "online"))

    model = get_model(config["model"], config, device)

    if config.get("use_checkpoint", None):
        model.load_state_dict(torch.load(out_path + f"{config['model']}_chkpt_{config['use_checkpoint']}.pth"))
        start_epoch = config["checkpoint_epochs"]
    else:
        start_epoch = 0

    train_model(model, device, config, log, out_path, start_epoch=start_epoch)
    torch.save(model.state_dict(), out_path + f"{config['model']}_final.tar")
    
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

def train_model(ensemble, device, config, log, out_path, start_epoch=0):
    ensemble.to(device)
    ensemble.train()

    before_all = time.time()

    for model_idx, (model, optimizer) in enumerate(ensemble.models_and_optimizers):
        log.info(f"==================================================")
        log.info(f"Training model {model_idx}")
        log.info(f"==================================================")

        scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])
        optimizer.init_grad_scaler(scaler)

        val_loader = wilds1.civil_comments_trainloader(config["data_path"], config["batch_size"], val=True, subsample=config["subsample"])
        trainloader = wilds1.civil_comments_trainloader(config["data_path"], batch_size=config["batch_size"], val=False, subsample=config["subsample"])

        log.info(f"Training on {len(trainloader)} minibatches")
        before = time.time()
        for epoch in range(start_epoch, config["epochs"]):
            epoch_loss = torch.tensor(0.0, device=device)
            for input, target, metadata in trainloader:
                input, target = input.to(device), target.to(device)

                def forward():
                    with torch.autocast(device_type="cuda", enabled=config["use_amp"]):
                        return F.nll_loss(model(input), target)

                def backward(loss):
                    scaler.scale(loss).backward()

                loss = optimizer.step(forward, backward, grad_scaler=scaler)
                if loss.detach().isnan().any():
                    log.info("==================================================")
                    log.info("==================== Diverged ====================")
                    log.info("==================================================")
                    raise RuntimeError("Diverged")
                scaler.update()
                epoch_loss += loss.detach()
            optimizer.complete_epoch()

            torch.save(ensemble.state_dict(), out_path + f"{config['model']}_chkpt_{model_idx}_{epoch}.pth")
            log.info(f"Epoch {epoch}: train loss {(epoch_loss / len(trainloader)):.5f}")
            wandb.log({"train_loss": epoch_loss / len(trainloader)})

            if config["eval_while_train"]:
                log.info(f"Evaluating...")
                ensemble.eval()
                eval_results = eval_all_groups(ensemble, val_loader, config, device)
                log.info(eval_results)
                wandb.log(eval_results)
                ensemble.train()
        log.info(f"Final loss: {(epoch_loss / len(trainloader)):.5f}")
        log.info(f"Training time: {time.time() - before}s")
    
    log.info(f"==================================================")
    log.info(f"Finished training")
    log.info(f"Total training time {time.time() - before_all}s")
    log.info(f"==================================================")

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

        run(device, config["params"], config["_rep_log_path"], l, rep)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(WildsExperiment)
    cw.run()
