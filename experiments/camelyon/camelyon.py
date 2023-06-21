import sys
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from experiments.base import wilds1
from experiments.camelyon.models import get_model
from experiments.base.multiclass_classification import _analyze_output
from src.eval.calibration import ClassificationCalibrationResults
from src.algos.ensemble import DeepEnsemble
import math

from cw2.cw_data import cw_logging
from cw2 import experiment, cw_error, cluster_work

import wandb

def eval_model_on_dataset(model, device, config, loader):
    outputs = []
    targets = []
    metadata = []
    with torch.no_grad():
        for input, target, meta in loader:
            input = input.to(device)
            with torch.autocast(device_type="cuda", enabled=config["use_amp"]):
                output = model.predict(lambda m: m(input), samples=config["eval_samples"])
            output = torch.logsumexp(output.float(), dim=0).cpu() - math.log(output.shape[0])
            outputs.append(output)
            targets.append(target)
            metadata.append(meta)
    return torch.cat(outputs), torch.cat(targets), torch.cat(metadata)

def eval_model(model, config, device, split="test", subsample=None):
    model.eval()

    dataset = wilds1.camelyon_split(config["data_path"], split)
    outputs, targets, metadata = eval_model_on_dataset(model, device, config, wilds1.camelyon_loader(dataset, config["batch_size"], subsample=subsample))
    print(outputs.shape)
    print(len(wilds1.camelyon_loader(dataset, config["batch_size"], subsample=subsample)))
    errors, confidences, log_likelihoods, _, _ = _analyze_output(outputs, targets, None)

    wilds_results = dataset.eval(outputs[...,1].exp().round(), targets, metadata)
    calibration = ClassificationCalibrationResults(config["ece_bins"], errors, confidences)
    return {
        "wilds": wilds_results,
        "accuracy": errors.sum() / len(errors),
        "log_likelihood": log_likelihoods.mean(),
        "ece": calibration.ece,
        "sece": calibration.signed_ece,
        "bin_accuracies": calibration.bin_accuracys,
        "bin_confidences": calibration.bin_confidences,
        "bin_counts": calibration.bin_counts
    }

def run(device, config, out_path, log, rep):
    if config.get("share_file_system", False):
        torch.multiprocessing.set_sharing_strategy('file_system')
    wandb_mode = "disabled" if config.get("disable_wandb") else "online"
    wandb.init(
        name=f"{config['model']}-{config['members']}-({rep})", 
        project="camelyon17", 
        group=config["model"],
        config=config,
        tags=[config['model']],
        mode=wandb_mode)

    model = get_model(config["model"], config, device)

    train_model(model, device, config, log, out_path)
    torch.save(model.state_dict(), out_path + f"{config['model']}_final.tar")

    eval_time = time.time()

    test_results = eval_model(model, config, device, split="test", subsample=config["test_subsample"])
    log.info(f"Test: {test_results}")

    val_results = eval_model(model, config, device, split="val", subsample=config["test_subsample"])
    log.info(f"OOD Validation: {val_results}")

    id_val_results = eval_model(model, config, device, split="id_val", subsample=config["test_subsample"])
    log.info(f"ID Validation: {id_val_results}")

    wandb.log({
        "test_results": test_results,
        "val_results": val_results,
        "id_val_results": id_val_results
    })
    log.info(f"Eval time: {time.time() - eval_time}s")

def train_model(ensemble: DeepEnsemble, device, config, log, out_path):
    ensemble.to(device)

    before_all = time.time()

    for model_idx, (model, optimizer) in enumerate(ensemble.models_and_optimizers):
        log.info(f"==================================================")
        log.info(f"Training model {model_idx}")
        log.info(f"==================================================")

        scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])
        optimizer.init_grad_scaler(scaler)

        if config["train_on_val"]:
            log.info("Training on the validation set")
            trainloader = wilds1.camelyon_loader(wilds1.camelyon_split(config["data_path"], "id_val"), config["batch_size"], subsample=config["subsample"])
        else:
            trainloader = wilds1.camelyon_loader(wilds1.camelyon_split(config["data_path"], "train"), config["batch_size"], subsample=config["subsample"])

        log.info(f"Training on {len(trainloader)} minibatches")
        before = time.time()
        for epoch in range(config["epochs"]):
            ensemble.train()
            epoch_loss = torch.tensor(0.0, device=device)
            for input, target, metadata in trainloader:
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
            epoch_loss /= len(trainloader)

            torch.save(ensemble.state_dict(), out_path + f"{config['model']}_chkpt_{model_idx}_{epoch}.pth")
            log.info(f"Epoch {epoch}: train loss {(epoch_loss):.5f}")
            wandb.log({"train_loss": epoch_loss}, step=(epoch + model_idx * config["epochs"]))

            if config["eval_while_train"]:
                log.info(f"Evaluating...")
                ensemble.eval()
                eval_results = eval_model(ensemble, config, device, "val", subsample=config["test_subsample"])
                log.info(eval_results)
                wandb.log({
                    "eval": eval_results
                }, step=(epoch + model_idx * config["epochs"]))
                ensemble.train()
        log.info(f"Final loss: {epoch_loss:.5f}")
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

        torch.manual_seed(rep)

        run(device, config["params"], config["_rep_log_path"], l, rep)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(WildsExperiment)
    cw.run()
