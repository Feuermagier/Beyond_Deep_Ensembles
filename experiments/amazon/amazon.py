import sys
sys.path.append("../../")

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from experiments.base import wilds1
from experiments.amazon.models import get_model
from experiments.base.multiclass_classification import _analyze_output
from src.eval.calibration import ClassificationCalibrationResults
import math

from cw2.cw_data import cw_logging
from cw2 import experiment, cw_error, cluster_work

import wandb

import os

def eval_model_on_dataset(model, device, config, loader, multisample=False):
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
                    output = model.predict(lambda m: m(input), samples=config["eval_samples"])
            output = torch.logsumexp(output.float(), dim=0).cpu() - math.log(output.shape[0])
            outputs.append(output)
            targets.append(target)
            metadata.append(meta)
    return torch.cat(outputs), torch.cat(targets), torch.cat(metadata)

def eval_model(model, config, device, split="test", subsample=None, multisample=False):
    model.eval()

    dataset = wilds1.amazon_split(config["data_path"], split)
    loader = wilds1.amazon_loader(dataset, config["batch_size"], subsample=subsample)
    outputs, targets, metadata = eval_model_on_dataset(model, device, config, loader, multisample=multisample)
    errors, confidences, log_likelihoods, _, _ = _analyze_output(outputs, targets, None)

    wilds_results = dataset.eval(torch.argmax(outputs, dim=1), targets, metadata)
    calibration = ClassificationCalibrationResults(config["ece_bins"], errors, confidences)
    print(wilds_results)
    return {
        "wilds": wilds_results,
        "10th_percentile_acc": wilds_results[0]["10th_percentile_acc"],
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
    wandb.init(
        name=f"{config['model']}-{rep}", 
        project="amazon", 
        group=config["model"],
        config=config,
        tags=[],
        mode=("disabled" if config["disable_wandb"] else "online"))

    model = get_model(config["model"], config, device)

    if config.get("use_checkpoint", None):
        if config.get("checkpoint_path", None) is not None:
            model.load_state_dict(torch.load(config.get("checkpoint_path")))
        else:
            model.load_state_dict(torch.load(out_path + f"{config['model']}_chkpt_{config['use_checkpoint']}.pth"))
        start_epoch = config["checkpoint_epochs"]
    else:
        start_epoch = 0


    if not config.get("eval_only", False):
        train_model(model, device, config, log, out_path, start_epoch=start_epoch)
        torch.save(model.state_dict(), out_path + f"{config['model']}_final.tar")
    else:
        print("============================= Eval only")
        load_path = config['load_path'] + f"rep_0{rep}{config['load_model']}_final.tar"
        print(f"Loading from '{load_path}'")
        model.load_state_dict(torch.load(load_path))
        #model.load_state_dict(torch.load(out_path + f"{config['model']}_final.tar"))
    
    eval_time = time.time()

    ood_test_results = eval_model(model, config, device, "test", subsample=config["test_subsample"])
    log.info({
        "ood_test_results": ood_test_results
    })
    id_test_results = eval_model(model, config, device, "id_test", subsample=config["test_subsample"])
    log.info({
        "id_test_results": id_test_results
    })

    # Don't overload wandb...
    del ood_test_results["wilds"]
    del id_test_results["wilds"]

    wandb.log({
        "ood_test_results": ood_test_results,
        "id_test_results": id_test_results
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

        trainloader = wilds1.amazon_loader(wilds1.amazon_split(config["data_path"], "train"), config["batch_size"], subsample=config["subsample"])

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
                scaler.update()
                epoch_loss += loss.detach()
            optimizer.complete_epoch()

            torch.save(ensemble.state_dict(), out_path + f"{config['model']}_chkpt_{model_idx}_{epoch}.pth")
            log.info(f"Epoch {epoch}: train loss {(epoch_loss / len(trainloader)):.5f}")
            wandb.log({"train_loss": epoch_loss / len(trainloader)})

            if config["eval_while_train"] and epoch % 3 == 0:
                log.info(f"Evaluating...")
                ensemble.eval()
                eval_results = eval_model(ensemble, config, device, "val", subsample=config["test_subsample"])

                log.info(eval_results)
                del eval_results["wilds"] # Don't overload wandb...
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

        torch.manual_seed(rep + 1 + config.get("seed_offset", 0))

        run(device, config["params"], config["_rep_log_path"], l, rep + config.get("seed_offset", 0))

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(WildsExperiment)
    cw.run()
