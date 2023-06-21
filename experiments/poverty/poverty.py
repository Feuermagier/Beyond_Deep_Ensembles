import sys
sys.path.append("../../")

import torch
import torch.nn.functional as F
import time

from experiments.base import wilds1
from src.algos.ensemble import DeepEnsemble
from src.algos.util import nll_loss
from src.eval.regresssion import RegressionResults

from cw2.cw_data import cw_logging
from cw2 import experiment, cw_error, cluster_work

from models import get_model, get_var_optimizer

import wandb

def eval_model(model, split, device, config, log):
    model.eval()

    torch.manual_seed(42)
    dataset = wilds1.poverty_split(config["data_path"], split, fold=config["fold"])
    loader = wilds1.poverty_loader(dataset, config["batch_size"], subsample=config["test_subsample"])
    
    outputs = []
    targets = []
    metadata = []
    with torch.no_grad():
        with torch.autocast(device_type="cuda", enabled=config["use_amp"]):
            for input, target, meta in loader:
                input = input.to(device)
                output = model.predict(lambda m: m(input), samples=config["eval_samples"]).cpu()
                outputs.append(output)
                targets.append(target)
                metadata.append(meta)
        
        outputs = torch.cat(outputs, dim=1)
        targets = torch.cat(targets)
        metadata = torch.cat(metadata)
        wilds_result = dataset.eval(outputs[:, :, :, 0].mean(dim=0), targets, metadata)
        reg_results = RegressionResults(outputs, targets)
        return {
            "pearson": min(wilds_result[0]['r_urban:0'], wilds_result[0]['r_urban:1']),
            "mse": reg_results.mse,
            "qce": reg_results.qce,
            "sqce": reg_results.sqce,
            "avg_lml": reg_results.average_lml,
            "avg_ll": reg_results.average_log_likelihood,
            "wilds": wilds_result
        }

def eval_model_id_ood(model, device, config, log):
    log.info("============= ID =============")
    id_results = eval_model(model, "id_test", device, config, log)

    log.info("============= OOD =============")
    ood_results = eval_model(model, "test", device, config, log)
    return {
        "id": id_results,
        "ood": ood_results
    }

def run(device, config, out_path, log, rep):
    wandb_mode = "disabled" if config.get("disable_wandb") else "online"
    wandb.init(
        name=f"{config['model']}-{config['members']}-({config['fold']})", 
        project=f"poverty", 
        group=config["model"],
        config=config,
        tags=[config['model']],
        mode=wandb_mode)

    model = get_model(config["model"], device, config)
    #with torch.autograd.anomaly_mode.detect_anomaly():
    train_model(model, device, config, log, out_path)
    torch.save(model.state_dict(), out_path + f"{config['model']}_final.tar")

    before = time.time()
    result = eval_model_id_ood(model, device, config, log)
    log.info(result)
    log.info(f"Eval time: {time.time() - before}")
    wandb.log(result)

def train_model(ensemble: DeepEnsemble, device, config, log, out_path):
    ensemble.to(device)
    ensemble.train()

    before_all = time.time()

    if config.get("move_model", False):
        for model in ensemble.models:
            model.cpu()

    for model_idx, (model, optimizer) in enumerate(ensemble.models_and_optimizers):
        log.info(f"==================================================")
        log.info(f"Training model {model_idx}")
        log.info(f"==================================================")

        torch.cuda.empty_cache()
        model.to(device)

        if config["learn_var"]:
            var_optimizer = get_var_optimizer(model, config)
        else:
            var_optimizer = None

        scaler = torch.cuda.amp.GradScaler(enabled=config["use_amp"])
        optimizer.init_grad_scaler(scaler)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer.get_base_optimizer(), gamma=config["lr_decay"])

        if config["train_on_val"]:
            log.info("Training on the validation set")
            trainloader = wilds1.poverty_loader(wilds1.poverty_split(config["data_path"], "id_val", config["fold"]), config["batch_size"], subsample=config["subsample"])
        else:
            trainloader = wilds1.poverty_loader(wilds1.poverty_split(config["data_path"], "train", config["fold"]), config["batch_size"], subsample=config["subsample"])
        
        log.info(f"Training on {len(trainloader)} minibatches")
        before = time.time()
        for epoch in range(config["epochs"]):
            epoch_loss = torch.tensor(0.0, device=device)
            for input, target, metadata in trainloader:
                input, target = input.to(device), target.to(device)

                if var_optimizer is not None:
                    var_optimizer.zero_grad() 

                def forward():
                    with torch.autocast(device_type="cuda", enabled=config["use_amp"]):
                        return nll_loss(model(input), target)

                def backward(loss):
                    scaler.scale(loss).backward()

                loss = optimizer.step(forward, backward, grad_scaler=scaler)
                if loss.isnan().any():
                    log.info("========================================================")
                    log.info("==================== Diverged ==========================")
                    log.info("========================================================")
                    raise RuntimeError("Diverged")

                if var_optimizer is not None:
                    var_optimizer.step()

                scaler.update()
                epoch_loss += loss.detach()
            optimizer.complete_epoch()
            scheduler.step()
            epoch_loss /= len(trainloader)

            if epoch % 50 == 0:
                torch.save(ensemble.state_dict(), out_path + f"{config['model']}_chkpt_{model_idx}_{epoch}.pth")
            log.info(f"Epoch {epoch}: train loss {(epoch_loss):.5f}")
            wandb.log({"train_loss": epoch_loss})

            if config["eval_while_train"] and epoch % 10 == 0:
                log.info(f"Evaluating...")
                ensemble.eval()
                eval_results = eval_model(ensemble, "val", device, config, log)
                log.info(eval_results)
                wandb.log(eval_results)
                ensemble.train()
        log.info(f"Final loss: {epoch_loss:.5f}")
        log.info(f"Training time: {time.time() - before}s")
    
    for model in ensemble.models:
        model.to(device)

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

        torch.manual_seed(rep * 42)

        run(device, config["params"], config["_rep_log_path"], l, rep)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(WildsExperiment)
    cw.run()
