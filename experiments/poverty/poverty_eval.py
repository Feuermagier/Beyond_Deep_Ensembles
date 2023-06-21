import sys
sys.path.append("../../")

import torch
import matplotlib.pyplot as plt
import time

from experiments.base import wilds1

from cw2.cw_data import cw_logging
from cw2 import experiment, cw_error, cluster_work

from experiments.wilds.models import get_model

from src.regresssion import RegressionResults
import os

class PovertyResult:
    @staticmethod
    def stack(results) -> torch.tensor:
        return torch.stack(list(map(lambda r: r.to_tensor(), results)))

    def __init__(self, pearson, mse, qce, sqce, lml, log_likelihood):
        super().__init__()
        self.pearson = pearson
        self.mse = mse
        self.qce = qce
        self.sqce = sqce
        self.lml = lml
        self.log_likelihood = log_likelihood

    def to_tensor(self) -> torch.tensor:
        return torch.tensor([self.pearson, self.mse, self.qce, self.sqce, self.lml, self.log_likelihood])

    def log(self, log, ty):
        log.info(f"{ty} Worst-U/R Pearson r: {self.pearson:.4f}")
        log.info(f"{ty} MSE: {self.mse.item():.4f}")
        log.info(f"{ty} QCE: {self.qce.item():.4f}")
        log.info(f"{ty} sQCE: {self.sqce.item():.4f}")
        log.info(f"{ty} LML: {self.lml.item():.4f}")
        log.info(f"{ty} Log Likelihood: {self.log_likelihood.item():.4f}")

def eval_model(model, name, testloader, load, reps, device, config, log, pre_eval_hook=None):
    model.eval()
    results = []
    for rep in range(reps):
        if load:
            path = f"/mnt/d/Uni/PdF/results/poverty/results/{name}/log/rep_{rep:02d}model.tar"
            if not os.path.exists(path):
                log.info(f"Path '{path}' does not exist - skipping model")
            model.load_state_dict(torch.load(path))
            log.info(path)
        model.to(device)

        if pre_eval_hook is not None:
            eval_model = pre_eval_hook(model)
        else:
            eval_model = model

        torch.manual_seed(42)
        outputs = []
        targets = []
        metadata = []
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=config["use_amp"]):
                for input, target, meta in testloader:
                    input = input.to(device)
                    output = eval_model.infer(input, samples=config["eval_samples"]).cpu()
                    outputs.append(output)
                    targets.append(target)
                    metadata.append(meta)
            
            outputs = torch.cat(outputs, dim=1)
            targets = torch.cat(targets)
            metadata = torch.cat(metadata)
            wilds_result = wilds1.eval_poverty(config["data_path"], outputs[:, :, :, 0].mean(dim=0), targets, metadata, ood=True)
            reg_results = RegressionResults(name, outputs, targets)
            result = PovertyResult(min(wilds_result[0]['r_urban:0'], wilds_result[0]['r_urban:1']), reg_results.mse, reg_results.qce, reg_results.sqce, reg_results.average_lml, reg_results.average_log_likelihood)
            result.log(log, "")
            results.append(result)
    return PovertyResult.stack(results)

def eval_model_id_ood(model, name, load, reps, device, config, log, pre_eval_hook=None):
    results = []
    log.info("============= ID =============")
    id_testloader = wilds1.poverty_testloader(config["data_path"], 64, False)
    results.append(eval_model(model, name, id_testloader, load, reps, device, config, log, pre_eval_hook))
    del id_testloader

    log.info("============= OOD =============")
    ood_testloader = wilds1.poverty_testloader(config["data_path"], 64, True)
    results.append(eval_model(model, name, ood_testloader, load, reps, device, config, log, pre_eval_hook))
    del ood_testloader
    return torch.stack(results)


def run(device, config, out_path, log):
    model = get_model(config["model"], device, None, torch.tensor(0.1), config, log, False)
    results = eval_model_id_ood(model, config["model"], True, config["reps"], device, config, log)
    torch.save(results, out_path + "/" + config["model"] + ".tar")

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

        torch.manual_seed(0)

        run(device, config["params"], config["_rep_log_path"], l)

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == "__main__":
    cw = cluster_work.ClusterWork(WildsExperiment)
    cw.run()
