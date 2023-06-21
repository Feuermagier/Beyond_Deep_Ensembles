import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import matplotlib.pyplot as plt
from netcal.metrics import ECE

from src.algos import util
from src.eval.calibration import ClassificationCalibrationResults


def eval_model(model, samples, testloader, device, wilson_baseline=None):
    torch.manual_seed(42)

    # Evaluate
    errors = []
    confidences = []
    log_likelihoods = []
    agreements = []
    total_variations = []
    targets = []
    marginals = []
    with torch.no_grad():
        for data, target in testloader:
            targets.append(target)
            data = data.to(device)

            output = model.predict(lambda m: m(data), samples)#.mean(dim=0).cpu()
            output = torch.logsumexp(output, dim=0).cpu() - torch.log(torch.tensor(output.shape[0]))
            marginals.append(output.exp())

            if wilson_baseline is not None:
                wilson_output = wilson_baseline.infer(data, samples)#.mean(dim=0).cpu()
                wilson_output = torch.logsumexp(wilson_output, dim=0).cpu() - torch.log(torch.tensor(wilson_output.shape[0]))
            else:
                wilson_output = None

            err, conf, ll, agreement, total_variation = _analyze_output(output, target, wilson_output)
            errors.append(err)
            confidences.append(conf)
            log_likelihoods.append(ll)
            agreements.append(agreement)
            total_variations.append(total_variation)

            # outputs = eval_fn(data.to(device), samples).cpu()
            # sample_preds = torch.transpose(
            #     torch.argmax(outputs, dim=2), 0, 1)
            # preds = torch.mode(sample_preds, dim=1)[0]
            # errors = torch.cat((errors, preds == target))
            # confs = outputs[:, torch.arange(
            #     outputs.shape[1]), preds].mean(dim=0).exp()
            # confidences = torch.cat((confidences, confs))
    errors = torch.cat(errors)
    confidences = torch.cat(confidences)
    accuracy = errors.sum() / len(errors)
    avg_log_likelihood = torch.cat(log_likelihoods).mean()
    avg_likelihood = torch.cat(log_likelihoods).exp().mean()

    if wilson_baseline is not None:
        agreements = torch.cat(agreements)
        total_variations = torch.cat(total_variations)
        agreement = agreements.mean()
        total_variation = total_variations.mean()
    else:
        agreement = None
        total_variation = None
    
    #netcal_ece = ECE(10, True).measure(torch.cat(marginals).numpy(), torch.cat(targets).numpy())

    calibration = ClassificationCalibrationResults(10, errors, confidences)
    return EvalResult(accuracy, avg_log_likelihood, avg_likelihood, calibration, agreement, total_variation)

# models = [(name, eval_fn, loss_over_time, [ece_over_time per dataset in the same order], eval_samples)]
# datasets = [(name, dataloader)]

class EvalResult:
    @staticmethod
    def stack(results) -> torch.tensor:
        return torch.stack(list(map(lambda r: r.to_tensor(), results)))

    def __init__(self, acc, avg_ll, avg_l, calibration, hmc_agreement, hmc_tv):
        self.acc = acc
        self.avg_ll = avg_ll
        self.avg_l = avg_l
        self.ece = calibration.ece
        self.sece = calibration.signed_ece
        self.hmc_agreement = hmc_agreement
        self.hmc_tv = hmc_tv

    def to_tensor(self) -> torch.tensor:
        tensor = torch.tensor([self.acc, self.avg_ll, self.avg_l, self.ece, self.sece])
        if self.hmc_agreement is not None:
            tensor = torch.cat((tensor, torch.tensor([self.hmc_agreement])))
        if self.hmc_tv is not None:
            tensor = torch.cat((tensor, torch.tensor([self.hmc_tv])))
        return tensor

    def log(self, log, ty):
        log.info(f"{ty} Accuracy: {self.acc:.4f}")
        log.info(f"{ty} Avg Log Likelihood: {self.avg_ll:.4f}")
        log.info(f"{ty} Avg Likelihood: {self.avg_l:.4f}")
        log.info(f"{ty} ECE: {self.ece:.4f}")
        log.info(f"{ty} sECE: {self.sece:.4f}")
        if self.hmc_agreement is not None:
            log.info(f"{ty} Agreement: {self.hmc_agreement:.4f}")
        if self.hmc_tv is not None:
            log.info(f"{ty} Total Variation: {self.hmc_tv:.4f}")


# def eval_multiple(models, datasets, device, include_ace=True, include_mce=False):
#     torch.manual_seed(42)
#     width = len(models)
#     height = len(datasets) + 1  # 2 * len(dataset) + 1
#     fig = plt.figure(figsize=(8 * width, 5 * height))
#     #fig.suptitle(name + f" (Test Accuracy {accuracy:.3f})", fontsize=16)

#     for i, (name, _, loss_over_time, _, _) in enumerate(models):
#         loss_ax = fig.add_subplot(height, width, i + 1)
#         loss_ax.annotate(name, xy=(0.5, 1), xytext=(0, 10), xycoords="axes fraction",
#                          textcoords="offset points",  ha="center", va="center", fontsize=16)
#         util.plot_losses(name, loss_over_time, loss_ax)

#     for i, (name, loader) in enumerate(datasets):
#         for j, (_, eval_fn, _, eces, eval_samples) in enumerate(models):
#             # ece_ax = fig.add_subplot(height, width, (2 * i + 1) * width + j + 1)
#             # ece_ax.set_xlabel("Epoch", fontsize=14)
#             # ece_ax.set_xticks(np.arange(1, len(eces) + 1, 1))
#             # ece_ax.set_ylabel("ECE", fontsize=14)
#             # if eces != []:
#             #     ece_ax.plot(np.arange(1, len(eces) + 1, 1), eces)

#             rel_ax = fig.add_subplot(height, width, (i + 1) * width + j + 1)
#             errors = []
#             confidences = []
#             with torch.no_grad():
#                 for data, target in loader:
#                     output = eval_fn(data.to(device), eval_samples).mean(dim=0).cpu()
#                     err, conf = _analyze_output(output, target)
#                     errors.append(err)
#                     confidences.append(conf)
#                     # outputs = eval_fn(data.to(device), eval_samples).cpu()
#                     # sample_preds = torch.transpose(
#                     #     torch.argmax(outputs, dim=2), 0, 1)
#                     # preds = torch.mode(sample_preds, dim=1)[0]
#                     # errors = torch.cat((errors, preds == target))
#                     # confs = outputs[:, torch.arange(
#                     #     outputs.shape[1]), preds].mean(dim=0).exp()
#                     # confidences = torch.cat((confidences, confs))
#             errors = torch.cat(errors)
#             confidences = torch.cat(confidences)
#             reliability_diagram(10, errors, confidences, rel_ax, include_ace, include_mce)

#             if j == 0:
#                 rel_ax.annotate(name, xy=(0, 0.5), xytext=(-rel_ax.yaxis.labelpad - 10, 0),
#                                 xycoords=rel_ax.yaxis.label, textcoords="offset points", fontsize=16, ha="left", va="center")

#     fig.tight_layout()
#     fig.subplots_adjust(left=0.2, top=0.95)
#     return fig

def _analyze_output(output, target, wilson_output):
    preds = torch.argmax(output, dim=1)
    errors = preds == target
    confidences = torch.clamp(torch.max(output, dim=1)[0].exp(), 0, 1)
    ll = output[torch.arange(output.shape[0]), target]

    if wilson_output is not None:
        agreements = (preds == wilson_output.argmax(dim=1)).float()
        total_variation = (output.exp() - wilson_output.exp()).abs().sum(dim=1) / 2
    else:
        agreements = None
        total_variation = None

    return errors, confidences, ll, agreements, total_variation

def binary_to_multiclass(outputs):
    outputs = outputs.squeeze(-1)
    return torch.stack((1 - outputs, outputs), dim=-1)
