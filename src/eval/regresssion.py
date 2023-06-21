import torch
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
import math
from tabulate import tabulate

from src.algos.util import gauss_logprob

class RegressionResults:
    @staticmethod
    def average(results):
        result = RegressionResults.__new__(RegressionResults)
        result.name = results[0].name
        result.mean_mse = torch.mean(torch.stack([r.mean_mse for r in results]), dim=0)
        result.mse_of_means = torch.mean(torch.stack([r.mse_of_means for r in results]), dim=0)
        result.lml = torch.mean(torch.stack([r.lml for r in results]), dim=0)
        result.average_lml = torch.mean(torch.stack([r.average_lml for r in results]), dim=0)
        result.observed_cdf = torch.mean(torch.stack([r.observed_cdf for r in results]), dim=0)
        result.quantile_ps = torch.mean(torch.stack([r.quantile_ps for r in results]), dim=0)
        result.qce = torch.mean(torch.stack([r.qce for r in results]), dim=0)
        return result

    def __init__(self, outputs, targets, cal_steps=10, target_mean=0, target_std=1):
        '''
        outputs = [samples, datapoints, out_dim, 2 = mean + var]
        '''

        samples = outputs.shape[0]
        datapoints = outputs.shape[1]

        means, stds = denormalize_outputs(outputs, target_mean, target_std)
        targets = targets * target_std + target_mean
        log_likelihoods = gauss_logprob(means, stds**2, targets)

        self.mse = F.mse_loss(means.mean(dim=0), targets)
        self.log_likelihood = -datapoints * math.log(samples) + torch.logsumexp(log_likelihoods, dim=0).sum()
        self.average_log_likelihood = self.log_likelihood / datapoints
        self.lml = -math.log(samples) + torch.logsumexp(log_likelihoods.sum(dim=1), dim=0).squeeze(-1)
        self.average_lml = self.lml / datapoints
        self.observed_cdf = calc_quantile_frequencies(means, stds, targets, cal_steps)
        self.quantile_ps = torch.linspace(0, 1, cal_steps)
        self.qce = (self.observed_cdf - self.quantile_ps).abs().mean()

    @property
    def sqce(self):
        return (self.observed_cdf - self.quantile_ps).mean()

def calc_quantile_frequencies(means, stds, targets, quantile_steps):
    quantile_ps = torch.linspace(0, 1, 2 * quantile_steps - 1).to(means.device)
    samples = torch.normal(means, stds)
    # samples = torch.distributions.Normal(means, stds).sample()

    quantiles = torch.stack([torch.quantile(samples, p, dim=0, keepdim=False, interpolation="nearest") for p in quantile_ps])

    quantile_frequencies = torch.zeros(2 * quantile_steps - 1)
    for i, quantile in enumerate(quantiles):
        quantile_frequencies[i] = (targets <= quantile).sum()
    quantile_frequencies /= targets.numel()
    
    obs_confidences = torch.zeros(quantile_steps)
    for i in range(quantile_steps):
        obs_confidences[i] = quantile_frequencies[quantile_steps + i - 1] - quantile_frequencies[quantile_steps - i - 1]
    return obs_confidences

def plot_calibration(title, results, ax, include_text=True):
    ax.set_xlabel("Expected Confidence Level", fontsize=14)
    ax.set_ylabel("Observed Confidence Level", fontsize=14)
    ax.plot([0, 1], [0,1], color="royalblue")
    ax.plot(results.quantile_ps, results.observed_cdf, "o-", color="darkorange")
    ax.set_xlim(0, 1)
    ax.set_xticks(results.quantile_ps)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(0, 1)
    ax.xaxis.grid(True, linestyle="-", alpha=0.4)
    if include_text:
        if title is not None:
            text = f"{title}\nQCE: {results.qce:.3f}"
        else:
            text = f"QCE: {results.qce:.3f}"
        ax.text(0.08, 0.9, text, 
            transform=ax.transAxes, fontsize=14, verticalalignment="top", 
            bbox={"boxstyle": "square,pad=0.5", "facecolor": "white"})

def plot_table(title, results, filename=None):
    average_lmls = torch.tensor([[res.average_lml for res in reses] for reses in results])
    mean_mses = torch.tensor([[res.mean_mse for res in reses] for reses in results])
    mse_of_means = torch.tensor([[res.mse_of_means for res in reses] for reses in results])
    qces = torch.tensor([[res.qce for res in reses] for reses in results])
    sqces = torch.tensor([[res.sqce for res in reses] for reses in results])
    divisor = math.sqrt(average_lmls.shape[1])
    texts = [[results[i][0].name, 
            f"{average_lmls[i].mean():.2f} ± {(average_lmls[i].std() / divisor):.2f}", 
            f"{mean_mses[i].mean():.3f} ± {(mean_mses[i].std() / divisor):.3f}", 
            f"{mse_of_means[i].mean():.3f} ± {(mse_of_means[i].std() / divisor):.3f}", 
            f"{qces[i].mean():.2f} ± {(qces[i].std() / divisor):.2f}"]
        for i in range(len(results))]
    cols = (title, "Avg LML", "Mean MSE", "MSE of Means", "QCE")
    table = tabulate(texts, headers=cols, tablefmt='orgtbl')
    print(table)

    if filename is not None:
        with open(filename, "w") as file:
            file.write(table)

    plt.plot(average_lmls.mean(dim=1))
    plt.xticks(torch.arange(1, len(results) + 1, 1))

    for result, lml, mean_mse, mse, qce, sqce in zip(results, average_lmls, mean_mses, mse_of_means, qces, sqces):
        print((
            f"{result[0].name}"
            f" & ${lml.mean():.2f} \\pm {(lml.std() / divisor):.2f}$"
            #f" & ${mean_mse.mean():.2f} \\pm {(mean_mse.std() / divisor):.2f}$"
            f" & ${mse.mean():.4f} \\pm {(mse.std() / divisor):.4f}$"
            f" & ${qce.mean():.2f}\\textrm{{{'O' if sqce.mean() < 0 else 'U'}}} \\pm {(qce.std() / divisor):.2f}$"
            " \\\\"
        ))

def normalize(x, data_mean, data_std):
    return (x - data_mean) / data_std

def denormalize(y, target_mean, target_std):
    return y * target_std + target_mean

def denormalize_outputs(outputs, target_mean, target_std):
    return outputs[:,:,:,0] * target_std + target_mean, outputs[:,:,:,1] * target_std