import torch
import numpy as np
import matplotlib

def calculate_ace(bin_count, errors, confidences):
    bins = _create_adaptive_bins(bin_count, confidences)
    return _mean_calibration_error(bins, errors, confidences)

def calculate_ece(bin_count, errors, confidences):
    bins = _create_static_bins(bin_count, confidences)
    return _mean_calibration_error(bins, errors, confidences)

def calculate_mce(bin_count, errors, confidences):
    bins = _create_static_bins(bin_count, confidences)
    return _max_calibration_error(bins, errors, confidences)

def reliability_diagram(bin_count, errors, confidences, ax, include_accuracy=True, include_ace=True, include_mce=False, include_ece=True, include_bin_sizes=True):
    assert len(errors) == len(confidences)
    
    static_bins = _create_static_bins(bin_count, confidences)

    bin_accuracys = np.array([errors[bin].sum() / len(bin) if len(bin) > 0 else 0 for bin in static_bins])
    mid = np.linspace(0, 1, bin_count)
    bin_errors = np.abs(np.array(bin_accuracys) - mid)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(color="tab:grey", linestyle=(0, (1, 5)), linewidth=1)
    interval = np.arange(0, 1, 1 / bin_count)
    ax.bar(interval, bin_accuracys, 1 / bin_count, align="edge", color="b", edgecolor="k")
    ax.bar(interval, bin_errors, 1 / bin_count, bottom=np.minimum(bin_accuracys, mid), align="edge", color="mistyrose", alpha=0.5, edgecolor="r", hatch="/")
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_xlabel('Confidence', fontsize=14)

    if include_bin_sizes:
        for (x, count) in zip(interval, static_bins):
            ax.text(x + 0.5 * 1 / bin_count, 0.01, str(len(count)), color="white", fontsize=14, ha="center")

    ident = [0.0, 1.0]
    ax.plot(ident,ident,linestyle='--',color="tab:grey")

    ece = _mean_calibration_error(static_bins, errors, confidences)
    text = ""

    if include_ece:
        text += f"ECE: {ece:.3f}"

    if include_mce:
        mce = _max_calibration_error(static_bins, errors, confidences)
        text += f"\MCE: {mce:.3f}"

    if include_ace:
        ace = calculate_ace(bin_count, errors, confidences)
        text += f"\nACE: {ace:.3f}"

    if include_accuracy:
        acc = errors.sum() / len(errors)
        text += f"\nAcc: {acc:.3f}"

    if text != "":
        ax.text(0.08, 0.9, text, 
            transform=ax.transAxes, fontsize=16, verticalalignment="top", 
            bbox={"boxstyle": "square,pad=0.5", "facecolor": "white"})

    return ece

def _create_static_bins(bin_count, confidences):
    bins = [[] for _ in range(bin_count)]
    for i, confidence in enumerate(confidences):
        bin = torch.clamp(torch.floor(confidence * bin_count).int(), min=0, max=bin_count - 1)
        bins[bin].append(i)
    return bins

def _create_adaptive_bins(bin_count, confidences):
    _, indices = torch.sort(confidences)
    return [bin.tolist() for bin in torch.chunk(indices, bin_count)]

def _mean_calibration_error(bins, errors, confidences):
    bin_accuracys = np.array([errors[bin].sum() / len(bin) if len(bin) > 0 else 0 for bin in bins])
    bin_confidences = np.array([confidences[bin].sum() / len(bin) if len(bin) > 0 else 0 for bin in bins])

    ece = 0
    for i in range(len(bins)):
        ece += len(bins[i]) * np.abs(bin_accuracys[i] - bin_confidences[i])
    ece /= len(confidences)

    return ece

def _max_calibration_error(bins, errors, confidences):
    bin_accuracys = np.array([errors[bin].sum() / len(bin) if len(bin) > 0 else 0 for bin in bins])
    bin_confidences = np.array([confidences[bin].sum() / len(bin) if len(bin) > 0 else 0 for bin in bins])
    return np.max(np.abs(bin_accuracys - bin_confidences))

class ClassificationCalibrationResults:
    @staticmethod
    def average(results):
        bin_counts = np.mean([r.bin_counts for r in results], axis=0)
        bin_accuracys = np.mean([r.bin_accuracys for r in results], axis=0)
        bin_confidences = np.mean([r.bin_confidences for r in results], axis=0)
        ece = np.mean([r.ece for r in results], axis=0)
        result = ClassificationCalibrationResults.__new__(ClassificationCalibrationResults)
        result.bin_accuracys = bin_accuracys
        result.bin_confidences = bin_confidences
        result.ece = ece
        result.bin_counts = bin_counts
        return result

    def __init__(self, bin_count, errors, confidences):
        bins = _create_static_bins(bin_count, confidences)
        self.bin_accuracys = np.array([errors[bin].sum() / len(bin) if len(bin) > 0 else 0 for bin in bins])
        self.bin_confidences = np.array([confidences[bin].sum() / len(bin) if len(bin) > 0 else 0 for bin in bins])
        self.ece = 0
        for i in range(len(bins)):
            self.ece += len(bins[i]) * np.abs(self.bin_accuracys[i] - self.bin_confidences[i])
        self.ece /= len(confidences)

        self.bin_counts = [len(bin) for bin in bins]

    @property
    def signed_ece(self):
        sece = 0
        for i in range(len(self.bin_counts)):
            sece += self.bin_counts[i] * (self.bin_accuracys[i] - self.bin_confidences[i])
        sece /= sum(self.bin_counts)

        return sece

    def plot_reliability(self, ax, include_text=True, title=None, fractions=True, actual_confidences=False, include_sizes=True):
        ax.set_xlabel("Confidence", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.set_xlim(0, 1)
        ax.plot([0, 1], [0,1], color="royalblue")
        
        if actual_confidences:
            conf_points = self.bin_confidences
        else:
            offset = 1 / len(self.bin_counts) / 2
            conf_points = torch.linspace(offset, 1 - offset, len(self.bin_counts))

        ax.plot(self.bin_confidences, self.bin_accuracys, "o-", color="darkorange")
        ax.set_xticks(conf_points)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
        ax.set_ylim(0, 1)
        ax.xaxis.grid(True, linestyle="-", alpha=0.4)

        if include_sizes:
            count_ax = ax.twiny()
            count_ax.set_xticks(conf_points)
            count_ax.set_xticklabels([f"{(c / sum(self.bin_counts)):.2f}" if fractions else f"{c}" for c in self.bin_counts])
            count_ax.set_xlabel("Datapoints", fontsize=14)

        if include_text:
            if title is not None:
                text = f"{title}\nECE: {self.ece:.3f}"
            else:
                text = f"ECE: {self.ece:.3f}"
            ax.text(0.08, 0.9, text, 
                transform=ax.transAxes, fontsize=14, verticalalignment="top", 
                bbox={"boxstyle": "square,pad=0.5", "facecolor": "white"})