# Beyond Deep Ensembles: A Large-Scale Evaluation of Bayesian Deep Learning under Distribution Shift

**TL;DR: We evaluate the generalization capability, calibration, and posterior approximation quality of many SOTA Bayesian deep learning algorithms on large-scale tasks, incorporating realistic distribution-shifted data from [WILDS](https://wilds.stanford.edu/). This repository contains robust PyTorch implementations of the algorithms and supporting evaluation code.**
---

This repository contains the algorithm implementation and evaluation code for the NeurIPS 2023 paper

[_Beyond Deep Ensembles: A Large-Scale Evaluation of Bayesian Deep Learning under Distribution Shift_](
http://arxiv.org/abs/2306.12306).

by [Florian Seligmann](https://github.com/Feuermagier), [Philipp Becker](https://alr.anthropomatik.kit.edu/21_72.php), [Michael Volpp](https://de.linkedin.com/in/michaelvolpp), and [Gerhard Neumann](https://alr.anthropomatik.kit.edu/21_65.php).



## Introduction
> Bayesian deep learning (BDL) is a promising approach to achieve well-calibrated predictions on distribution-shifted data. Nevertheless, there exists no large-scale survey that evaluates recent SOTA methods on diverse, realistic, and challenging benchmark tasks in a systematic manner. To provide a clear picture of the current state of BDL research, we evaluate modern BDL algorithms on real-world datasets from the WILDS collection containing challenging classification and regression tasks, with a focus on generalization capability and calibration under distribution shift. We compare the algorithms on a wide range of large, convolutional and transformer-based neural network architectures. In particular, we investigate a signed version of the expected calibration error that reveals whether the methods are over- or underconfident, providing further insight into the behavior of the methods. Further, we provide the first systematic evaluation of BDL for fine-tuning large pre-trained models, where training from scratch is prohibitively expensive. Finally, given the recent success of Deep Ensembles, we extend popular single-mode posterior approximations to multiple modes by the use of ensembles. While we find that ensembling single-mode approximations generally improves the generalization capability and calibration of the models by a significant margin, we also identify a failure mode of ensembles when finetuning large transformer-based language models. In this setting, variational inference based approaches such as last-layer Bayes By Backprop outperform other methods in terms of accuracy by a large margin, while modern approximate inference algorithms such as SWAG achieve the best calibration.

Please cite our work if you find it useful in your research:
```bibtex
@article{seligmann2023bayes-eval,
    title = {Beyond Deep Ensembles: A Large-Scale Evaluation of Bayesian Deep Learning under Distribution Shift},
    author = {Seligmann, Florian and Becker, Philipp and Volpp, Michael and Neumann, Gerhard},
    journal = {arXiv preprint arXiv:2306.12306},
    year = {2023}
}
```
If you have any question, feel free to open an issue in this repository!


## Implemented Algorithms
We provide PyTorch code for the following algorithms:
- Maximum A Posteriori (MAP)
- [Monte Carlo Dropout](https://arxiv.org/abs/1506.02142) (MCD)
- [Deep Ensembles](https://arxiv.org/abs/1612.01474)
- [Bayes By Backprop](https://arxiv.org/abs/1505.05424) (BBB)
- [Rank-1 Variational Inference](https://arxiv.org/abs/2005.07186) (Rank-1 VI)
- [Stochastic Weight Averaging - Gaussian (SWAG)](https://arxiv.org/abs/1902.02476) (SWAG)
- [Stein Variational Gradient Descent](https://arxiv.org/abs/1608.04471) (SVGD)
- [Improved Variational Online-Newton (iVON)](https://arxiv.org/abs/2002.10060) (iVON)
- [Laplace Approximation](https://arxiv.org/abs/2106.14806)
- [Spectrally-Normalized Gaussian Processes (SNGP)](https://arxiv.org/abs/2006.10108)

as well as a general framework to ensemble any Bayesian algorithm ("MultiX").


## Structure of the Code
[`src`](./src/) contains the implementation of the algorithms ([`src/algos`](./src/algos/)), evaluation metrics ([`src/eval`](./src/eval/)), and architectures that we implemented from scratch ([`src/architectures`](./src/architectures/)).


## Usage of the Algorithms
All algorithms are implemented as PyTorch optimizers.
Because many algorithms require special handling of the forward and backward pass, the optimizer's `step` methods require `forward` and `backward` closures to be passed to them.
The `forward_closure` closure should execute a single forward pass and must *not* call `backward()` on the loss, but return it.
The `backward_closure` closure should execute a single backward pass on the passed `loss`: `loss.backward()` or `scaler.scale(loss).backward()` if using a gradient scaler.
You need to call `complete_epoch()` on the optimizer after each *epoch*, as some algorithms (mainly SWAG) want to do some bookkeeping here.

All algorithms are subclasses of the [`BayesianOptimizer`](./src/algos/algo.py), which contains further documentation. 
This class also contains special code to handle gradient scalers.


## Reproduction of the Experiments

### Setup
Make sure that you have PyTorch 2.0 and a compatible version of TorchVision installed.
Then run
```
pip install matplotlib tabulate wilds netcal cw2 transformers wandb laplace-torch
pip install git+https://github.com/treforevans/uci_datasets.git
```
WILDS also requires a version of [TorchScatter](https://github.com/rusty1s/pytorch_scatter) that is compatible with PyTorch 2.0.

Use the following code snippet to selectively download WILDS datasets (e.g. for iwildcam):
```python
from .experiments.base import wilds1

wilds1.download_dataset("./data/", "iwildcam")
```
You can also specify a different path, but then you have to adapt all pathes in the experiment configuration files.

If you want to reproduce the CIFAR-10 experiments, you also need to run the following commands:
```
pip install jax==0.4.1 jaxlib==0.4.1+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tensorflow tensorflow_datasets
pip install dm-haiku
```
Finally, you need to download the HMC samples that where made available by Izmailow et al. with `gsutil`:
```
gsutil -m cp -r gs://gresearch/bnn-posteriors/v1/hmc/cifar10/ ./data/Wilson/
```

### Running the Experiments
For each task there is a corresponding directory below [`experiments`](./experiments/).
Each directory contains a Python file with the name of the task (e.g. `iwildcam.py`) and a YAML file with the same name.
First, run the non-MultiX algorithms by running e.g.  `python3 iwildcam.py iwildcam.yaml` for iWildCam.
Then, evaluate the MultiX models (reuses the trained models from the single-mode algorithms) by running `python3 eval_ensembles.py eval_ensembles.yaml` in the same directory as before.
Finally, fit the Laplace approximations on top of the MAP models by running `python3 fit_laplace.py fit_laplace.yaml` in the same directory.
All scripts print their results to stdout and to WandB if you are logged in and `disable_wandb` is `False` in the YAML files.
The experiment directories also contain Jupyter Notebooks to query the results from WandB, plot them, and print LaTeX tables from them.

For UCI, you only need to run `python3 uci.py uci.yaml`, as this script also fits the Laplace approximations and evaluates MultiX.
For PovertyMap-wilds, the `eval_ensembles` script is also not required as the main script also trains the ensembles.

## License
The `google-bnn-hmc` folder has been copied from https://github.com/google-research/google-research/tree/master/bnn_hmc.
This code is licensed under the Apache-2.0 license (see https://github.com/google-research/google-research/tree/master).

The `mnist-c` submodule is licensed under the Apache-2.0 license.
