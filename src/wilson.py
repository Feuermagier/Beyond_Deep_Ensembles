import torch
import torch.nn as nn

import jax
from jax import numpy as jnp
import numpy as onp

import sys
sys.path.append("google-bnn-hmc/")

from bnn_hmc.utils import checkpoint_utils
from bnn_hmc.utils import models
from bnn_hmc.utils import metrics
from bnn_hmc.utils import precision_utils
from bnn_hmc.utils import data_utils

dataset_configs = {
    "cifar10": (261, 3, jax.nn.log_softmax)
    # UCI is unpublished as of now
}

class WilsonHMC(nn.Module):
    def __init__(self, path, dataset):
        super().__init__()
        self.path = path
        self.dataset = dataset
        net_apply, net_init = models.get_model("resnet20_frn_swish", {"num_classes": 10})
        self.net_apply = jax.jit(precision_utils.rewrite_high_precision(net_apply))
        self.params = [self.get_params_and_state(i) for i in range(dataset_configs[dataset][0])]

    def forward(self, input, samples=1):
        return self.infer(input, samples)

    def infer(self, input, samples):
        # Ignoring the wanted sample count to achieve maximal accuracy of the posterior approximation
        device = input.device
        input = jnp.transpose(jnp.asarray(input.cpu().detach().numpy(), dtype=jnp.float32), (0, 2, 3, 1))
        result = []
        for params, net_states in self.params:
            for chain_id in range(dataset_configs[self.dataset][1]):
                param, net_state = self.get_chain(params, net_states, chain_id)
                preds, _ = self.net_apply(param, net_state, None, (input, None), False)
                preds = dataset_configs[self.dataset][2](preds)
                result.append(torch.from_numpy(onp.asarray(preds).copy()).to(device))
        return torch.stack(result)

    def get_params_and_state(self, sample_id):
        ckpt_dict = checkpoint_utils.load_checkpoint(
            f"{self.path}/{self.dataset}/state-{sample_id}.pkl".format(sample_id))
        params = ckpt_dict["params"]
        net_state = ckpt_dict["net_state"]
        return params, net_state

    def get_chain(self, params, net_state, chain_id):
        params = jax.tree_map(lambda p: p[chain_id], params)
        net_state = jax.tree_map(lambda p: p[chain_id], net_state)
        return params, net_state
