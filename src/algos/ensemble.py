from types import LambdaType
from typing import Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepEnsemble(nn.Module):
    '''
        Stores modules together with their optimizers
    '''
    def __init__(self, models_and_optimizers):
        super().__init__()
        self.models = nn.ModuleList(list(map(lambda p: p[0], models_and_optimizers)))
        self.optimizers = list(map(lambda p: p[1], models_and_optimizers))

    def state_dict(self, prefix='', keep_vars=False):
        return {
            "models": self.models.state_dict(prefix=prefix, keep_vars=keep_vars),
            "optimizers": list(map(lambda o: o.state_dict(), self.optimizers))
        }

    def load_state_dict(self, state_dict, strict=True):
        self.models.load_state_dict(state_dict["models"], strict=strict)
        for optimizer, optimizer_state in zip(self.optimizers, state_dict["optimizers"]):
            optimizer.load_state_dict(optimizer_state)

    def predict(self, predict_closure, samples, multisample=False):
        '''
            Makes <samples> predictions with this ensemble

            predict_closure takes a model (that is part of this ensemble) as an argument and makes a single prediction with it. This function already calls sample_parameters()
        '''
        if len(self.models) == 1 and getattr(self.models[0], "supports_multisample", False) and multisample:
            return predict_closure(self.models[0], n_samples=samples)

        output = []
        preds_per_model = samples // len(self.models_and_optimizers)
        for i, (model, optimizer) in enumerate(self.models_and_optimizers):
            model_samples = preds_per_model if i > 0 else (samples - (len(self.models_and_optimizers) - 1) * preds_per_model)
            for _ in range(model_samples):
                optimizer.sample_parameters()
                output.append(predict_closure(model))
        return torch.stack(output)

    @property
    def models_and_optimizers(self):
        return list(zip(self.models, self.optimizers))
