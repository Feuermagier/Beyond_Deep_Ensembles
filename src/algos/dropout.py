import torch
import torch.nn as nn
import torch.nn.functional as F

class FixableDropout(nn.Module):
    '''
        A special version of PyTorchs torch.nn.Dropout that applies dropout when the model is in evaluation mode.
        If freeze_on_eval is True, the same dropout mask will be used for the entire minibatch when in evaluation mode (not in train model!)
    '''

    def __init__(self, p, freeze_on_eval=True):
        super().__init__()
        self.p = torch.tensor(p)
        self.freeze_on_eval = freeze_on_eval

    def forward(self, x):
        if not self.training and self.freeze_on_eval:
            mask = (1 - self.p).expand(x.shape[1:])
            mask = torch.bernoulli(mask).to(x.device)
            return x * mask
        else:
            return F.dropout(x, self.p)
    
    def __repr__(self) -> str:
        return f"FixableDropout({self.p:0.3})"

def patch_dropout(module, freeze_on_eval=False, override_p=None, patch_fixable=False):
    '''
        Replaces all torch.nn.Dropout layers by FixableDropout layers.
        If override_p is None, the original dropout rate is being conserved.
        Otherwise, the rate is set to dropout_p.
        If patch_fixable is True, FixableDropout layers get also replace (useful for changing the dropout rates)
    '''
    patched = 0
    for name, m in list(module._modules.items()):
        if m._modules:
            patched += patch_dropout(m, freeze_on_eval=freeze_on_eval, override_p=override_p, patch_fixable=patch_fixable)
        elif m.__class__.__name__ == "Dropout" or (patch_fixable and m.__class__.__name__ == "FixableDropout"):
            patched += 1
            if override_p is not None:
                setattr(module, name, FixableDropout(override_p, freeze_on_eval))
            else:
                setattr(module, name, FixableDropout(m.p, freeze_on_eval))
    return patched
