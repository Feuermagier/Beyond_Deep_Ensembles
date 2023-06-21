import torch

def apply_lr(source: torch.optim.Optimizer, target: torch.optim.Optimizer):
    '''
        Sets the learning rate of each of the target's param groups to those of the source.
        Useful when using nested optimizers together with LR schedulers
    '''
    for group_src, group_target in zip(source.param_groups, target.param_groups):
        if "lr" in group_src:
            if group_src["lr"] != group_target["lr"]:
                print(group_src["lr"], group_target["lr"])
            group_target["lr"] = group_src["lr"]
