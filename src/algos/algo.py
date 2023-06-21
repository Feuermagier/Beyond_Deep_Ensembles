from typing import Any, Dict
import torch
from torch.optim import Optimizer

class BayesianOptimizer(Optimizer):
    '''
        An optimizer that optimizes a distribution over the parameter of a model (i.e. approximate inference).
        Use on of these instead of PyTorch's standard optimizer to infer a distribution over the parameter of your model.
        
        Use the optimizer returned by get_base_optimizer for learning rate schedulers (don't apply the scheduler to the Bayesian optimizer, unless get_base_optimizer returns self!). Also, you may get a warning that the lr scheduler's step method has been called before the optimizer's step methode. This seems to be a false-positive.
        
        If youre are using a GradScaler, make sure to call optimizer.init_grad_scaler(grad_scaler) before doing the first step.
    '''

    def __init__(self, params, defaults):
        super().__init__(params, defaults)
        self._step_supports_amp_scaling = True

    def step(self, forward_closure, backward_closure):
        '''
            Makes a single step.
            BayesianOptimizer supports PyTorch's amp module (i.e. autocast), as long as the used GradScaler has been passed as a constructor argument to this optimizer.
            
            Args:
                forward_closure is a function that evaluates the loss given the current state of the module. Contrary to the closure of PyTorch's optimizer, this closure must neither clear the gradients nor call backward() on the loss!
            
                backward_closure implements a single backward pass of the model. If you use a GradScaler, call scale() on the loss passed to this closure.
        '''
        raise NotImplementedError()

    def complete_epoch(self):
        '''
            Completes a training epoch. Algorithms may want to do some bookkeeping here. Call this after every training epoch.
        '''
        pass

    def sample_parameters(self):
        '''
            Samples concrete values for all parameter of this optimizer.
            During evaluation, call this before calling the forward() function of your model.
        '''
        raise NotImplementedError()

    def init_grad_scaler(self, grad_scaler: torch.cuda.amp.GradScaler):
        '''
            Initializes the GradScaler. This is required as GradScalers usually initialize themself lazily on the first step/unscale, but we need to access it before that
        '''
        if grad_scaler is not None and grad_scaler.is_enabled() and grad_scaler._scale is None:
            grad_scaler._lazy_init_scale_growth_tracker(self._params_device())

    def get_base_optimizer(self):
        '''
            Returns the optimizer that does the actual parameter updates. Use this optimizer for e.g. learning rate schedulers.
        '''
        pass

    def _params_device(self):
        return self.param_groups[0]["params"][0].device

    def _params(self):
        for group in self.param_groups:
            for param in group["params"]:
                yield param

    def _prepare_and_check_grads(self, grad_scaler: torch.cuda.amp.GradScaler, optimizer=None):
        if grad_scaler is None or not grad_scaler.is_enabled():
            return True

        opt = self if optimizer is None else optimizer
        grad_scaler.unscale_(opt)

        # From GradScaler._maybe_opt_step
        return sum(v.item() for v in self.state["found_inf_per_device"].values()) == 0

    def _set_grad_scaler_state(self, grad_scaler: torch.cuda.amp.GradScaler, stage, optimizer=None):
        if grad_scaler is None or not grad_scaler.is_enabled():
            return

        opt = self if optimizer is None else optimizer
        grad_scaler._per_optimizer_states[id(opt)]["stage"] = stage


class LastLayerBayesianOptimizer(BayesianOptimizer):
    '''
        Joins the given ll_bayesian_optimizer and the deterministic_optimizer. 
        This is useful for e.g. a Bayesian last layer and a deterministic rest of the network.
    
        Behavior of this optimizer may be arbitrarily wrong if the two passed optimizer have joined parameters.
        Especially make sure that the ll_bayesian_optimizer doesn't touch the gradients of the parameters that have been passed to the deterministic optimizer.
    '''

    def __init__(self, ll_bayesian_optimizer: BayesianOptimizer, deterministic_optimizer: Optimizer):
        self.ll_bayesian_optimizer = ll_bayesian_optimizer
        self.deterministic_optimizer = deterministic_optimizer
    
    def step(self, forward_closure, backward_closure, grad_scaler=None):
        if grad_scaler is not None and grad_scaler.is_enabled():
            raise ValueError("Doesn't support grad scaler")
        
        self.deterministic_optimizer.zero_grad()
        loss = self.ll_bayesian_optimizer.step(forward_closure, backward_closure) # Makes at least one forward & backward pass, therefore creates gradients for the deterministic optimizer
        
        self.deterministic_optimizer.step()

        return loss

    def complete_epoch(self):
        self.ll_bayesian_optimizer.complete_epoch()

    def sample_parameters(self):
        self.ll_bayesian_optimizer.sample_parameters()

    def init_grad_scaler(self, grad_scaler: torch.cuda.amp.GradScaler):
        if grad_scaler.is_enabled():
            raise RuntimeError("Doesn't support grad scaler")
        # self.ll_bayesian_optimizer.init_grad_scaler(grad_scaler)
        # self.grad_scaler_for_deterministic = torch.cuda.amp.GradScaler(enabled=grad_scaler.is_enabled())

    def get_base_optimizer(self):
        raise RuntimeError("There is no defined base optimizer on the ll optimizer. Call get_base_optimizer directly on the passed ll bayesian optimizer")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "ll_bayesian_optimizer": self.ll_bayesian_optimizer.state_dict(),
            "deterministic_optimizer": self.deterministic_optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.ll_bayesian_optimizer.load_state_dict(state_dict["ll_bayesian_optimizer"])
        self.deterministic_optimizer.load_state_dict(state_dict["deterministic_optimizer"])

    def __repr__(self) -> str:
        return "LL Bayesian Optimizer: \n\n" + self.ll_bayesian_optimizer.__repr__() + "\n==================================\nDeterministic Optimizer:\n\n" + self.deterministic_optimizer.__repr__()
