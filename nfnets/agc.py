import torch
from torch import nn, optim

from nfnets.utils import unitwise_norm


class AGC(optim.Optimizer):
    """Generic implementation of the Adaptive Gradient Clipping

    Args:
      params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
      optim (torch.optim.Optimizer): Optimizer with base class optim.Optimizer
      clipping (float, optional): clipping value (default: 1e-3)
      eps (float, optional): eps (default: 1e-3)
    """
    def __init__(self, params, optim: optim.Optimizer, clipping: float=1e-2, eps: float=1e-3):
        if clipping < 0.0:
            raise ValueError("Invalid clipping value: {}".format(clipping))
        if eps < 0.0:
            raise ValueError("Invalid eps value: {}".format(eps))

        self.optim = optim

        defaults = dict(clipping=clipping, eps=eps)
        defaults = {**defaults, **optim.defaults}
        super(AGC, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                param_norm = torch.max(unitwise_norm(
                    p), torch.tensor(group['eps']).to(p.device))
                grad_norm = unitwise_norm(p.grad)
                max_norm = param_norm * group['clipping']

                trigger = grad_norm > max_norm

                clipped_grad = p.grad * \
                    (max_norm / torch.max(grad_norm,
                                          torch.tensor(1e-6).to(grad_norm.device)))
                p.grad.data.copy_(torch.where(trigger, clipped_grad, p.grad))

        return self.optim.step(closure)
        