import torch
from torch import nn

from nfnets import WSConv2d


def replace_conv(module: nn.Module, conv_class=WSConv2d):
    """Recursively replaces every convolution with WSConv2d.

    Usage: replace_conv(model) #(In-line replacement)
    Args:
      module (nn.Module): target's model whose convolutions must be replaced.
      conv_class (Class): Class of Conv(WSConv2d or ScaledStdConv2d)
    """
    for name, mod in module.named_children():
        target_mod = getattr(module, name)
        if type(mod) == torch.nn.Conv2d:
            setattr(module, name, conv_class(target_mod.in_channels, target_mod.out_channels, target_mod.kernel_size,
                                           target_mod.stride, target_mod.padding, target_mod.dilation, target_mod.groups, target_mod.bias))
        
        if type(mod) == torch.nn.BatchNorm2d:
            setattr(module, name, torch.nn.Identity())

    for name, mod in module.named_children():
        replace_conv(mod, conv_class)

def unitwise_norm(x: torch.Tensor):
    if x.ndim <= 1:
        dim = 0
        keepdim = False
    elif x.ndim in [2, 3]:
        dim = 0
        keepdim = True
    elif x.ndim == 4:
        dim = [1, 2, 3]
        keepdim = True
    else:
        raise ValueError('Wrong input dimensions')

    return torch.sum(x**2, dim=dim, keepdim=keepdim) ** 0.5
