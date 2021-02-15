import torch
from torch import nn


class WSConv2d(nn.Conv2d):
    """WSConv2d

    Reference: https://github.com/deepmind/deepmind-research/blob/master/nfnets/base.py#L121
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                         dilation=dilation, groups=groups, bias=bias, padding_mode=padding_mode)

        nn.init.kaiming_normal_(self.weight)
        self.gain = torch.ones(self.weight.size(0), requires_grad=True)

    def standardize_weight(self, eps):
        var, mean = torch.var_mean(self.weight, dim=(1, 2, 3), keepdims=True)
        fan_in = torch.prod(torch.tensor(self.weight.shape[0:]))

        scale = torch.rsqrt(torch.max(
            var * fan_in, torch.tensor(eps).to(var.device))) * self.gain.view_as(var).to(var.device)
        shift = mean * scale
        return self.weight * scale - shift

    def forward(self, input, eps=1e-4):
        self.weight.data.copy_(self.standardize_weight(eps))
        return super().forward(input)
