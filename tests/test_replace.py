import torch
from torch import nn
from torchvision.models import vgg16

from nfnets import replace_conv, WSConv2d

def test_replace_conv():
    model = vgg16()
    replace_conv(model, WSConv2d)
    for module in model.modules():
        assert type(module) is not nn.Conv2d, "Conv2d found, test failed."