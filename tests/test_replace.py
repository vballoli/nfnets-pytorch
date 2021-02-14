import torch
from torch import nn
from torchvision.models import resnet18

from nfnets import replace_conv

def test_replace_conv():
    model = resnet18()
    replace_conv(model)
    for module in model.modules():
        
        assert type(module) is not nn.Conv2d, "Conv2d found, test failed."