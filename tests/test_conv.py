import torch

from nfnets import WSConv2d

def test_wsconv2d():
    c = WSConv2d(3,6,3)
    assert c(torch.randn(1,3,32,32)) is not None, "Conv failed."