import torch

from nfnets import WSConv2d, WSConvTranspose2d, WSConv1d

def test_wsconv1d(k=5):
    c = WSConv1d(3, 6, kernel_size=k, padding=int(k / 2 + 1))
    assert c(torch.randn(1, 3, 100)) is not None, "Conv failed."

def test_wsconv2d():
    c = WSConv2d(3,6,3)
    assert c(torch.randn(1,3,32,32)) is not None, "Conv failed."

def test_wsconvtranspose2d():
    c = WSConvTranspose2d(3,6,3)
    assert c(torch.randn(1,3,32,32)) is not None, "Conv failed."
