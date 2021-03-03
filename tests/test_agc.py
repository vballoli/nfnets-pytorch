import torch
from torch.optim import SGD
from torchvision.models import resnet18

from nfnets import replace_conv, AGC

def test_agc():
  model = resnet18()
  replace_conv(model)
  optim = SGD(model.parameters(), 1e-3)
  optim = AGC(model.parameters(), optim, model=model)
  optim.zero_grad()
  model(torch.randn(1,3,64,64)).sum().backward()
  optim.step()