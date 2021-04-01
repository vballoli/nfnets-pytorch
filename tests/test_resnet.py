from nfnets.models.resnet import nf_resnet18, nf_resnet50

import torch


def test_resnet():
  model = nf_resnet18(activation='celu')
  model2 = nf_resnet50(activation='celu')
  
  assert model(torch.randn(1,3,224,224)) is not None
  assert model2(torch.randn(1,3,224,224)) is not None
  