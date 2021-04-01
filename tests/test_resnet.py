from nfnets.models.resnet import nf_resnet18, nf_resnet50

import torch


def test_resnet():
  model = nf_resnet18(activation='celu')
  model2 = nf_resnet50(activation='celu')
  
  input = torch.randn(1,3,224,224)
  
  out = model(input)
  out2 = model2(input)
  
  assert out is not None
  assert out2 is not None
  
  model = nf_resnet18(activation='gelu')
  model2 = nf_resnet50(activation='gelu')
  
  assert model(input) is not None
  assert model2(input) is not None
  
  model = nf_resnet18(activation='silu')
  model2 = nf_resnet50(activation='silu')
  
  assert model(input) is not None
  assert model2(input) is not None
  
  model = nf_resnet18(activation='softplus')
  model2 = nf_resnet50(activation='softplus')
  
  assert model(input) is not None
  assert model2(input) is not None
  
  model = nf_resnet18(activation='elu')
  model2 = nf_resnet50(activation='elu')
  
  assert model(input) is not None
  assert model2(input) is not None
  