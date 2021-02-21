# PyTorch implementation of Normalizer-Free Networks and Adaptive Gradient Clipping
![Python Package](https://github.com/vballoli/nfnets-pytorch/workflows/Upload%20Python%20Package/badge.svg)
![Docs](https://readthedocs.org/projects/nfnets-pytorch/badge/?version=latest
)

Paper: https://arxiv.org/abs/2102.06171.pdf

Original code: https://github.com/deepmind/deepmind-research/tree/master/nfnets

Do star this repository if it helps your work, and [don't forget to cite](https://github.com/vballoli/nfnets-pytorch#cite-this-repository) if you use this code in your research!

# Installation

Install from PyPi:

`pip3 install nfnets-pytorch`

or install the latest code using:

`pip3 install git+https://github.com/vballoli/nfnets-pytorch`
# Usage
## WSConv2d

Use `WSConv1d, WSConv2d, ScaledStdConv2d(timm)` and `WSConvTranspose2d` like any other `torch.nn.Conv2d` or `torch.nn.ConvTranspose2d` modules.

```python
import torch
from torch import nn
from nfnets import WSConv2d, WSConvTranspose2d, ScaledStdConv2d

conv = nn.Conv2d(3,6,3)
w_conv = WSConv2d(3,6,3)

conv_t = nn.ConvTranspose2d(3,6,3)
w_conv_t = WSConvTranspose2d(3,6,3)
```

## Generic AGC (recommended)
```python
import torch
from torch import nn, optim
from torchvision.models import resnet18

from nfnets import WSConv2d
from nfnets.agc import AGC # Needs testing

conv = nn.Conv2d(3,6,3)
w_conv = WSConv2d(3,6,3)

optim = optim.SGD(conv.parameters(), 1e-3)
optim_agc = AGC(conv.parameters(), optim) # Needs testing

# Ignore fc of a model while applying AGC.
model = resnet18()
optim = torch.optim.SGD(model.parameters(), 1e-3)
optim = AGC(model.parameters(), optim, model=model, ignore_agc=['fc'])
```
## SGD - Adaptive Gradient Clipping

Similarly, use `SGD_AGC` like `torch.optim.SGD`
```python
# The generic AGC is preferable since the paper recommends not applying AGC to the last fc layer.
import torch
from torch import nn, optim
from nfnets import WSConv2d, SGD_AGC

conv = nn.Conv2d(3,6,3)
w_conv = WSConv2d(3,6,3)

optim = optim.SGD(conv.parameters(), 1e-3)
optim_agc = SGD_AGC(conv.parameters(), 1e-3)
```

## Using it within any PyTorch model

`replace_conv` replaces the convolution in your model with the convolution class and replaces the batchnorm with identity. While the identity is not ideal, it shouldn't cause a major difference in the latency. 
```python
import torch
from torch import nn
from torchvision.models import resnet18

from nfnets import replace_conv, WSConv2d, ScaledStdConv2d

model = resnet18()
replace_conv(model, WSConv2d) # This repo's original implementation
replace_conv(model, ScaledStdConv2d) # From timm

"""
class YourCustomClass(nn.Conv2d):
  ...
replace_conv(model, YourCustomClass)
"""
```

# Docs

Find the docs at [readthedocs](https://nfnets-pytorch.readthedocs.io/en/latest/)

# Cite Original Work

To cite the original paper, use:
```
@article{brock2021high,
  author={Andrew Brock and Soham De and Samuel L. Smith and Karen Simonyan},
  title={High-Performance Large-Scale Image Recognition Without Normalization},
  journal={arXiv preprint arXiv:},
  year={2021}
}
```

# Cite this repository

To cite this repository, use:
```
@misc{nfnets2021pytorch,
  author = {Vaibhav Balloli},
  title = {A PyTorch implementation of NFNets and Adaptive Gradient Clipping},
  year = {2021},
  howpublished = {\url{https://github.com/vballoli/nfnets-pytorch}}
}
```

# TODO
- [x] WSConv2d
- [x] SGD - Adaptive Gradient Clipping
- [x] Function to automatically replace Convolutions in any module with WSConv2d
- [x] Documentation
- [x] Generic AGC wrapper.(See [this comment](https://github.com/vballoli/nfnets-pytorch/issues/1#issuecomment-778853439) for a reference implementation) (Needs testing for now)
- [x] WSConvTranspose2d
- [x] WSConv1d(Thanks to [@shi27feng](https://github.com/shi27feng))