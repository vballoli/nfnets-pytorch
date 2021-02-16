# PyTorch implementation of Normalizer-Free Networks and SGD - Adaptive Gradient Clipping
![Python Package](https://github.com/vballoli/nfnets-pytorch/workflows/Upload%20Python%20Package/badge.svg)
![Docs](https://readthedocs.org/projects/nfnets-pytorch/badge/?version=latest
)

Paper: https://arxiv.org/abs/2102.06171.pdf

Original code: https://github.com/deepmind/deepmind-research/tree/master/nfnets

Do star this repository if it helps your work!

> Note: See [this comment](https://github.com/vballoli/nfnets-pytorch/issues/1#issuecomment-778853439) for a generic implementation for any optimizer as a temporary reference for anyone who needs it.

# Installation

Install from PyPi:

`pip3 install nfnets-pytorch`

or install the latest code using:

`pip3 install git+https://github.com/vballoli/nfnets-pytorch`
# Usage
## WSConv2d

Use `WSConv2d` like any other `torch.nn.Conv2d`.

```python
import torch
from torch import nn
from nfnets import WSConv2d

conv = nn.Conv2d(3,6,3)
w_conv = WSConv2d(3,6,3)
```
## SGD - Adaptive Gradient Clipping

Similarly, use `SGD_AGC` like `torch.optim.SGD`
```python
import torch
from torch import nn, optim
from nfnets import WSConv2d, SGD_AGC

conv = nn.Conv2d(3,6,3)
w_conv = WSConv2d(3,6,3)

optim = optim.SGD(conv.parameters(), 1e-3)
optim_agc = SGD_AGC(conv.parameters(), 1e-3)
```

## Generic AGC
```python
import torch
from torch import nn, optim
from nfnets import WSConv2d
from nfnets.agc import AGC # Needs testing

conv = nn.Conv2d(3,6,3)
w_conv = WSConv2d(3,6,3)

optim = optim.SGD(conv.parameters(), 1e-3)
optim_agc = AGC(conv.parameters(), optim) # Needs testing
```

## Using it within any PyTorch model

```python
import torch
from torch import nn
from torchvision.models import resnet18

from nfnets import replace_conv

model = resnet18()
replace_conv(model)
```

# Docs

Find the docs at [readthedocs](https://nfnets-pytorch.readthedocs.io/en/latest/)

# TODO
- [x] WSConv2d
- [x] SGD - Adaptive Gradient Clipping
- [x] Function to automatically replace Convolutions in any module with WSConv2d
- [x] Documentation
- [x] Generic AGC wrapper.(See [this comment](https://github.com/vballoli/nfnets-pytorch/issues/1#issuecomment-778853439) for a reference implementation) (Needs testing for now)
- [x] WSConvTranspose2d
- [ ] NFNets 
- [ ] NF-ResNets

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
