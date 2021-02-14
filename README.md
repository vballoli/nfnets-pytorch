# PyTorch implementation of Normalizer-Free Networks and SGD - Adaptive Gradient Clipping

Paper: https://arxiv.org/abs/2102.06171.pdf
Original code: https://github.com/deepmind/deepmind-research/tree/master/nfnets

# TODO
- [x] WSConv2d
- [x] SGD - Adaptive Gradient Clipping
- [x] Function to automatically replace Convolutions in any module with WSConv2d
- [ ] NFNets 
- [ ] NF-ResNets
- [ ] Documentation

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

## Using it within any PyTorch model

```python
import torch
from torch import nn
from torchvision.models import resnet18

from nfnets import replace_conv

model = resnet18()
replace_conv(model)
```

# Cite

To cite the original paper, use:
```
@article{brock2021high,
  author={Andrew Brock and Soham De and Samuel L. Smith and Karen Simonyan},
  title={High-Performance Large-Scale Image Recognition Without Normalization},
  journal={arXiv preprint arXiv:},
  year={2021}
}
```
