.. NFNets PyTorch documentation master file, created by
   sphinx-quickstart on Sun Feb 14 19:35:12 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NFNets PyTorch's documentation!
==========================================
NFNets-PyTorch is an implementation of the paper: "High-Performance Large-Scale Image Recognition Without Normalization
". Original paper can be found at arxiv_. You can find other implementations at PapersWithCode_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. _arxiv: https://arxiv.org/abs/2102.06171v1.pdf

.. _PapersWithCode: https://paperswithcode.com/paper/high-performance-large-scale-image

*********
Install
*********
Stable release

.. code-block:: console

   pip3 install nfnets-pytorch

Latest code

.. code-block:: console

   pip3 install git+https://github.com/vballoli/nfnets-pytorch

******************
Sample usage
******************

.. code-block:: python

   import torch
   from torch import nn
   from torchvision.models import resnet18

   from nfnets import replace_conv, SGD_AGC

   model = resnet18()
   replace_conv(model)
   optim = SGD_AGC(model.parameters(), 1e-3)

.. toctree::
   :maxdepth: 2
   :caption: API reference

   nfnets
