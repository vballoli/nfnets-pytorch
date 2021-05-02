import torch
from torch import nn

from nfnets import WSConv2d, ScaledStdConv2d



class SqueezeExcite(nn.Module):
  
  def __init__(self, in_channels, out_channels, se_ratio=0.5, hidden_channels=None, activation='relu'):
    assert (se_ratio != None) or ((se_ratio is None) and (hidden_channels is not None))
    
    if se_ratio is None:
      hidden_channels = hidden_channels
    else:
      hidden_channels = max(1, se_ratio * in_channels)
      
    self.fc0 = nn.Linear(in_channels, hidden_channels)
    self.fc1 = nn.Linear(hidden_channels, out_channels)
    
    self.activation  = activation_fn[activation]
    super(SqueezeExcite, self).__init__()
    
  def forward(self, x):
    h = torch.mean(x, [2,3])
    h = self.fc0(h)
    h = self.fc1(self.activation(h))
    
    return h.expand_as(x)
  
  
class NFBlock(nn.Module):
  
  def __init__(self, in_channels, out_channels, expansion=0.5, se_ratio=0.5, kernel_shape=3, group_size=128, stride=1, beta=1.0, alpha=0.2, conv=ScaledStdConv2d, activation='gelu'):
    
    width = int(self.out_channels * expansion)
    self.groups = width // group_size
    self.width = group_size * self.groups
    
    self.conv0 = conv(in_channels, self.width, 1)
    
    self.conv1 = conv(self.width, self.width, 3, groups=self.groups)
    
    self.alpha = alpha
    self.beta = beta
    