import numpy as np
import torch
import torch.nn as nn

class DecoderBlock(nn.Module):

  def __init__(self, n_in_planes, n_out_planes):
    super().__init__()
    self.block = nn.Sequential(
      deconv4x4(n_in_planes, n_out_planes, True),
      nn.BatchNorm2d(n_out_planes),
      nn.ReLU(inplace=True),
      conv3x3(n_out_planes, n_out_planes, 1, True),
      nn.BatchNorm2d(n_out_planes)
    )

    self.upsample = lambda x: nn.functional.upsample(
      x, scale_factor=2, mode='nearest')
    self.shortcut_conv = nn.Sequential()
    if n_in_planes != n_out_planes:
      self.shortcut_conv = nn.Sequential(
        nn.Conv2d(n_in_planes, n_out_planes, kernel_size=1),
        nn.BatchNorm2d(n_out_planes)
      )

  def forward(self, x):
    out = self.block(x)
    shortcut = self.shortcut_conv(x)
    shortcut = self.upsample(shortcut)

    out += shortcut
    out = nn.functional.relu(out)
    return out

class celebA_Decoder(nn.Module):

  def __init__(self, d_latent, device='cuda', log_dir=''):
    super().__init__()

    self.d_latent = d_latent
    self.device = device

    self.mult = 8
    self.latent_mapping = nn.Sequential(
      nn.Linear(self.d_latent, 4 * 4 * 128 * self.mult),
      nn.BatchNorm1d(4 * 4 * 128 * self.mult),
      nn.ReLU()
    )
    self.block1 = DecoderBlock(128 * self.mult, 64 * self.mult)
    self.block2 = DecoderBlock(64 * self.mult, 32 * self.mult)
    self.block3 = DecoderBlock(32 * self.mult, 16 * self.mult)
    self.block4 = DecoderBlock(16 * self.mult, 8 * self.mult)
    self.output_conv = conv3x3(8 * self.mult, 3, 1, True)

    self.apply(variable_init)
    #self.to(device)
    #utils.model_info(self, 'celebA_decoder', log_dir)

  def forward(self, y):
    x = self.latent_mapping(y)
    x = x.view(-1, 128 * self.mult, 4, 4)
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.output_conv(x)
    return x


def variable_init(m, neg_slope=0.0):
  with torch.no_grad():
    if isinstance(m, (nn.Linear, nn.Conv2d)):
      nn.init.kaiming_uniform_(m.weight, neg_slope)
      if m.bias is not None:
        m.bias.zero_()
    elif isinstance(m, nn.BatchNorm2d):
      if m.weight is not None:
        m.weight.fill_(1)
      if m.bias is not None:
        m.bias.zero_()
      if m.running_mean is not None:
        m.running_mean.zero_()
      if m.running_var is not None:
        m.running_var.zero_()

def deconv4x4(n_in_planes, n_out_planes, bias=True):
  """4x4 convolution with padding"""
  return nn.ConvTranspose2d(n_in_planes, 
                           n_out_planes, 
                           kernel_size=4, 
                           stride=2,
                           padding=1,
                           bias=bias) 

def conv3x3(in_planes, out_planes, stride=1, bias=True):
  """3x3 convolution with padding"""
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=bias) 

class SigmoidNoGradient(torch.autograd.Function):

  def forward(self, x):
    return torch.nn.functional.sigmoid(x)

  def backward(self, g):
    return g.clone()

class PlusMinusOne(object):
  def __call__(self, x):
    return x  * 2.0 - 1.0