import torch
import torch.nn as nn

def _down_sample(x):
    return nn.functional.avg_pool2d(x, 2, 2)

def _increase_planes(x, n_out_planes):
    n_samples, n_planes, spatial_size = x.size()[:-1]
    x_zeros = torch.zeros(
    n_samples, n_out_planes - n_planes, spatial_size, spatial_size, 
    dtype=x.dtype, device=x.device)
    return torch.cat([x, x_zeros], 1)

def _downsample_and_increase_planes(x, n_out_planes):
    x = _down_sample(x)
    x = _increase_planes(x, n_out_planes)
    return x

def identity_func(n_in_planes, n_out_planes, stride):
    identity = lambda x: x
    if stride == 2 and n_in_planes != n_out_planes:
      identity = lambda x: _downsample_and_increase_planes(x, n_out_planes)
    elif stride == 2:
      identity = _down_sample
    elif n_in_planes != n_out_planes:
      identity = lambda x: _increase_planes(x, n_out_planes)
    return identity

class BasicBlock(nn.Module):

  expansion = 1

  def __init__(self, n_in_planes, n_out_planes, stride=1):
    super().__init__()
    assert stride == 1 or stride == 2

    self.block = nn.Sequential(
      conv3x3(n_in_planes, n_out_planes, stride),
      nn.BatchNorm2d(n_out_planes),
      nn.ReLU(inplace=True),
      conv3x3(n_out_planes, n_out_planes),
      nn.BatchNorm2d(n_out_planes)
    )

    self.identity = identity_func(n_in_planes, n_out_planes, stride)

  def forward(self, x):
    out = self.block(x)
    identity = self.identity(x)

    out += identity
    out = nn.functional.relu(out)
    return out

class celebA_Encoder(nn.Module):

    def __init__(self, d_latent, device='cuda', log_dir=''):
        super().__init__()
        self.d_latent = d_latent
        self.device = device
            
        n_blocks = [1, 1, 1, 1]
        mult = 8
        n_output_planes = [16 * mult, 32 * mult, 64 * mult, 128 * mult]
        self.n_in_planes = n_output_planes[0]
        
        self.layer0 = nn.Sequential(
          conv3x3(3, self.n_in_planes, 1),
          nn.BatchNorm2d(self.n_in_planes),
          nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(BasicBlock, n_blocks[0], n_output_planes[0], 2)
        self.layer2 = self._make_layer(BasicBlock, n_blocks[1], n_output_planes[1], 2)
        self.layer3 = self._make_layer(BasicBlock, n_blocks[2], n_output_planes[2], 2)
        self.layer4 = self._make_layer(BasicBlock, n_blocks[3], n_output_planes[3], 2)
        self.latent_mapping = nn.Sequential(
            nn.Linear(n_output_planes[3] * BasicBlock.expansion, d_latent, True),
            nn.BatchNorm1d(d_latent),
            nn.Tanh()
        )
        
        self.apply(variable_init)
    
    def _make_layer(self, block, n_blocks, n_out_planes, stride=1):
        layers = []
        layers.append(block(self.n_in_planes, n_out_planes, stride))
        self.n_in_planes = n_out_planes * block.expansion
        for i in range(1, n_blocks):
          layers.append(block(self.n_in_planes, n_out_planes))

        return nn.Sequential(*layers)
    
    def forward(self, x, get_embeddings=False):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        spatial_size = x.size(2)
        x = nn.functional.avg_pool2d(x, spatial_size, 1)
        x = x.view(x.size(0), -1)
        x = self.latent_mapping(x)
        if get_embeddings:
            return x, x
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