import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import numpy as np
import torchvision
from .srm_filter_kernel import all_normalized_hpf_list
from . import MPNCOV 
from .vit import Transformer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class TLU(nn.Module):
  def __init__(self, threshold):
    super(TLU, self).__init__()

    self.threshold = threshold

  def forward(self, input):
    output = torch.clamp(input, min=-self.threshold, max=self.threshold)

    return output

class HPF(nn.Module):
  def __init__(self):
    super(HPF, self).__init__()

    #Load 30 SRM Filters
    all_hpf_list_5x5 = []

    for hpf_item in all_normalized_hpf_list:
      if hpf_item.shape[0] == 3:
        hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')

      all_hpf_list_5x5.append(hpf_item)
    # all_hpf_list_5x5 = all_hpf_list_5x5[-2::]
    hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_5x5).view(30, 1, 5, 5), requires_grad=False)


    self.hpf = nn.Conv2d(1, 30, kernel_size=5, padding=2, bias=False)
    self.hpf.weight = hpf_weight

    #Truncation, threshold = 3 
    self.tlu = TLU(3.0)

  def forward(self, input):

    output = self.hpf(input)
    # output = self.tlu(output)

    return output

class base(nn.Module):
  def __init__(self):
    super(base, self).__init__()
      # dim, depth, heads, dim_head, mlp_dim, dropout=0.
    self.transformer = Transformer(528, 2, 4, 16, 128, 0)

    self.patch_size = 64
    self.group1 = HPF()


    self.group1_b = nn.Sequential(
      nn.Conv2d(90, 32, kernel_size=3, padding=1),
      nn.LayerNorm([32,self.patch_size,self.patch_size]),
      nn.Hardtanh(min_val=-5,max_val=5),

    )

    self.group2 = nn.Sequential(
      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.LayerNorm([32,self.patch_size,self.patch_size]),
      nn.ReLU(),

      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.LayerNorm([32,self.patch_size,self.patch_size]),
      nn.ReLU(),

      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.LayerNorm([32,self.patch_size,self.patch_size]),
      nn.ReLU(),

      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.LayerNorm([32,self.patch_size,self.patch_size]),
      nn.ReLU(),

      nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
    )

    self.group3 = nn.Sequential(
      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.LayerNorm([32,self.patch_size//2,self.patch_size//2]),
      nn.ReLU(),

      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.LayerNorm([32,self.patch_size//2,self.patch_size//2]),
      nn.ReLU(),

      nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
    )

    self.group4 = nn.Sequential(
      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.LayerNorm([32,self.patch_size//4,self.patch_size//4]),
      nn.ReLU(),

      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.LayerNorm([32,self.patch_size//4,self.patch_size//4]),
      nn.ReLU(),

      nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
    )

    self.group5 = nn.Sequential(
      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.LayerNorm([32,self.patch_size//8,self.patch_size//8]),
      nn.ReLU(),

      nn.Conv2d(32, 32, kernel_size=3, padding=1),
      nn.LayerNorm([32,self.patch_size//8,self.patch_size//8]),
      nn.ReLU(),

    )


    self.relu = nn.ReLU()
    self.advpool = nn.AdaptiveAvgPool2d((4,4))




  def forward(self, input):

    img = input
    bs,n,c,h,w = img.shape
    img = img.reshape(bs,n*c,h,w)
    img = img.reshape(bs*n*c,1,h,w)
    img = self.group1(img)
    img = img.reshape(bs, n, c , 30, h ,w)
    
    img = img.reshape(bs*n,c,30,h,w)
    img = img.reshape(bs*n,c*30,h,w)

    img = self.group1_b(img)
    
    output = self.group2(img)
    f1 = output.clone()
    output = self.group3(output)
    f2 = output.clone()
    output = self.group4(output)
    f3 = output.clone()
    output = self.group5(output)
    f4 = output.clone()
    # output = self.advpool(output)
    # output = output.view(output.size(0), -1)
    bs_n,c,w,h = output.shape
    output = output.reshape(bs,n,c,w,h)
    output = output.reshape(bs*n,c,w,h)
    
    output = MPNCOV.CovpoolLayer(output)
    output = MPNCOV.SqrtmLayer(output, 5)
    output = MPNCOV.TriuvecLayer(output).squeeze()
    
    output = output.reshape(bs,n,-1)
    f5 = output.clone()

    input_vit = output.reshape(bs,n,-1)
    output_vit = self.transformer(input_vit)
    output_vit = output_vit[:,0,:]
    f = output_vit.clone()

    return f,f1,f2,f3,f4,f5


def initWeights(module):
  if type(module) == nn.Conv2d:
    if module.weight.requires_grad:
      nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

  if type(module) == nn.Linear:
    nn.init.normal_(module.weight.data, mean=0, std=0.01)
    nn.init.constant_(module.bias.data, val=0)
    
