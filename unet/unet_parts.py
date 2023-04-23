""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,batchnorm = True, gamma = 1.0, img_h =1,img_w = 1, IN = True):
        super().__init__()
        self.gamma = gamma
        self.IN = IN
        if not mid_channels:
            mid_channels = out_channels
          
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        )
        self.bn1 = nn.LayerNorm((mid_channels,img_h,img_w))#nn.BatchNorm2d(mid_channels,affine = False,track_running_stats = True) if IN == False else nn.InstanceNorm2d(mid_channels)  
        #self.bn1 = nn.BatchNorm2d(mid_channels,affine = False,track_running_stats = True) if IN == False else nn.InstanceNorm2d(mid_channels)  
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )
        self.bn2 = nn.LayerNorm((out_channels,img_h,img_w))#nn.BatchNorm2d(out_channels,affine = False,track_running_stats =True)if IN == False else nn.InstanceNorm2d(out_channels)
        #self.bn2 = nn.BatchNorm2d(out_channels,affine = False,track_running_stats = True) if IN == False else nn.InstanceNorm2d(out_channels)     

    def forward(self, x):
        x = x.requires_grad_()
        half_channel = x.shape[1]//2
        self.x2 = x
        out = self.conv1(x)
        self.x =out 
        self.sigma = out.std(axis = (0,2,3))
        self.mean = out.mean(axis = (0,2,3))
        self.size = out[:,0,:,:].flatten().shape[0] if self.IN == False else out[0,0,:,:].flatten().shape[0]
        out = self.bn1(out)
        #self.feature1 = out
        self.scaled =(out-torch.broadcast_to(self.bn1.bias[None,:,None,None],out.shape))/(torch.broadcast_to(self.bn1.weight[None,:,None,None],out.shape)) if False else out
        out = out* self.gamma #if self.IN else out
        self.out = out
        out = self.relu(out)
        self.feature1 = out
        out = self.conv2(out)
        out = self.bn2(out)
        #self.scaled =(out-torch.broadcast_to(self.bn2.bias[None,:,None,None],out.shape))/(torch.broadcast_to(self.bn2.weight[None,:,None,None],out.shape)) if False else out
        out = out* self.gamma #if self.IN else out
        out = self.relu(out)
        self.feature2 = out
        self.feature = out.cpu().detach().numpy()
        self.out_norm = torch.norm(out)
        return out

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,img_h=1 , img_w=1,batchnorm = True,gamma = 1.0, IN = True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels,img_h=img_h,img_w=img_w,batchnorm = batchnorm,gamma = gamma, IN = IN)
        )
        self.params = 0
        for p in self.maxpool_conv.parameters():
            if p.requires_grad:
              self.params += p.numel()
        #print(self.params)
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True,conv_channels = 0, mid_channels = None,img_h = 1,img_w = 1,batchnorm = True,gamma = 1.0, IN = True,coef = 1.0):
        super().__init__()
        in_channels2 = in_channels if conv_channels == 0 else conv_channels
    
        self.conv1 = nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=1)
        """
        self.coef = coef
        self.norm = nn.Sequential(
            nn.BatchNorm2d(in_channels//2),
            #nn.InstanceNorm2d(mid_channels),
            #nn.LayerNorm([mid_channels,img_h,img_w]),
            nn.ReLU(inplace=True)
          )
        """
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(out_channels, out_channels, in_channels // 2,img_h = img_h,img_w = img_w,batchnorm = batchnorm,gamma = gamma, IN = IN)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,mid_channels = None,img_h = img_h,img_w = img_w,batchnorm = batchnorm,gamma = gamma, IN = IN)
        self.mult = False
        self.params = 0
        #for p in self.conv.parameters():
         #   if p.requires_grad:
          #    self.params += p.numel()


    def forward(self, x1,x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.add(x2, (self.conv1(x1)))*self.coef if self.mult else torch.add(x2, (self.conv1(x1)))
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
