""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,batchnorm = True, gamma = 1.0, img_h =1,img_w = 1):
        super().__init__()
        self.gamma = gamma
        if not mid_channels:
            mid_channels = out_channels
        if True:#batchnorm:
          
          self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            #nn.InstanceNorm2d(mid_channels),
            nn.BatchNorm2d(mid_channels,affine = False),
           )
          
          #self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
          self.conv3 = nn.Sequential(
            #nn.BatchNorm2d(mid_channels,affine = False),
            #nn.InstanceNorm2d(mid_channels),
            #nn.LayerNorm([mid_channels,img_h,img_w]),
            #nn.ReLU(inplace=True)
          )
          self.relu = nn.ReLU(inplace=True)
        if True:#batchnorm:
          self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            #nn.InstanceNorm2d(out_channels),
            nn.BatchNorm2d(out_channels,affine = False),
           )
          self.conv4 = nn.Sequential(
            #nn.BatchNorm2d(out_channels),
            #nn.InstanceNorm2d(out_channels),
            #nn.LayerNorm([out_channels,img_h,img_w]),
            #nn.ReLU(inplace=True)
          )

    def forward(self, x):
        x = x.requires_grad_()
        out = self.conv1(x)
        out *= self.gamma
        self.x = x
        self.sigma = out.std(axis = (0,2,3))
        #self.jacobi = torch.stack([torch.autograd.grad([y[i].sum()], [x], retain_graph=True, create_graph=True, allow_unused = True)[0] for i in range(y.size(0))], dim=-1).squeeze().t()
        out = self.conv3(out)
        self.out = out
        out = self.relu(out)
        self.size = out[:,0,:,:].flatten().shape[0]
        #self.scaled =(out-self.conv3[0].running_mean)/np.sqrt(self.conv3[0].running_var)
        out = out.requires_grad_(True)
        out = self.conv2(out)
        #out *= self.gamma
        #self.x = x
        out = self.conv4(out)
        #self.sigma = out.std(axis = (0,2,3))
        #self.out = out
        return out

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,img_h=1 , img_w=1,batchnorm = True,gamma = 1.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels,img_h=img_h,img_w=img_w,batchnorm = batchnorm,gamma = gamma)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True,conv_channels = 0, mid_channels = None,img_h = 1,img_w = 1,batchnorm = True,gamma = 1.0):
        super().__init__()
        in_channels2 = in_channels if conv_channels == 0 else conv_channels
        self.conv1 = nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=1)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(in_channels//2),
            #nn.InstanceNorm2d(mid_channels),
            #nn.LayerNorm([mid_channels,img_h,img_w]),
            nn.ReLU(inplace=True)
          )
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(out_channels, out_channels, in_channels // 2,img_h = img_h,img_w = img_w,batchnorm = batchnorm,gamma = gamma)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,mid_channels = None,img_h = img_h,img_w = img_w,batchnorm = batchnorm,gamma = gamma)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.add(x2, (self.conv1(x1)))
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
