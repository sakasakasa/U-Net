""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,  img_h =1,img_w = 1, IN = "BN"):
        super().__init__()
        
        self.IN = IN
        if not mid_channels:
            mid_channels = out_channels
          
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        )
        if self.IN == "BN":
            self.bn1_1 = nn.BatchNorm2d(mid_channels,affine = False,track_running_stats =True)
            self.bn1_2 = nn.BatchNorm2d(mid_channels,affine = False,track_running_stats =True)
            self.bn1_3 = nn.BatchNorm2d(mid_channels,affine = False,track_running_stats =True)
            self.bn1_4 = nn.BatchNorm2d(mid_channels,affine = False,track_running_stats =True)
            self.bn1_5 = nn.BatchNorm2d(mid_channels,affine = False,track_running_stats =True)
            self.bn1_6 = nn.BatchNorm2d(mid_channels,affine = False,track_running_stats =True)
            self.norm1 = [self.bn1_1,self.bn1_2,self.bn1_3,self.bn1_4,self.bn1_5,self.bn1_6]
        elif self.IN == "IN":
            self.norm1 = nn.InstanceNorm2d(mid_channels)
        else:
            self.norm1 = nn.LayerNorm((mid_channels,img_h,img_w))
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        )
        
        if self.IN == "BN":
            self.bn2_1 = nn.BatchNorm2d(out_channels,affine = False,track_running_stats =True)
            self.bn2_2 = nn.BatchNorm2d(out_channels,affine = False,track_running_stats =True)
            self.bn2_3 = nn.BatchNorm2d(out_channels,affine = False,track_running_stats =True)
            self.bn2_4 = nn.BatchNorm2d(out_channels,affine = False,track_running_stats =True)
            self.bn2_5 = nn.BatchNorm2d(out_channels,affine = False,track_running_stats =True)
            self.bn2_6 = nn.BatchNorm2d(out_channels,affine = False,track_running_stats =True)
            self.norm2 = [self.bn2_1,self.bn2_2,self.bn2_3,self.bn2_4,self.bn2_5,self.bn2_6]
        elif self.IN == "IN":
            self.norm2 = nn.InstanceNorm2d(out_channels)
        else:
            self.norm2 = nn.LayerNorm((out_channels,img_h,img_w))
    def forward(self, x,depth):
        x = x.requires_grad_()
        self.x2 = x
        out = self.conv1(x)
        self.x =out 
        self.sigma = out.std(axis = (0,2,3))
        self.mean = out.mean(axis = (0,2,3))
        self.size = out[:,0,:,:].flatten().shape[0] if self.IN == False else out[0,0,:,:].flatten().shape[0]
        out = self.norm1(out) if self.IN != "BN" else self.norm1[depth-2](out)
        self.out = out
        out = self.relu(out)
        self.feature1 = out
        out = self.conv2(out)
        out = self.norm2(out) if self.IN != "BN" else self.norm2[depth-2](out)
        out = self.relu(out)
        self.feature2 = out
        self.feature = out.cpu().detach().numpy()
        self.out_norm = torch.norm(out)
        return out

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,img_h=1 , img_w=1, IN = True):
        super().__init__()
     
        self.pool =  nn.MaxPool2d(2)
        self.conv =  DoubleConv(in_channels, out_channels,img_h=img_h,img_w=img_w, IN = IN)
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels,img_h=img_h,img_w=img_w,IN = IN)
        )
    def forward(self, x, depth):
        out = self.pool(x)
        out = self.conv(out,depth)
        return out


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True,img_h = 1,img_w = 1, IN = True):
        super().__init__()
    
        self.conv1 = nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=1)
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(out_channels, out_channels, in_channels // 2,img_h = img_h,img_w = img_w, IN = IN)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,mid_channels = None,img_h = img_h,img_w = img_w, IN = IN)



    def forward(self, x1,x2,depth):
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
        return self.conv(x,depth)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
