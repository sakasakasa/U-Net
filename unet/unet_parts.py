""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None,batchnorm = True, p = False, img_h =1,img_w = 1):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        """
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        """
        if True:#batchnorm:
          self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
           )
          self.conv3 = nn.Sequential(
            nn.BatchNorm2d(mid_channels),
            #nn.InstanceNorm2d(mid_channels),
            #nn.LayerNorm([mid_channels,img_h,img_w]),
            nn.ReLU(inplace=True)
          )
        else: 
          self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
          )
        if True:#batchnorm:
          """
          self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            #nn.InstanceNorm2d(mid_channels),
            #nn.LayerNorm([mid_channels,img_h,img_w]),
            nn.ReLU(inplace=True)
          )
          """
          self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
           )
          self.conv4 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            #nn.InstanceNorm2d(out_channels),
            #nn.LayerNorm([out_channels,img_h,img_w]),
            nn.ReLU(inplace=True)
          )

        else:
           self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1), 
            nn.ReLU(inplace=True) 
          ) 
        self.depth = 0
        self.p = False
        self.p1 = False
        self.p2 = False
        self.p3 = False
        self.p4 = False
        self.feature = ([[],[],[],[],[],[]])#[self.feature3,self.feature4,self.feature5,self.feature6,self.feature7]
    def forward(self, x):
        if self.p1:
              self.feature  = np.concatenate([self.feature,x.cpu().detach().numpy()]) if len(self.feature)!=0 else x.cpu().detach().numpy()
 
        out = self.conv1(x)
        out = self.conv3(out)
        if self.p2:
              self.feature  = np.concatenate([self.feature,out.cpu().detach().numpy()]) if len(self.feature)!=0 else out.cpu().detach().numpy()

        """ 
        if self.test is not None:
          mask = self.test*np.ones((out.shape[0],out.shape[1],out.shape[2],out.shape[3]))
          mask_tensor = torch.from_numpy(mask).clone()
          mask_tensor = mask_tensor.to("cuda")
          out = out*mask_tensor.float()
        """  
        out = self.conv2(out)
        out = self.conv4(out)
        if self.p3:
              self.feature  = np.concatenate([self.feature,out.cpu().detach().numpy()]) if len(self.feature)!=0 else out.cpu().detach().numpy()
        if self.p4: 
            print(out)
        if self.p:
            self.feature[self.depth] = out.cpu().detach().numpy()
        
        return out

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels,img_h=1 , img_w=1,batchnorm = True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels,img_h=img_h,img_w=img_w,batchnorm = batchnorm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True,conv_channels = 0, mid_channels = None,img_h = 1,img_w = 1,batchnorm = True, p = False):
        super().__init__()
        in_channels2 = in_channels if conv_channels == 0 else conv_channels
        self.conv1 = nn.Conv2d(in_channels,in_channels//2, kernel_size=3, padding=1)
        self.norm = nn.Sequential(
            nn.BatchNorm2d(in_channels//2),
            #nn.InstanceNorm2d(mid_channels),
            #nn.LayerNorm([mid_channels,img_h,img_w]),
            nn.ReLU(inplace=True)
          )
        self.p = False
        self.scale = 1
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels2, out_channels, in_channels // 2,img_h = img_h,img_w = img_w,batchnorm = batchnorm, p = p)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,mid_channels = None,img_h = img_h,img_w = img_w,batchnorm = batchnorm)


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
        x = torch.cat([x2, (self.conv1(x1))*self.scale], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
