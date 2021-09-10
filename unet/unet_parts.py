""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
        self.test = None
        self.p = p
        self.feature = None
        self.train_mean = None
        self.depth = 0
        self.feature3 = None
        self.feature5 = None
        self.feature3_test = None
        self.feature5_test = None
    def forward(self, x):
        """ 
        if self.p:
           if not self.test:
            print(np.mean(x.abs().cpu().detach().numpy(),axis = (0,2,3)))
        
        if self.p:
            #print(np.mean(x.cpu().detach().numpy(),axis = (0,2,3)))
            print("weight",self.conv1[0].weight.abs().mean(axis = [0,2,3]))
        """
        out = self.conv1(x)
        #if self.p:
         #    self.train_mean = out.mean(axis = [0,2,3])
         
        if self.p:
           if not self.test:
            print(np.mean(out.abs().cpu().detach().numpy(),axis = (0,2,3)))
            if self.depth == 3:
              self.feature3  = np.std(out.cpu().detach().numpy(),axis = (0,2,3))
            elif self.depth == 5:
              self.feature5  = np.std(out.cpu().detach().numpy(),axis = (0,2,3))
           else:
            #print(np.std(out.cpu().detach().numpy(),axis = (0,2,3)))
            if self.depth == 3:
              self.feature3_test  = np.std(out.cpu().detach().numpy(),axis = (0,2,3))
            elif self.depth == 5:
              self.feature5_test  = np.std(out.cpu().detach().numpy(),axis = (0,2,3))
            #self.feature = np.std(out.cpu().detach().numpy(),axis = (0,2,3))
        
        out = self.conv3(out)
        
        if self.p:
           #print(np.sqrt(self.conv3[0].running_var.cpu()))
           self.var = np.sqrt(self.conv3[0].running_var.cpu())
           self.mean = (self.conv3[0].running_mean.cpu())
       
        """ 
        if self.test is not None:
          mask = self.test*np.ones((out.shape[0],out.shape[1],out.shape[2],out.shape[3]))
          mask_tensor = torch.from_numpy(mask).clone()
          mask_tensor = mask_tensor.to("cuda")
          out = out*mask_tensor.float()
        """  
        out = self.conv2(out)
        """
        if self.p:
           if not self.test:
            print(np.std(out.cpu().detach().numpy(),axis = (0,2,3)))
            if self.depth == 3:
              self.feature3  = np.mean(out.cpu().detach().numpy(),axis = (0,2,3))
            elif self.depth == 5:
              self.feature5  = np.mean(out.cpu().detach().numpy(),axis = (0,2,3))
           else:
            print(np.std(out.cpu().detach().numpy(),axis = (0,2,3)))
            if self.depth == 3:
              self.feature3_test  = np.mean(out.cpu().detach().numpy(),axis = (0,2,3))
            elif self.depth == 5:
              self.feature5_test  = np.mean(out.cpu().detach().numpy(),axis = (0,2,3))
        """
        out = self.conv4(out)
        
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
    def __init__(self, in_channels, out_channels, bilinear=True,conv_channels = True, mid_channels = None,img_h = 1,img_w = 1,batchnorm = True, p = False):
        super().__init__()
        in_channels2 = in_channels if conv_channels else 1536
        self.conv1 = nn.Conv2d(in_channels,in_channels//2, kernel_size=3, padding=1)
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(in_channels//2),
            #nn.InstanceNorm2d(mid_channels),
            #nn.LayerNorm([mid_channels,img_h,img_w]),
            nn.ReLU(inplace=True)
          )
        self.p = False
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
        #print(x2.shape,self.conv1(x1).shape)
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, (self.conv1(x1))], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
