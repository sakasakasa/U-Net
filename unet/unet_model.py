""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, depth = 5,img_h = None,img_w = None,gamma = 1.0,IN = True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64,img_h = img_h,img_w = img_w,IN = IN)
        self.down1 = Down(64, 128,img_h//2,img_w//2,gamma = gamma,IN = IN)
        self.down2 = Down(128, 256,img_h//4,img_w//4,gamma = gamma,IN = IN)
        self.down3 = Down(256, 512,img_h//8,img_w//8,gamma = gamma,IN = IN)
        factor = 1
        self.down4 = Down(512, 1024 // factor,img_h//16,img_w//16,gamma = gamma,IN = IN)
        self.down5 = Down(1024, 1024 // factor,img_h//32,img_w//32,gamma = gamma, IN = IN)
        self.down6 = Down(1024,1024  // factor,gamma = gamma, IN = IN)
        self.up7 = Up(1024, 1024 // factor, bilinear,conv_channels = 1024,gamma = gamma, IN = IN)
        self.up6 = Up(1024, 1024 // factor, bilinear,conv_channels = 512,gamma = gamma, IN = IN)
        self.up1 = Up(1024, 512 // factor, img_h = img_h//8,img_w = img_w//8, bilinear = bilinear,conv_channels = 512,gamma =gamma, IN = IN)
        self.up2 = Up(512, 256 // factor, img_h = img_h//4,img_w = img_w//4,bilinear = bilinear,conv_channels = 256,gamma = gamma, IN = IN)
        self.up3 = Up(256, 128 // factor, img_h = img_h//2,img_w = img_w//2,bilinear = bilinear,conv_channels = 128,gamma = gamma, IN = IN)
        self.up4 = Up(128, 64, img_h = img_h,img_w = img_w,bilinear = bilinear ,conv_channels = 64,gamma = gamma, IN = IN)
        self.outc = OutConv(64, n_classes)
        self.depth = depth
        self.img_h = img_h
        self.img_w = img_w
    def forward(self, x):
      if self.depth == 7:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x = self.up7(x7, x6)
        x = self.up6(x, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
      if self.depth == 6:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        x_dummy = np.zeros(x7.shape)#((16,1024,9,12))
        x_dummy = x_dummy.astype(np.float32)
        x_tensor = torch.from_numpy(x_dummy).clone().cuda()
        x = self.up7(x_tensor, x6)
        x = self.up6(x, x5)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
 
      if self.depth == 5:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x_dummy = np.zeros(x6.shape)#((16,1024,9,12))
        x_dummy = x_dummy.astype(np.float32)
        x_tensor = torch.from_numpy(x_dummy).clone().cuda()
        x_dummy2 = np.zeros(x4.shape)#((16,1024,9,12))
        x_dummy2 = x_dummy2.astype(np.float32)
        x_tensor2 = torch.from_numpy(x_dummy2).clone().cuda()
        x_dummy3 = np.zeros(x3.shape)#((16,1024,9,12))
        x_dummy3 = x_dummy3.astype(np.float32)
        x_tensor3 = torch.from_numpy(x_dummy3).clone().cuda()
        x_dummy4 = np.zeros(x2.shape)#((16,1024,9,12))
        x_dummy4 = x_dummy4.astype(np.float32)
        x_tensor4 = torch.from_numpy(x_dummy4).clone().cuda()
        x_dummy5 = np.zeros(x1.shape)#((16,1024,9,12))
        x_dummy5 = x_dummy5.astype(np.float32)
        x_tensor5 = torch.from_numpy(x_dummy5).clone().cuda()

        x = self.up6(x_tensor,x5)
        x = self.up1(x,x4)#x_tensor2)#x4)
        x = self.up2(x, x3)#x_tensor3)#x3)
        x = self.up3(x, x2)#x_tensor4)#x2)
        x = self.up4(x, x1)#x_tensor5)#x1)
        logits = self.outc(x)
        y = logits.cpu()
        return logits 
      elif self.depth == 4:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_dummy = np.zeros(x5.shape)#((16,1024,9,12))
        x_dummy = x_dummy.astype(np.float32)
        x_tensor = torch.from_numpy(x_dummy).clone().cuda()
        x = self.up1(x_tensor,x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        y = logits.cpu()
        return logits 
      elif self.depth == 3:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x_dummy = np.zeros(x4.shape)#((16,1024,9,12))
        x_dummy = x_dummy.astype(np.float32)
        x_tensor = torch.from_numpy(x_dummy).clone().cuda()
        x = self.up2(x_tensor, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        y = logits.cpu()
        return logits 
      else:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x_dummy = np.zeros(x3.shape)
        x_dummy = x_dummy.astype(np.float32)
        x_tensor = torch.from_numpy(x_dummy).clone().cuda()
        x = self.up3(x_tensor, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
