import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet
import random
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
import dice_loss
from torch.utils.data.dataset import Subset
import cka
from heatmap import save_map,save_map_dropout
import seaborn as sns
from statistics import mean
dir_checkpoint = 'checkpoints/'
dir_img = '../CVC-ClinicDB/Original/'
dir_mask = '../CVC-ClinicDB/Ground_Truth/'
depth_list_all = [2,3,4,5,6,7]
feature = [[],[],[],[],[],[]]
iter_num = 1
def reshape(x):
            return np.reshape(x,(x.shape[0],-1))
def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.2,
              save_cp=True,
              img_scale=0.5,
              IN = True,
              dropout = False
              ):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train = Subset(dataset,list(range(0,n_train)))
    val = Subset(dataset,list(range(n_train,len(dataset))))
   
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.SGD(net.parameters(),lr = 0.01,momentum = 0.9)
    sample_depth = 1 if dropout else len(depth_list)
    criterion = dice_loss.DiceLoss() 
    for epoch in range(epochs):
        net.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                optimizer.zero_grad()
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                
                for i in range(sample_depth):
                  #structured dropout
                  net.depth = depth_list[i] if not dropout else random.choice(depth_list)
                  masks_pred = net(imgs)
                  loss = criterion(masks_pred, true_masks)

                  pbar.set_postfix(**{'loss (batch)': loss.item()})
                  loss.backward(retain_graph = True)
                optimizer.step()
                pbar.update(imgs.shape[0])
                global_step += 1

        val_list = np.array([])
        for j in range(len(depth_list_all)):
              net.depth = depth_list_all[j]
              print("depth:{}".format(net.depth))
              val_score,val_loss,feature_mean,feature_std = eval_net(net, val_loader, device, IN = IN)
              feature[j] = reshape(net.up4.conv.feature)
              val_list = np.append(val_list,val_score)
              #print(depth_list_all,val_list)
              logging.info('Validation Dice Coeff: {}'.format(val_score))  
              logging.info('Validation Loss: {}'.format(val_loss))            
    return val_list

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=16,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--BN',  dest = "BN",default = "BN")
    parser.add_argument('--d',  dest = "depth",default = "all")
    parser.add_argument('--dr',  dest = "dropout",default = False)


    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    dataset = BasicDataset(dir_img, dir_mask, args.scale)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    for batch in train_loader:
       size = batch['image'].shape[1:]
       break
    result_sum = []
    
    depth_list = [2,3,4,5,6,7] if args.depth == "all" else [int(args.depth)] 
    try:
      for i in range(iter_num):
        net = UNet(n_channels=3, n_classes=1, bilinear=True,depth = 5,img_h = size[1],img_w = size[2], IN = args.BN) 
        net.to(device=device)
        feature_list = [net.up7,net.up6,net.up1,net.up2,net.up3,net.up4]
        feature_down = [net.down1,net.down2,net.down3,net.down4,net.down5,net.down6]
    
        new_result =  train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  IN = args.BN,
                  dropout = args.dropout
                  )
        print(new_result)
        result_sum +=[new_result]
        
        incremental_cka = cka.IncrementalCKA(6,6)
        
      for index_x in range(6):
                for index_y in range(6):
                    incremental_cka.increment_cka_score(index_x, index_y, feature[index_x],feature[index_y]
                    )
      #if args.dropout: 
      #   save_map_dropout(incremental_cka.cka()[::-1],args.BN,args.depth)
      #else:
      #    save_map(incremental_cka.cka()[::-1],args.BN,args.depth)
      result_sum = np.array(result_sum)
      result_sum = result_sum.T
      print(result_sum)
      print((np.mean(result_sum,axis = 1)*100),np.std(result_sum,axis = 1))

     
      
    except KeyboardInterrupt:
         torch.save(net.state_dict(), 'INTERRUPTED.pth')
         logging.info('Saved interrupt')
         try:
            sys.exit(0)
         except SystemExit:
            os._exit(0)
