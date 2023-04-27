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
from utils.dataset_synapse import BasicDataset
from torch.utils.data import DataLoader, random_split
import dice_loss
from torch.utils.data.dataset import Subset
import cka
from heatmap import save_map_synapse
import seaborn as sns
#dir_img = 'data/imgs/Synapse/train_npz/'
#dir_mask = 'data/imgs/Synapse/train_npz/'
dir_checkpoint = 'checkpoints/'
dir_img = '../Synapse/train_npz/'
dir_mask = '../Synapse/train_npz/'
depth_list_all = [2,3,4,5,6,7]
depth_list = [2,3,4,5,6,7]
feature = [[],[],[],[],[],[]]
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
              dropout = False):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    num = [117,131,163,149,148,142,96,124,131,88,89,153,93,104,98,99,90,195]
    sum_num = 0
    train_list = np.array([])
    val_list = np.array([])
    for i in range(10):
      count = 0
      for j in range(num[i]):
        if dataset[sum_num+j]["mask"].sum()>0:
          train_list = np.append(train_list,sum_num+j)#np.append(train_list,np.array(list(map(int,range(sum_num, sum_num + 50)))))
          count += 1
      sum_num += int(num[i])
    train_list = [int(i) for i in train_list]
    n_train = len(train_list)
    sample_depth = 1 if dropout else len(depth_list)

    for i in range(10,16):
      count = 0
      for j in range(num[i]):
        if dataset[sum_num+j]["mask"].sum()>0:
          val_list = np.append(val_list,int(sum_num+j))
          count += 1
      sum_num += num[i]
    val_list = [int(i) for i in val_list]

    n_val = int(len(val_list))
    #n_train = len(dataset) - n_val
    train = Subset(dataset,train_list)
    val = Subset(dataset,val_list)

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    feature_list = [net.up7,net.up6,net.up1,net.up2,net.up3,net.up4]
    feature_down = [net.down1,net.down2,net.down3,net.down4,net.down5,net.down6]
    #writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
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

    #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    optimizer = optim.SGD(net.parameters(),lr = 0.01,momentum = 0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
        criterion = dice_loss.DiceLoss()
        criterion2 = dice_loss.L1_Loss()
    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
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
                  net.depth = depth_list[i]
                  #print(net.depth)
                  masks_pred = net(imgs)
 
                  loss = criterion(masks_pred, true_masks)#+(1E-8)*criterion2(teacher,student)
                  epoch_loss += loss.item()
                  #writer.add_scalar('Loss/train', loss.item(), global_step)

                  pbar.set_postfix(**{'loss (batch)': loss.item()})
                  loss.backward()

                optimizer.step()
                pbar.update(imgs.shape[0])
                global_step += 1
        scale_list = np.array([])
         
        for j in range(len(depth_list_all)):
              net.depth = depth_list_all[j]
              val_score,val_loss = eval_net(net, val_loader, device)
              scale_list = np.append(scale_list,val_score)
              logging.info('Validation Dice Coeff: {}'.format(val_score))  
              print("depth:{}".format(net.depth))
              if epoch == epochs-1:
                       val_score,val_loss = eval_net(net, val_loader, device, print_slim = True if epoch == epochs-1 else False)
                       print((reshape(net.up4.conv.feature)).shape)
                       feature[j] = reshape(net.up4.conv.feature)
                       val_list = np.append(val_list,val_score)
                       logging.info('Validation Dice Coeff: {}'.format(val_score))
                       logging.info('Validation Loss: {}'.format(val_loss))
              else:
                val_score,val_loss = eval_net(net, val_loader, device, print_slim = True if epoch == epochs-1 else False)
                val_list = np.append(val_list,val_score)
                logging.info('Validation Dice Coeff: {}'.format(val_score))
                logging.info('Validation Loss: {}'.format(val_loss))
              net.mult_change(False)

    #writer.close()
    return [scale_list,scale_list]


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
    parser.add_argument('-g', '--gamma', dest='gamma', type=float, default=1,
                        help='gamma')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--BN',  dest = "BN",default = "BN")
    parser.add_argument('--d',  dest = "depth",default = "all")
    parser.add_argument('--dr',  dest = "dropout",default = False)



    return parser.parse_args()


if __name__ == '__main__':
    print("BN-6 individuals")
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
    #net = UNet(n_channels=1, n_classes=1, bilinear=True,depth = 5,img_h = size[1],img_w = size[2])
    #logging.info(f'Network:\n'
    #             f'\t{net.n_channels} input channels\n#'
    #             f'\t{net.n_classes} output channels (classes)\n'
    #            f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')
    """
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
    """
    #net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    result = np.array([])
    scale_result = np.array([])
    depth_list = [2,3,4,5,6,7] if args.depth == "all" else [int(args.depth)]

    #feature = np.array([[],[],[],[],[],[]])
    try:
      for i in range(1):
        #depth_list = [2,3,4,5,6,7]
        net = UNet(n_channels=1, n_classes=1, bilinear=True,depth = 5,img_h = size[1],img_w = size[2],gamma = args.gamma,IN= args.BN) 
        net.to(device=device)
        new_result =  train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  dropout = args.dropout)
        feature_list = [net.up7,net.up6,net.up1,net.up2,net.up3,net.up4]
        feature_down = [net.down1,net.down2,net.down3,net.down4,net.down5,net.down6]
        def feature_func(num):
           if num == 0: 
               feature_temp = net.inc.feature
               return np.reshape(feature_temp,(feature_temp.shape[0],-1))
           elif num < 7:
               feature_temp = feature_down[num-1].maxpool_conv[1].feature
               return np.reshape(feature_temp,(feature_temp.shape[0],-1))
           else:
               feature_temp = feature_list[num-7].conv.feature
               return np.reshape(feature_temp,(feature_temp.shape[0],-1))
        def reshape(x):
            return np.reshape(x,(x.shape[0],-1))
        
        final_layer = net.up4.conv
        incremental_cka = cka.IncrementalCKA(6,6)
        #feature[i] = (reshape(final_layer.feature[0]))
      
      for index_x in range(6):
                for index_y in range(6):
                    incremental_cka.increment_cka_score(index_x, index_y, feature[index_x],feature[index_y]
                    )
      save_map_synapse(incremental_cka.cka()[::-1],args.BN,args.depth)
      print("finished!!")
      
    except KeyboardInterrupt:
         torch.save(net.state_dict(), 'INTERRUPTED.pth')
         logging.info('Saved interrupt')
         try:
            sys.exit(0)
         except SystemExit:
            os._exit(0)
