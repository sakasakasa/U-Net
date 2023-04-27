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
from heatmap import save_map 
import seaborn as sns
from statistics import mean
#dir_img = 'data/imgs/Synapse/train_npz/'
#dir_mask = 'data/imgs/Synapse/train_npz/'
dir_checkpoint = 'checkpoints/'
dir_img = '../CVC-ClinicDB/Original/'#'data/imgs/CVC/'
dir_mask = '../CVC-ClinicDB/Ground_Truth/'#'data/masks/CVC/'
#depth_list = [2,3,4,5,6,7]
depth_list_all = [2,3,4,5,6,7]
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
              gamma=1.0,
              IN = True,
              dropout = False
              ):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    print("len",len(dataset))
    train = Subset(dataset,list(range(0,503)))
    val = Subset(dataset,list(range(503,len(dataset))))
    print(len(val))
    #train, val = random_split(dataset, [n_train, n_val])
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
    gamma_list = [0.2,0.4,0.6,0.8,1.0,1.2,1.4]
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
        criterion = dice_loss.DiceLoss()
    sample_depth = 1 if dropout else len(depth_list) 
    for epoch in range(epochs):
        net.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            lip_list = ([[],[],[],[],[],[]])
            lip2_list = ([[],[],[],[],[],[]])
            lip_loss_list = ([[],[],[],[],[],[]])
            term1_list = [[],[],[],[],[],[]]
            term2_list = [[],[],[],[],[],[]]
            term3_list = [[],[],[],[],[],[]]
            sup_list = [[],[],[],[],[],[]]
            sigma_list = [[],[],[],[],[],[]]
            feature_list = [[],[],[],[],[],[]]
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
                
                for i in range(len(depth_list)):
                  net.depth = random.choice(depth_list)#depth_list[i]
                  masks_pred = net(imgs)
                  #print(imgs.shape)
                  #for j in range(len(gamma_list)):
                     #net.set_gamma(gamma_list[j])
 
                  loss = criterion(masks_pred, true_masks)
                  #writer.add_scalar('Loss/train', loss.item(), global_step)

                  pbar.set_postfix(**{'loss (batch)': loss.item()})
                  loss.backward(retain_graph = True)
                  lip_loss = torch.norm(torch.autograd.grad([loss],[net.up3.conv.x],retain_graph = True)[0][0,0,:,:])
                  half_channel = net.up3.conv.x.shape[1]//2
                  lip_loss2 = torch.norm(torch.autograd.grad([loss],[net.up3.conv.x2],retain_graph = True)[0][0,half_channel:,:,:])
                  #print(lip_loss2)
                  #lip2 = torch.norm(torch.autograd.grad([torch.mean(net.up3.conv.out)],[net.up3.conv.x],retain_graph = True)[0][0,0,:,:])
                  lip = torch.norm(torch.autograd.grad([torch.std(torch.sigmoid(masks_pred))],[net.up3.conv.x],retain_graph = True)[0][0,0,:,:])#torch.norm(torch.autograd.grad([loss],[net.up3.conv.x],retain_graph = True)[0][0,0,:,:])
                  term1 = torch.norm(torch.autograd.grad([loss],[net.up3.conv.out],retain_graph = True)[0][0,0,:,:])
                  term2 = (torch.autograd.grad([loss],[net.up3.conv.out],retain_graph = True)[0][0,0,:,:]).sum()
                  term3 = torch.dot(torch.flatten(torch.autograd.grad([loss],[net.up3.conv.out],retain_graph = True)[0][0,0,:,:]),torch.flatten(net.up3.conv.scaled[0,0,:,:]))
                  size = net.up3.conv.size
                  gamma2 = gamma#net.up3.conv.bn1.weight[0] if not IN else gamma
                  #print(lip,term1,net.up4.conv.sigma.mean().item(),term2.item(),term3.item())
                  sup = ((term1**2-(term2**2)*(2/size-1/(size*size))-term3**2/size) *gamma2*gamma2/net.up3.conv.sigma/net.up3.conv.sigma).mean()
                  lip_list[i].append(lip.item()**2)
                  lip2_list[i].append(lip_loss2.item()**2)
                  lip_loss_list[i].append(lip_loss.item()**2)
                  term1_list[i].append(term1.item()**2)
                  term2_list[i].append(term2.item()**2)
                  term3_list[i].append(term3.item()**2)
                  sup_list[i].append(sup.item())
                  sigma_list[i].append(net.up3.conv.sigma[0].item())
                  #feature_list[i].append(net.up3.conv.feature1)

                  #print(torch.norm(net.state_dict()['up3.conv.conv1.0.weight']))
                  #print(net.state_dict().keys())
                  #net.up3.conv.bn1.track_running_stats = False
                  #net.up3.conv.bn2.track_running_stats = False
                  #print(net.up3.bn1.running_var)

                optimizer.step()
                pbar.update(imgs.shape[0])
                global_step += 1

        val_list = np.array([])
        if epoch == epochs-1:
            """
            for i in range(len(depth_list)):
                    print("depth = {}".format(depth_list[i]))
                    #print("lipschitz = {}".format(mean(lip_list[i])))
                    print("lipschitz_loss_half = {}".format(mean(lip2_list[i])))
                    print("lipschitz_loss = {}".format(mean(lip_loss_list[i])))
                    print("term1 = {}".format(mean(term1_list[i])))
                    print("term2 = {}".format(mean(term2_list[i])))
                    print("term3 = {}".format(mean(term3_list[i])))
                    print("sup = {}".format(mean(sup_list[i])))
                    print("sigma = {}".format(mean(sigma_list[i])))
            """
            print("evaluation starts")
        for j in range(len(depth_list_all)):
              net.depth = depth_list_all[j]
              print("depth:{}".format(net.depth))
              gamma_list = [gamma]
              if epoch == epochs-1:
                  for k in range(len(gamma_list)): 
                       net.set_gamma(gamma_list[k])
                       val_score,val_loss = eval_net(net, val_loader, device, print_slim = True if epoch == epochs-1 else False, gamma = gamma,IN = IN)
                       feature[j] = reshape(net.up4.conv.feature)
                       val_list = np.append(val_list,val_score)
                       logging.info('Validation Dice Coeff: {}'.format(val_score))  
                       logging.info('Validation Loss: {}'.format(val_loss))
              else:
                val_score,val_loss = eval_net(net, val_loader, device, print_slim = True if epoch == epochs-1 else False, gamma = gamma,IN = IN)
                val_list = np.append(val_list,val_score)
                logging.info('Validation Dice Coeff: {}'.format(val_score))  
                logging.info('Validation Loss: {}'.format(val_loss))
              net.mult_change(False)
    #writer.close()
    return [val_list]


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
    parser.add_argument('-g', '--gamma', dest='gamma', type=float, default=1.0,
                        help='gamma')
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
    #net = UNet(n_channels=3, n_classes=1, bilinear=True,depth = 5,img_h = size[1],img_w = size[2],gamma = args.gamma)
    #logging.info(f'Network:\n'
    #             f'\t{net.n_channels} input channels\n'
    #             f'\t{net.n_classes} output channels (classes)\n'
    #             f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
    
    #net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    result = np.array([])
    scale_result = np.array([])
    depth_list = [2,3,4,5,6,7] if args.depth == "all" else [int(args.depth)] 
    #feature = [[],[],[],[],[],[]]
    print("start")
    try:
      for i in range(1):
        #print("start")
        #depth_list = [i+2]
        net = UNet(n_channels=3, n_classes=1, bilinear=True,depth = 5,img_h = size[1],img_w = size[2],gamma = args.gamma, IN = args.BN) 
        net.to(device=device)
        feature_list = [net.up7,net.up6,net.up1,net.up2,net.up3,net.up4]
        feature_down = [net.down1,net.down2,net.down3,net.down4,net.down5,net.down6]
        total = 0
        params = 0
        for p in net.inc.parameters():
               if p.requires_grad:
                 params += p.numel()
        print(params)
        total += params

        for features in feature_down:
            params = 0
            for p in features.parameters():
               if p.requires_grad:
                 params += p.numel()
            print(params)
            total += params
        for features in feature_list:
            params = 0
            for p in features.conv.parameters():
               if p.requires_grad:
                 params += p.numel()
            print(params)
            total += params
        params = 0
        for p in net.outc.parameters():
               if p.requires_grad:
                 params += p.numel()
        print(params)
        total += params
        print("total=",total)


        """
        for p in feature_list[4].conv.parameters():
            print(p.shape,p.numel())
        print("up4")
        for p in feature_list[5].conv.parameters():
            print(p.shape,p.numel())
        """
        new_result =  train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  gamma = args.gamma,
                  IN = args.BN
                  dropout = args.dropout
                  )
    
        #feature_list = [net.up7,net.up6,net.up1,net.up2,net.up3,net.up4]
        #feature_down = [net.down1,net.down2,net.down3,net.down4,net.down5,net.down6]
        print("params")
        for i in range(len(feature_down)):
            print(feature_down[i].params)
        for i in range(len(feature_list)):
            print(feature_list[i].params)

        """
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
        """
        #final_layer = net.up3.conv
        incremental_cka = cka.IncrementalCKA(6,6)
        
        #for j in range(6):
          #print(j)
         #  feature[j] = reshape(final_layer.feature[j])
        #print(feature[i])
      for index_x in range(6):
                for index_y in range(6):
                    incremental_cka.increment_cka_score(index_x, index_y, feature[index_x],feature[index_y]
                    )
      save_map(incremental_cka.cka()[::-1],args.BN,args.depth)
      print("finished!!")
      
    except KeyboardInterrupt:
         torch.save(net.state_dict(), 'INTERRUPTED.pth')
         logging.info('Saved interrupt')
         try:
            sys.exit(0)
         except SystemExit:
            os._exit(0)
