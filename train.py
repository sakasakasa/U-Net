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
#dir_img = 'data/imgs/Synapse/train_npz/'
#dir_mask = 'data/imgs/Synapse/train_npz/'
dir_checkpoint = 'checkpoints/'
dir_img = '../CVC-ClinicDB/Original/'#'data/imgs/CVC/'
dir_mask = '../CVC-ClinicDB/Ground_Truth/'#'data/masks/CVC/'
depth_list_all = [2,3,4,5,6,7]
depth_list = []
def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.2,
              save_cp=True,
              img_scale=0.5):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train = Subset(dataset,list(range(0,503)))
    val = Subset(dataset,list(range(503,len(dataset))))
    #train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    feature_list = [net.up7,net.up6,net.up1,net.up2,net.up3,net.up4]
    feature_down = [net.down1,net.down2,net.down3,net.down4,net.down5,net.down6]
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
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
    val_loss_list3 = np.array([])
    val_loss_list4 = np.array([])
    val_loss_list5 = np.array([])
    val_loss_list6 = np.array([])
    val_loss_list7 = np.array([])
    val_loss_list = [val_loss_list3,val_loss_list4,val_loss_list5,val_loss_list6,val_loss_list7]
    #val_loss_list = val_loss_list[:3]
    for epoch in range(epochs):
        net.train()
        """
        for j in range(6):
          feature_list[j].conv.feature  = []
          feature_down[j].maxpool_conv[1].feature  = []
        """
        epoch_loss = 0
        loss_list = np.zeros(len(depth_list))
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
                
                for i in range(len(depth_list)):
                  net.depth = depth_list[i]
                  if epoch == epochs-1:
                    #print(net.up4.conv.feature)
                    #feature_list[5].conv.depth  = i#depth_list_all[i]
                    feature_list[5].conv.p = True
                    """
                  if i == (len(depth_list)-1):
                   if epoch == epochs-1:
                     for j in range(6):
                       if j > 0:
                         feature_down[j].maxpool_conv[1].p3 = True
                       else:
                         feature_down[j].maxpool_conv[1].p2 = True
                       feature_list[j].conv.p3 = True
                     net.inc.p3 = True
                  else:
                      for j in range(6):
                        if j > 0:
                          feature_down[j].maxpool_conv[1].p3 = False
                        else:
                          feature_down[j].maxpool_conv[1].p2 = False
                        feature_list[j].conv.p3 = False
                      net.inc.p3 = False
                  """
                  masks_pred = net(imgs)
 
                  loss = criterion(masks_pred, true_masks)#+(1E-8)*criterion2(teacher,student)
                  loss_list[i] += loss
                  epoch_loss += loss.item()
                  writer.add_scalar('Loss/train', loss.item(), global_step)

                  pbar.set_postfix(**{'loss (batch)': loss.item()})
                #optimizer.zero_grad()
                  loss.backward()

                #nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                pbar.update(imgs.shape[0])
                global_step += 1
                """
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info('Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
                """
        scale_list = np.array([])
        #net.up1.scale = 1/3
        #net.up2.scale = 2/3
        """ 
        for j in range(len(depth_list)):
              net.depth = depth_list[j]
              val_score,val_loss = eval_net(net, val_loader, device,feature = feature_list[i])
              scale_list = np.append(scale_list,val_score)
              logging.info('Validation Dice Coeff: {}'.format(val_score))  
              
        for i in range(6):
           feature_list[i].conv.test = False
        net.up1.scale = 1
        net.up1.scale = 1
        net.up2.scale = 1
        val_list = np.array([])
        
        val_loss_list3 = np.array([])
        val_loss_list4 = np.array([])
        val_loss_list5 = np.array([])
        val_loss_list = [val_loss_list3,val_loss_list4,val_loss_list5]
        
         
        for j in range(6):
         for i in range(len(feature_list[j].conv.feature)):
          feature_list[j].conv.feature_test[i]  = []

        for j in range(len(depth_list)):
              net.depth = depth_list[j]
              for i in range(6):
               feature_list[i].conv.p = True
               feature_list[i].conv.test = True
               feature_list[i].conv.depth = net.depth

              val_score,val_loss = eval_net(net, val_loader, device)
              val_list = np.append(val_list,val_score)
              val_loss_list[j] = np.append(val_loss_list[j],val_loss)
              logging.info('Validation Dice Coeff-nonscale: {}'.format(val_score))
        net.loss = val_loss_list 
        #print(net.up6.conv.feature6)
        for i in range(6):
           feature_list[i].conv.test = False
        
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
        """
    writer.close()
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
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    print("SGD-feature-0.01")
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
    net = UNet(n_channels=3, n_classes=1, bilinear=True,depth = 5,img_h = size[1],img_w = size[2])
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')
    
    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True
    result = np.array([])
    scale_result = np.array([])
    feature = [[],[],[],[],[],[]]
    #for i in range(6):
    try:
      for i in range(6):
        depth_list = [i]
        net = UNet(n_channels=3, n_classes=1, bilinear=True,depth = 5,img_h = size[1],img_w = size[2]) 
        net.to(device=device)
        new_result =  train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
        """
        if i == 0:
           result = new_result[0][np.newaxis,:]
           scale_result = new_result[1][np.newaxis,:]
        else:
           result = np.vstack((result,new_result[0]))
           scale_result = np.vstack((scale_result,new_result[1]))
        """
        path = []
        path_val = []
        feature_list = [net.up7,net.up6,net.up1,net.up2,net.up3,net.up4]
        """      
        for feature_num in range(6):
         path = []
         path_val = []
         for j in range(len(depth_list)):
          path.append("SGD-0.01-BN-feature_up{}_depth{}.txt".format(feature_num,depth_list[j]))
          path_val.append("SGD-0.01-BN-feature_up{}_depth{}_val.txt".format(feature_num,depth_list[j]))
          if len(feature_list[feature_num].conv.feature[j])>0:
            #print(j,feature_num)
            with open(path[j], mode='w' if i == 0 else "a") as f:
              f.write('\n'.join(feature_list[feature_num].conv.feature[j].astype(str)))
            #print(feature_list[feature_num].conv.feature_test[j])
            with open(path_val[j], mode='w'if i == 0 else "a") as f:
              f.write('\n'.join(feature_list[feature_num].conv.feature_test[j].astype(str)))
        
        
         path_mean = "SGD-BN-feature_up{}_running_mean.txt".format(feature_num)
         path_var = "SGD-BN-feature_up{}_running_var.txt".format(feature_num)
     
         with open(path_mean, mode='w' if i == 0 else "a") as f:
              f.write('\n'.join(feature_list[feature_num].conv.mean.detach().numpy().astype(str)))
         with open(path_var, mode='w' if i == 0 else "a") as f:
              f.write('\n'.join(feature_list[feature_num].conv.var.detach().numpy().astype(str)))
        
        """
        feature_down = [net.down1,net.down2,net.down3,net.down4,net.down5,net.down6]
        """
        for j in range(5):
         feature1 = feature_down[j].maxpool_conv[1].feature[4]
         feature2 = feature_down[j+1].maxpool_conv[1].feature[4]
         feature1 = np.reshape(feature1,(feature1.shape[0],feature1.shape[1],feature1.shape[2]*feature1.shape[3]))
         feature2 = np.reshape(feature2,(feature2.shape[0],feature2.shape[1],feature2.shape[2]*feature2.shape[3]))
         feature1 = np.transpose(feature1,(1,0,2))
         feature2 = np.transpose(feature2,(1,0,2))
         print(feature1.shape,feature2.shape)
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
        final_layer = net.up4.conv
        incremental_cka = cka.IncrementalCKA(6,6)
        #print(final_layer.feature,i)
        feature[i] = reshape(final_layer.feature[0])
      for index_x in range(6):
                for index_y in range(6):
                    incremental_cka.increment_cka_score(index_x, index_y, feature[index_x],feature[index_y]
                    )
      save_map(incremental_cka.cka()[::-1])
      print("finished!!")

    except KeyboardInterrupt:
         torch.save(net.state_dict(), 'INTERRUPTED.pth')
         logging.info('Saved interrupt')
         try:
            sys.exit(0)
         except SystemExit:
            os._exit(0)
