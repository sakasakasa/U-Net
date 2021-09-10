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

#dir_img = 'data/imgs/Synapse/train_npz/'
#dir_mask = 'data/imgs/Synapse/train_npz/'
dir_checkpoint = 'checkpoints/'
dir_img = 'data/imgs/CVC/'
dir_mask = 'data/masks/CVC/'
depth_list = [5]
 
def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.2,
              save_cp=True,
              img_scale=0.5):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    #print("shape",dataset.shape)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    #print(dataset.shape)
    train = Subset(dataset,list(range(0,503)))
    val = Subset(dataset,list(range(503,len(dataset))))
    #train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

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

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
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
    val_loss_list = [val_loss_list3,val_loss_list4,val_loss_list5]

    for epoch in range(epochs):
        net.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            net.up2.conv.p = True
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
                #distillation
                """
                net.distillation = True
                teacher = net(imgs)
                net.distillation = False  
                """
                for i in range(len(depth_list)):
                  
                  net.depth = depth_list[i]#random.choice(depth_list)
                  net.up2.conv.depth = net.depth

                  masks_pred = net(imgs)
                  """
                  net.distillation = True
                  student = net(imgs)
                  net.distillation = False
                  """
                  loss = criterion(masks_pred, true_masks)#+(1E-8)*criterion2(teacher,student)
                  epoch_loss += loss.item()
                  writer.add_scalar('Loss/train', loss.item(), global_step)

                  pbar.set_postfix(**{'loss (batch)': loss.item()})
                #optimizer.zero_grad()
                  loss.backward()
                #nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                net.up2.conv.p = False
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
        for j in range(len(depth_list)):
              net.up2.conv.p = True
              net.up2.conv.test = True
              net.depth = depth_list[j]
              net.up2.conv.depth = net.depth
              val_score = eval_net(net, val_loader, device)
              
              scale_list = np.append(scale_list,val_score)
              logging.info('Validation Dice Coeff: {}'.format(val_score))  
        net.up2.conv.test = False
        
        val_list = np.array([])
        """
        val_loss_list3 = np.array([])
        val_loss_list4 = np.array([])
        val_loss_list5 = np.array([])
        val_loss_list = [val_loss_list3,val_loss_list4,val_loss_list5]
        """
        for j in range(len(depth_list)):
              net.depth = depth_list[j]
              val_score,val_loss = eval_net(net, val_loader, device)
              val_list = np.append(val_list,val_score)
              val_loss_list[j] = np.append(val_loss_list[j],val_loss)
              logging.info('Validation Dice Coeff-nonscale: {}'.format(val_score))
        net.loss = val_loss_list 
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()
    return [val_list,scale_list]


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
    #print("depth=5-batchnorm-CVC")
    print("skip")
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
       #print("shape",batch['image'].shape)
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
    for i in range(1):
     try:
        net = UNet(n_channels=3, n_classes=1, bilinear=True,depth = 5,img_h = size[1],img_w = size[2]) 
        net.to(device=device)
        new_result =  train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
        #print(new_result,result)
        if i == 0:
           result = new_result[0][np.newaxis,:]
           scale_result = new_result[1][np.newaxis,:]
        else:
           result = np.vstack((result,new_result[0]))
           scale_result = np.vstack((scale_result,new_result[1]))
        path = "feature_up2_depth5-individual.txt"
        path2 = "feature_up2_depth5-individual.txt"
        path3 = "feature_up2_depth3_test-individual.txt"
        path4 = "feature_up2_depth5_test-individual.txt"
        path5 = "feature_up2_running_mean.txt"
        path6 = "feature_up2_running_var-IN.txt" 
        path_loss = ["val_loss3-IN.txt","val_loss4-IN.txt","val_loss5-IN.txt"]

        """
        with open(path, mode='w') as f:
              f.write('\n'.join(net.up2.conv.feature5.astype(str)))
        with open(path2, mode='w') as f:
              f.write('\n'.join(net.up2.conv.feature5.astype(str)))
        with open(path3, mode='w') as f:
              f.write('\n'.join(net.up2.conv.feature3_test.astype(str)))
        with open(path4, mode='w') as f:
              f.write('\n'.join(net.up2.conv.feature5_test.astype(str)))
        
        with open(path5, mode='w') as f:
              f.write('\n'.join(net.up2.conv.mean.detach().numpy().astype(str)))
        with open(path6, mode='w') as f:
              f.write('\n'.join(net.up2.conv.var.detach().numpy().astype(str)))
        """
        """
         
        for i in range(len(depth_list)):
          with open(path_loss[i], mode='w') as f:
              f.write('\n'.join(net.loss[i].astype(str)))
        """

     except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    #print(result)
    print("average",np.mean(result,axis = 0))
    print("std",np.std(result,axis = 0))
    print("scale_average",np.mean(scale_result,axis = 0))
    print("scale_std",np.std(scale_result,axis = 0))
