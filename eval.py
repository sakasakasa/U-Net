import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from dice_loss import dice_coeff
import dice_loss
from statistics import mean, stdev

def eval_net(net, loader, device,feature = None,gamma = 1.0,IN = True):
    """Evaluation without the densecrf with the dice coefficient"""
    
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    criterion = dice_loss.DiceLoss()
    epoch_loss = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
   
        
        diff_feature2 = 0.
        diff_std_feature2 = 0
       
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)

            true_masks = true_masks.to(device=device, dtype=mask_type)

            net.eval()
            mask_pred = net(imgs)
            
            
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            tot += dice_coeff(pred, true_masks).item()
            loss = criterion(mask_pred, true_masks)
            
            
            decoder = [net.up7,net.up6,net.up1,net.up2,net.up3,net.up4]
            depth = net.depth
            num = 5
            feature2_eval = decoder[num].conv.feature2
            net.train()
            
            feature2_train = decoder[num].conv.feature2
            
            diff_feature2 += (torch.norm(feature2_eval-feature2_train)).item()
            diff_std_feature2 += (torch.std((feature2_eval-feature2_train),axis = (1,2,3))).sum().item()
        
            net.eval()
            out = net(imgs)

                     
            epoch_loss += loss.item()
            pbar.update()
    net.train()

    """
    if not IN:
        print("bn_sigma_mean = ",torch.mean(torch.sqrt(net.up3.conv.bn1.running_var)))
        print("bn_sigma_std = ",torch.std(torch.sqrt(net.up3.conv.bn1.running_var)))
        print("bn_mu_mean = ",torch.mean((net.up3.conv.bn1.running_mean)))
        print("bn_mu_std = ",torch.std((net.up3.conv.bn1.running_mean)))
    
    print("diffK_K =",diff/len(loader)) 
    print("diffK_K_std =",diff_std/len(loader))
    """
    print("diffK_K_feature2 =",diff_feature2/len(loader))
    print("diffK_K_std_feature2 =",diff_std_feature2/len(loader))


   


    return tot / n_val,epoch_loss, diff_feature2/len(loader), diff_std_feature2/len(loader)
