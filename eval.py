import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from dice_loss import dice_coeff
import dice_loss
from statistics import mean

def eval_net(net, loader, device,feature = None):
    """Evaluation without the densecrf with the dice coefficient"""
    
    #net.eval()
    net.train()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    tot2 = 0
    criterion = nn.BCEWithLogitsLoss()
    criterion = dice_loss.DiceLoss()
    epoch_loss = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        lip_list = []
        term1_list = []
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            #print(imgs.requires_grad)
            #imgs.requires_grad = True
            #print(net.up3.conv.conv1[0].weight.requires_grad)

            true_masks = true_masks.to(device=device, dtype=mask_type)

            #with torch.no_grad():
            mask_pred = net(imgs)
                #lip = torch.norm(torch.autograd.grad([loss],[net.up3.conv.x],retain_graph = True)[0][0,0,:,:])
                #term1 = torch.norm(torch.autograd.grad([loss],[net.up3.conv.out],retain_graph = True)[0][0,0,:,:])
                #lip_list.append(lip.item())
                #term1_list.append(term1.item())

 
            
            if feature is not None:
                feature.conv.p = False
            
            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                #x_dummy = np.zeros(pred.shape)#((16,1024,9,12))
                #x_dummy = x_dummy.astype(np.float32)
                #x_tensor = torch.from_numpy(x_dummy).clone().cuda()
                tot += dice_coeff(pred, true_masks).item()
                loss = criterion(mask_pred, true_masks)
                #print(imgs.requires_grad)
                #print(net.up3.conv.conv1[0].weight.requires_grad)
                #print(loss.requires_grad)
                #loss.backward()
                #print(net.up3.conv.x.requires_grad)
                lip = torch.norm(torch.autograd.grad([loss],[net.up3.conv.x],retain_graph = True)[0][0,0,:,:])
                term1 = torch.norm(torch.autograd.grad([loss],[net.up3.conv.out],retain_graph = True)[0][0,0,:,:])
                lip_list.append(lip.item())
                term1_list.append(term1.item())
                epoch_loss += loss.item()
            pbar.update()
            print("lipschitz_eval=",mean(lip_list))
            print("term1_eval=",mean(term1_list))
    net.train()
    """
    #print("train") 
    n_val = len(loader)
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    tot2 = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            with torch.no_grad():
                #print("train")
                mask_pred = net(imgs)
                #print("train!")
                #print(mask_pred.sum())
                feature2 = net.up4.conv.feature
            if False:#net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                #print("mask",mask_pred.sum())
                #print(mask_pred)
                pred2 = torch.sigmoid(mask_pred)
                #print(pred.sum())
                pred2 = (pred2 > 0.5).float()
                #print(pred.sum())
                tot2 += dice_coeff(pred2, true_masks).item()
    print(tot2/n_val)
    #print("difference",(feature1-feature2).abs().sum().cpu().numpy().copy())
    """
    return tot / n_val,epoch_loss
