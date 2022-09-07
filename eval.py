import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from dice_loss import dice_coeff
import dice_loss
from statistics import mean

def eval_net(net, loader, device,feature = None,print_slim = False, gamma = 1.0,IN = True):
    """Evaluation without the densecrf with the dice coefficient"""
    
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0
    criterion = nn.BCEWithLogitsLoss()
    criterion = dice_loss.DiceLoss()
    epoch_loss = 0
    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        lip_list = []
        term1_list = []
        term2_list = []
        term3_list = []
        sup_list = []
        sigma_list = []
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)

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
                tot += dice_coeff(pred, true_masks).item()
                loss = criterion(mask_pred, true_masks)
                lip = torch.norm(torch.autograd.grad([loss],[net.up3.conv.x],retain_graph = True)[0][0,0,:,:])
                term1 = torch.norm(torch.autograd.grad([loss],[net.up3.conv.out],retain_graph = True)[0][0,0,:,:])
                term2 = (torch.autograd.grad([loss],[net.up3.conv.out],retain_graph = True)[0][0,0,:,:]).sum()
                term3 = torch.dot(torch.flatten(torch.autograd.grad([loss],[net.up3.conv.out],retain_graph = True)[0][0,0,:,:]),torch.flatten(net.up3.conv.scaled[0,0,:,:]))
                size = net.up3.conv.size
                gamma2 = net.up3.conv.bn1.weight[0] if IN == False else gamma
                sup = ((term1**2-(term2**2)*(2/size-1/(size*size))-term3**2/size) *gamma2*gamma2/net.up3.conv.sigma/net.up3.conv.sigma).mean() if IN else ((term1**2) *gamma2*gamma2/net.up3.conv.sigma/net.up3.conv.sigma).mean()
                sigma = torch.sqrt(net.up3.conv.bn1.running_var)[0].item() if IN == False else net.up3.conv.sigma[0].item()
                #print(lip,term1,net.up4.conv.sigma.mean().item(),term2.item(),term3.item())
                #print(lip,term1,net.up4.conv.sigma.mean().item(),term2.item(),term3.item())
                lip_list.append(lip.item()**2)
                term1_list.append(term1.item()**2)
                term2_list.append(term2.item()**2)
                term3_list.append(term3.item()**2)
                sup_list.append(sup.item())
                sigma_list.append(sigma)

                epoch_loss += loss.item()
            pbar.update()
    if print_slim:
      #print(net.up2.bn1.running_var)
      print("lipschitz_eval=",mean(lip_list))
      print("term1_eval=",mean(term1_list))
      print("term2_eval=",mean(term2_list))
      print("term3_eval=",mean(term3_list))
      print("sup_eval=",mean(sup_list))
      print("sigma_eval=",mean(sigma_list))
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
