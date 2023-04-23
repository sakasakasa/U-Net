import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import numpy as np
from dice_loss import dice_coeff
import dice_loss
from statistics import mean, stdev

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
        lip2_list = []
        lip_loss_list = []
        term1_list = []
        term2_list = []
        term3_list = []
        sup_list = []
        sigma_list = []
        mu_list = []
        sigma7_list = []
        mu7_list = []
        
        diff = 0.
        diff_std = 0
        diff7_K = 0
        diff7_K_std = 0
        diff7_K_eval = 0
        diff7_K_eval_std = 0
        diff_feature2 = 0.
        diff_std_feature2 = 0
        diff7_K_feature2 = 0
        diff7_K_std_feature2 = 0
        diff7_K_eval_feature2 = 0
        diff7_K_eval_std_feature2 = 0
        up3_mean = 0
        up3_std = 0

        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)

            true_masks = true_masks.to(device=device, dtype=mask_type)

            #with torch.no_grad():
            net.eval()
            mask_pred = net(imgs)
            
            if feature is not None:
                feature.conv.p = False
            
            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, true_masks).item()
                loss = criterion(mask_pred, true_masks)
                half_channel = net.up3.conv.x.shape[1]//2
                lip_loss = torch.norm(torch.autograd.grad([loss],[net.up3.conv.x],retain_graph = True)[0][0,0,:,:])
                lip_loss2 = torch.norm(torch.autograd.grad([loss],[net.up3.conv.x2],retain_graph = True)[0][0,half_channel,:,:])
                #lip2 = torch.norm(torch.autograd.grad([torch.mean(net.up3.conv.out)],[net.up3.conv.x],retain_graph = True)[0][0,0,:,:])
                lip = torch.norm(torch.autograd.grad([torch.std(torch.sigmoid(mask_pred))],[net.up3.conv.x],retain_graph = True)[0][0,0,:,:])#torch.norm(torch.autograd.grad([loss],[net.up3.conv.x],retain_graph = True)[0][0,0,:,:])
                term1 = torch.norm(torch.autograd.grad([loss],[net.up3.conv.out],retain_graph = True)[0][0,0,:,:])
                term2 = (torch.autograd.grad([loss],[net.up3.conv.out],retain_graph = True)[0][0,0,:,:]).sum()
                term3 = torch.dot(torch.flatten(torch.autograd.grad([loss],[net.up3.conv.out],retain_graph = True)[0][0,0,:,:]),torch.flatten(net.up3.conv.scaled[0,0,:,:]))
                size = net.up3.conv.size
                gamma2 = gamma#net.up3.conv.bn1.weight[0] if IN == False else gamma
                sup = ((term1**2-(term2**2)*(2/size-1/(size*size))-term3**2/size) *gamma2*gamma2/net.up3.conv.sigma/net.up3.conv.sigma).mean() if IN else ((term1**2) *gamma2*gamma2/net.up3.conv.sigma/net.up3.conv.sigma).mean()
                sigma = net.up3.conv.sigma.tolist()#torch.sqrt(net.up3.conv.bn1.running_var)[0].item() if IN == False else net.up3.conv.sigma[0].item()
                lip_list.append(lip.item()**2)
                lip2_list.append(lip_loss2.item()**2)
                lip_loss_list.append(lip_loss.item()**2)
                term1_list.append(term1.item()**2)
                term2_list.append(term2.item()**2)
                term3_list.append(term3.item()**2)
                sup_list.append(sup.item())
                #if net.depth == 2:
                sigma_list.extend(sigma)
                mu_list.extend(net.up3.conv.mean.tolist())
                decoder = [net.up7,net.up6,net.up1,net.up2,net.up3,net.up4]
                depth = net.depth
                num = 5#num = 6-depth if depth != 7 else 0
                feature1_eval = decoder[num].conv.feature1
                feature2_eval = decoder[num].conv.feature2
                #up3_mean += feature1_eval.mean(axis = (0,2,3)).sum()
                #up3_std += feature1_eval.std(axis = (0,2,3)).sum()
                net.train()
                pred2 = net(imgs)
                feature1_train = decoder[num].conv.feature1
                feature2_train = decoder[num].conv.feature2
                diff += (torch.norm(feature1_eval-feature1_train)).item()
                diff_std += (torch.std((feature1_eval-feature1_train),axis = (1,2,3))).sum().item()
                diff_feature2 += (torch.norm(feature2_eval-feature2_train)).item()
                diff_std_feature2 += (torch.std((feature2_eval-feature2_train),axis = (1,2,3))).sum().item()
                
                if True:#net.depth == 2:
                    #decoder = [net.up7,net.up6,net.up1,net.up2,net.up3,net.up4]
                    depth = net.depth
                    net.depth = 7
                    pred2 = net(imgs)
                    #num = 4
                    feature1_train_7 = decoder[num].conv.feature1
                    feature2_train_7 = decoder[num].conv.feature2
                    diff7_K += (torch.norm(feature1_train_7-feature1_train)).item()
                    diff7_K_std += (torch.std((feature1_train_7-feature1_train),axis = (1,2,3))).sum().item()
                    diff7_K_feature2 += (torch.norm(feature2_train_7-feature2_train)).item()
                    diff7_K_std_feature2 += (torch.std((feature2_train_7-feature2_train),axis = (1,2,3))).sum().item()

                    net.eval()
                    pred2 = net(imgs)
                    sigma7_list.extend(net.up3.conv.sigma.tolist())
                    mu7_list.extend(net.up3.conv.mean.tolist())

                    feature1_eval_7 = decoder[num].conv.feature1
                    feature2_eval_7 = decoder[num].conv.feature2
                    #sigma_sum += net.up3.conv.sigma
                    #mean_sum = net.up3.conv.mean
                    diff7_K_eval += (torch.norm(feature1_eval_7-feature1_eval)).item()
                    diff7_K_eval_std += (torch.std((feature1_eval_7-feature1_eval),axis = (1,2,3))).sum().item()
                    diff7_K_eval_feature2 += (torch.norm(feature2_eval_7-feature2_eval)).item()
                    diff7_K_eval_std_feature2 += (torch.std((feature2_eval_7-feature2_eval),axis = (1,2,3))).sum().item()

                    net.depth = depth
                net.eval()
                out = net(imgs)

                     
                epoch_loss += loss.item()
            pbar.update()
    if print_slim:
      #print(torch.mean(mask_pred))
      #print("lipschitz_eval=",mean(lip_list))
      #print("lipschitz_half_eval=",mean(lip2_list))
      #print("lipschitz_loss_eval=",mean(lip_loss_list))
      #print("term1_eval=",mean(term1_list))
      #print("term2_eval=",mean(term2_list))
      #print("term3_eval=",mean(term3_list))
      #print("sup_eval=",mean(sup_list))
      #print("sigma_eval=",mean(sigma_list))
      #print("weight=",(torch.norm(net.state_dict()['up3.conv.conv2.0.weight'])))
      #print("bias=",torch.norm(net.state_dict()['up3.conv.conv2.0.bias']))
      pass
    net.train()
    if True:#net.depth == 2:
     print("sigma_mean =",mean(sigma_list))
     print("sigma_std =",stdev(sigma_list))
     print("mu_mean =",mean(mu_list))
     print("mu_std =",stdev(mu_list))
     print("sigma7_mean =",mean(sigma7_list))
     print("sigma7_std =",stdev(sigma7_list))
     print("mu7_mean =",mean(mu7_list))
     print("mu7_std =",stdev(mu7_list))
     #print("feature_mean =",up3_mean/len(loader))
     #print("feature_std =",up3_std/len(loader))

     if not IN:
        print("bn_sigma_mean = ",torch.mean(torch.sqrt(net.up3.conv.bn1.running_var)))
        print("bn_sigma_std = ",torch.std(torch.sqrt(net.up3.conv.bn1.running_var)))
        print("bn_mu_mean = ",torch.mean((net.up3.conv.bn1.running_mean)))
        print("bn_mu_std = ",torch.std((net.up3.conv.bn1.running_mean)))

     #print("mean =",mean_sum/len(loader))

     print("diffK_K =",diff/len(loader)) 
     print("diffK_K_std =",diff_std/len(loader))
     print("diffK_K_feature2 =",diff_feature2/len(loader))
     print("diffK_K_std_feature2 =",diff_std_feature2/len(loader))


     print("diff7_K_train=",diff7_K/len(loader))
     print("diff7_K_std =",diff7_K_std/len(loader))
     print("diff7_K_train_feature2=",diff7_K_feature2/len(loader))
     print("diff7_K_std_feature2 =",diff7_K_std_feature2/len(loader))


     print("diff7_K_eval=",diff7_K_eval/len(loader))
     print("diff7_K_eval_std =",diff7_K_eval_std/len(loader))
     print("diff7_K_eval_feature2=",diff7_K_eval_feature2/len(loader))
     print("diff7_K_eval_std_feature2 =",diff7_K_eval_std_feature2/len(loader))



    return tot / n_val,epoch_loss
