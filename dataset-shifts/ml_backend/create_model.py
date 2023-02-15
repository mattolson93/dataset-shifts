#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import argparse

import torch


from data import celebadata
from utils import get_embeddings

from models.CustomModel import  SugiyamaNet
from models.inception import inception_v3
import train


attrs_default = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard', 
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 
    'Wearing_Necktie', 'Young'
]

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch celeba weird Training')
    parser.add_argument('--attrs'    , dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')
    parser.add_argument('--img_size' , default=299, type=int, help='img size')
    parser.add_argument('--batch_size', '-bs', default=64, type=int, help='minibatch size')
    parser.add_argument('--data'      , default='CelebA', type=str,  choices=['CelebA'])
    parser.add_argument('--data_path' , dest='data_path', type=str, default='./data/celebA')
    parser.add_argument('--attr_path' , dest='attr_path', type=str, default='./data/celebA/list_attr_celeba.txt')
    parser.add_argument('--trainset_total' , type=int, default=50000)
    parser.add_argument('--heldout_total' , type=int, default=10000)
    parser.add_argument('--outlier_total' , type=int, default=1000)
    return parser.parse_args()
    

def main():
    args=get_args()

    print(' Building imagenet model ')
    net = inception_v3('./data/imagenet.pth').to(device)
    for name, param in net.named_parameters():
        param.requires_grad = False
    net.eval()

    print("getting embeddings to train DRE model")
    datasets = celebadata.all_celeba(args)
    train_zs   = get_embeddings(net, datasets.train_dataloader)
    inlier_zs  = get_embeddings(net, datasets.inlier_dataloader)
    outlier_zs = get_embeddings(net, datasets.outlier_dataloader)

    '''if True:
        torch.save(train_zs,    "temp_trainzs.pth")
        torch.save(inlier_zs,   "temp_inlierzs.pth")
        torch.save(outlier_zs,  "temp_outlierzs.pth")
    else:
        train_zs   = torch.load("temp_trainzs.pth")
        inlier_zs  = torch.load("temp_inlierzs.pth")
        outlier_zs = torch.load("temp_outlierzs.pth")'''



    epochs = 10
    size = net.latent_size
    batch_size = args.batch_size
    #compute dr data
    dr_train_loader        = torch.utils.data.DataLoader(train_zs, batch_size=batch_size, shuffle=True,drop_last=False)
    dr_test_loader         = torch.utils.data.DataLoader(torch.cat([inlier_zs, outlier_zs]), batch_size=batch_size, shuffle=True)
    dr_inlier_loader       = torch.utils.data.DataLoader(inlier_zs, batch_size=batch_size, shuffle=False, drop_last=False)
    dr_outlier_loader      = torch.utils.data.DataLoader(outlier_zs, batch_size=batch_size, shuffle=False, drop_last=False)

    torch.cuda.empty_cache()
    dr_model = train.dre(SugiyamaNet, size, dr_train_loader, dr_test_loader, dr_inlier_loader, dr_outlier_loader, epochs, device)
    #                           model, size, train_loader   , test_loader   , inlier_data     , outlier_data     , epochs, device

    torch.save(dr_model.state_dict(), "dre.pth")



if __name__ == "__main__":
    main()