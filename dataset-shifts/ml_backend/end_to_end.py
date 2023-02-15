#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import argparse
import json

import torch


from data import celebadata
from utils import get_embeddings, get_scores, test_auc

from models.CustomModel import  SugiyamaNet, ComboModel
from models.inception import inception_v3
import train

from get_model_data import process_model_data
from non_cluster_json import calc_non_cluster_info
from cluster_json import calc_cluster_info2

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
def round_floats(o):
    if isinstance(o, float): return round(o, 3)
    if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
    return o

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch celeba weird Training')
    parser.add_argument('--attrs'    , dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')
    parser.add_argument('--img_size' , default=299, type=int, help='img size')
    parser.add_argument('--batch_size', '-bs', default=128, type=int, help='minibatch size')
    parser.add_argument('--data'      , default='CelebA', type=str,  choices=['CelebA'])
    parser.add_argument('--data_path' , dest='data_path', type=str, default='./data/celebA')
    parser.add_argument('--attr_path' , dest='attr_path', type=str, default='./data/celebA/list_attr_celeba.txt')
    parser.add_argument('--shift1', type=str, default="Eyeglasses")
    parser.add_argument('--shift2', type=str, default="Smiling")
    parser.add_argument('--shift3', type=str, default="Wearing_Necktie")
    parser.add_argument('--shiftval1', type=int, default=-1) #val in trainset
    parser.add_argument('--shiftval2', type=int, default=-1)
    parser.add_argument('--shiftval3', type=int, default=-1)
    parser.add_argument('--trainset_total' , type=int, default=5000)
    parser.add_argument('--heldout_total' , type=int, default=10000)
    parser.add_argument('--outlier_total' , type=int, default=2000)
    parser.add_argument('--seed', type=int, default=1)

    return parser.parse_args()
    

def main():
    args=get_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = False

    print(' Building imagenet model ')
    net = inception_v3('./data/imagenet.pth').to(device)
    for name, param in net.named_parameters():
        param.requires_grad = False
    net.eval()

    print("getting embeddings to train DRE model")
    datasets = celebadata.all_celeba(args)
    what_shifts = datasets.get_shift_str()

    train_zs   = get_embeddings(net, datasets.train_dataloader)
    inlier_zs  = get_embeddings(net, datasets.inlier_dataloader)
    outlier_zs = get_embeddings(net, datasets.outlier_dataloader)

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
    known_score   = get_scores(dr_model, dr_inlier_loader )
    unknown_score = get_scores(dr_model, dr_outlier_loader)
    cur_auc = int(test_auc(known_score, unknown_score) *100)


    t = args.trainset_total
    h = args.heldout_total
    o = args.outlier_total

    model = ComboModel(net, dr_model).cuda()
    model.eval()
    print("processing")
    cur_state = process_model_data(model, datasets)
    #cur_state.save_data("state.p")
    #exit()
    cur_state.set_user_study_score()
    for do_cluster in [True, False]:
        for do_og in [True, False]:
            cur_state.set_space_type(og=do_og)
            print("making json data")
            if do_cluster:
                ret = calc_cluster_info2(cur_state, outlier_count = 10)
            else:
                ret = calc_non_cluster_info(cur_state, outlier_count = 100)
            print("saving data to json")

            og_str = "og" if do_og else "dr"
            cluster_str = "cluster" if do_cluster else "nearest"
            outfile = f"{og_str}_{cluster_str}_results{what_shifts}.json"
            #outfile = f"{og_str}_{cluster_str}_eye.json"

            with open(outfile,'w') as f:
                ret = round_floats(ret)
                json.dump(ret, f, indent=0)





if __name__ == "__main__":
    main()