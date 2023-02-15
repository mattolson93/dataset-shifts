'''Train CIFAR10 with PyTorch.'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data

from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score, pairwise_distances
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

import os
import argparse
from tqdm import tqdm, trange
from data import celebadata

from utils import get_scores, get_embeddings, get_scores_and_embeddings, get_neighbors,test_auc,get_combomodel_dataset_info

import matplotlib
matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import umap
import time
import pickle
import json
from scipy import stats

from models.CustomModel import ResNet18, AutoEncoder, SugiyamaNet, ComboModel, VAE, SugiyamaNet
from models.inception import inception_v3
import train

from state import state

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
    parser.add_argument('--shift1', type=str, default="Eyeglasses")
    parser.add_argument('--shift2', type=str, default="Wearing_Hat")
    parser.add_argument('--shift3', type=str, default="Wearing_Necktie")
    parser.add_argument('--shiftval1', type=int, default=-1) #val in trainset
    parser.add_argument('--shiftval2', type=int, default=-1)
    parser.add_argument('--shiftval3', type=int, default=-1)
    parser.add_argument('--trainset_total' , type=int, default=50000)
    parser.add_argument('--heldout_total' , type=int, default=10000)
    parser.add_argument('--outlier_total' , type=int, default=1000)
    return parser.parse_args()



def close_lists_to_dict(indices, vals):

    return [{"id": i.item(), "val":v.item()}  for i, v in zip(indices,vals)]


def process_model_data(model, datasets, k = 30):
    print("getting embeddings")
    #scores, labels, ret_paths, zs
    train_scores,   tlabels, train_paths,   og_train_zs  , dr_train_zs   = get_combomodel_dataset_info(model, datasets.train_dataloader)
    inlier_scores,  ilabels, inlier_paths,  og_inlier_zs , dr_inlier_zs  = get_combomodel_dataset_info(model, datasets.inlier_dataloader)
    outlier_scores, olabels, outlier_paths, og_outlier_zs, dr_outlier_zs = get_combomodel_dataset_info(model, datasets.outlier_dataloader)
    all_paths = np.concatenate([train_paths,inlier_paths,outlier_paths]).tolist()
    all_labels = np.concatenate([tlabels, ilabels, olabels]).tolist()
    all_shift_labels =  np.concatenate([ np.array(['[-1,-1,-1]'] * (og_train_zs.shape[0] + og_inlier_zs.shape[0])), \
                        np.array([np.array2string(np.array(shift_lab)) for shift_lab in datasets.heldout_outlier_data.shift_labels]) ]).tolist()


    print("determining nearest neighbors")
    og_eval_zs        = np.concatenate([og_inlier_zs, og_outlier_zs])
    dr_eval_zs        = np.concatenate([dr_inlier_zs, dr_outlier_zs])
    auc            = test_auc(inlier_scores, outlier_scores)
    print("validating auc is good at: ",auc)

    og_topk_closest_train, og_topk_closest_test, all_og_distances = calc_neighbors(og_train_zs, og_eval_zs, k_closest=k)
    dr_topk_closest_train, dr_topk_closest_test, all_dr_distances = calc_neighbors(dr_train_zs, dr_eval_zs, k_closest=k)

    n_train = og_train_zs.shape[0]
    print(f"calculating distance ratios {all_og_distances.shape[0] * all_dr_distances.shape[0]} divisions")
    distance_ratios = all_og_distances / (all_dr_distances + 1e-8)
    for i in range(distance_ratios.shape[0]): distance_ratios[i,i] = np.Inf
    topk_closest_train_ratio = []
    for i in tqdm(range(distance_ratios.shape[0])):
        #train
        k_closests_indices = np.argpartition(distance_ratios[i][:n_train], k)[:k] 
        #argpartition doesn't sort, it just guarantees the <k elements are smaller
        k_closests_indices = k_closests_indices[np.argsort(distance_ratios[i][:n_train][k_closests_indices])]
        k_closests_indices_vals = distance_ratios[i][k_closests_indices]
        topk_closest_train_ratio.append(close_lists_to_dict(k_closests_indices, k_closests_indices_vals))


    print("running UMAP")
    def rescale(a, max_val=10):
        return max_val*((a - np.min(a))/np.ptp(a))

    og_trans = umap.UMAP(n_neighbors=15).fit(og_eval_zs)
    dr_trans = umap.UMAP(n_neighbors=15).fit(dr_eval_zs)
    og_emb_zs = rescale(og_trans.transform(np.concatenate([og_train_zs,og_inlier_zs, og_outlier_zs])))

    dr_emb_zs = rescale(dr_trans.transform(np.concatenate([dr_train_zs,dr_inlier_zs, dr_outlier_zs])))

    #embs_inlier     = trans.transform(inlier_zs)
    #embs_outlier    = trans.transform(outlier_zs)
    #import pdb; pdb.set_trace()
    return state(train_scores,  og_train_zs  , dr_train_zs, inlier_scores, og_inlier_zs , dr_inlier_zs, outlier_scores, og_outlier_zs, dr_outlier_zs, \
                 og_topk_closest_train, og_topk_closest_test,  dr_topk_closest_train, dr_topk_closest_test, all_paths, all_labels, all_shift_labels, og_emb_zs, dr_emb_zs , topk_closest_train_ratio)
    


def calc_neighbors(train_zs, eval_zs, k_closest=30):

    n_train = train_zs.shape[0]
    all_data = np.concatenate([train_zs,eval_zs])
    print("getting all distances")
    all_distances = pairwise_distances(all_data, metric='l2', n_jobs=8)
    #all_distances = np.array([torch.norm(point - all_data, dim=1).cpu().numpy() for point in all_data])

    for i in range(all_data.shape[0]): all_distances[i,i] = np.Inf


    print(f"finding top k={k_closest} neighbors")
    k = k_closest
    topk_closest_test = []
    topk_closest_train = []
    for i in tqdm(range(all_data.shape[0])):
        #train
        k_closests_indices = np.argpartition(all_distances[i][:n_train], k)[:k] 
        #argpartition doesn't sort, it just guarantees the <k elements are smaller
        k_closests_indices = k_closests_indices[np.argsort(all_distances[i][:n_train][k_closests_indices])]
        k_closests_indices_vals = all_distances[i][k_closests_indices]
        topk_closest_train.append(close_lists_to_dict(k_closests_indices, k_closests_indices_vals))

        #test
        k_closests_indices = np.argpartition(all_distances[i][n_train:], k)[:k] 
        k_closests_indices = k_closests_indices[np.argsort(all_distances[i][n_train:][k_closests_indices])] + n_train  #need the offset
        k_closests_indices_vals = all_distances[i][k_closests_indices]
        topk_closest_test.append(close_lists_to_dict(k_closests_indices, k_closests_indices_vals))

    return topk_closest_train, topk_closest_test, all_distances



        

def main():

    args = get_args()
    datasets = celebadata.all_celeba(args)
    
    # Model
    print('==> Building model..')

    net = inception_v3('./data/imagenet.pth').cuda()
    for name, param in net.named_parameters():
        param.requires_grad = False
    dr_model = SugiyamaNet(2048).cuda()
    dr_model.load_state_dict(torch.load("dre.pth"))

    model = ComboModel(net, dr_model).cuda()
    model.eval()
    cur_state = process_model_data(model, datasets)
    print("saving state")
    cur_state.save_data("state.p")
    


if __name__ == "__main__":
    main()