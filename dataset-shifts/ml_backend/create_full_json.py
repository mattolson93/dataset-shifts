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

from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


import os
import argparse
from tqdm import tqdm, trange
from data import CelebA

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


model_dict = {
        "classifier": (train.classifier,ResNet18),
        "ae":         (train.ae, AutoEncoder),
        "vae":        (train.vae, VAE),
        'gan':        (train.gan, None),
        "dre":        (train.dre_scratch,None),
        "imagenet":   (train.imagenet, inception_v3)
    }

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch celeba weird Training')
    parser.add_argument('--attrs', dest='attrs', default=["Male"], help='attribute to classify')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--img_size', default=64, type=int, help='img size')
    parser.add_argument('-bs','--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--data', default='CelebA', type=str,  choices=['CelebA', 'CelebA-HQ'])
    parser.add_argument('--net', default="imagenet", type=str, choices=model_dict.keys())
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--data_path', dest='data_path', type=str, default='./data/celebA')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='./data/celebA/list_attr_celeba.txt')
    parser.add_argument('--remove_att', type=str, default="")
    parser.add_argument('--remove_att_val', type=int, default=-1)
    parser.add_argument('--trainset_total', type=int, default=50000)
    parser.add_argument('--heldout_total', type=int, default=10000)
    parser.add_argument('--outlier_total', type=int, default=1000)
    return parser.parse_args()






        

def calc_full_info(cur_state, outlier_frac = .1, n_clusters=4):
    all_paths = cur_state.all_paths

    eval_zs = cur_state.get_eval_zs()
    eval_scores = cur_state.get_eval_scores()

    print("running cluster")
    n_eval_to_select = int(eval_zs.shape[0] * outlier_frac)
    ind_top_outliers = np.argpartition(eval_scores, n_eval_to_select)[:n_eval_to_select]
    selected_zs = eval_zs[ind_top_outliers]

    ind_not_outliers = np.ones(eval_zs.shape[0], np.bool)
    ind_not_outliers[ind_top_outliers] = 0
   

    #clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(eval_zs[ind_top_outliers])
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(selected_zs)
    cluster_labels = clustering.labels_


    #seperate the eval data into the repsective clusters
    cluster_data = [[] for i in range(n_clusters+1) ]
    for ind, label in zip(ind_top_outliers, cluster_labels):
        full_ind = ind +  cur_state.train_offset #hardcoded assumption that our original data file is ordered: train, inlier, outlier
        cluster_data[label].append((full_ind,eval_scores[ind]))
      
    #sort those clusters by score (and json friendly)
    for i in range(n_clusters):
        cluster_data[i].sort(key=lambda tup: tup[1])
        new_cluster_data = []
        for j, item in enumerate(cluster_data[i]):
            jsonable_item = cur_state.get_json_friendly(item[0].item())

            jsonable_item["cluster_no"] = i
            jsonable_item["rank_no_within_cluster"] = j


            new_cluster_data.append(jsonable_item)
        cluster_data[i] = new_cluster_data

    #get the rest of the eval dataset
    for i, val in enumerate(ind_not_outliers):
        if not val: continue
        jsonable_item = cur_state.get_json_friendly(i)
        jsonable_item["cluster_no"] = -1
        jsonable_item["rank_no_within_cluster"] = -1
        cluster_data[-1].append(jsonable_item)
   
    #for every item in the selected eval data, find the closest training point
    #UNIQUENESS IS HARD
    '''used_train = [False]*cur_state.train_offset
    train_cluster_data = [[] for i in range(n_clusters + 1) ]
    for i in range(n_clusters):
        for j, item in enumerate(cluster_data[i]):
            error = True
            for closest_train_dict in cur_state.dr_topk_closest_train[item["id"]]:
                closest_train_ind = closest_train_dict['id']
                if not used_train[closest_train_ind]:
                    used_train[closest_train_ind] = True

                    jsonable_item = cur_state.get_json_friendly(closest_train_ind)

                    jsonable_item["cluster_no"] = i
                    jsonable_item["rank_no_within_cluster"] = j


                    train_cluster_data[i].append(jsonable_item)
                    error = False
                    break
            if error: 
                k = len(cur_state.dr_topk_closest_train[item["id"]])
                exit(f"cluster {i}, item {j} :only using top k={k} closest trains was not enough for uniqueness")
    
    #get the rest of the train dataset
    for i, val in enumerate(used_train):
        if val: continue
        jsonable_item = cur_state.get_json_friendly(i)
        jsonable_item["cluster_no"] = -1
        jsonable_item["rank_no_within_cluster"] = -1
        train_cluster_data[-1].append(jsonable_item)

    '''
    train_data = []
    for i in range(cur_state.train_offset):
        jsonable_item = cur_state.get_json_friendly(i)
        jsonable_item["cluster_no"] = -1
        jsonable_item["rank_no_within_cluster"] = -1
        train_data.append(jsonable_item)
   



    d = {
        "test": cluster_data,
        "train": train_data
    }
    return d
    
def round_floats(o):
    if isinstance(o, float): return round(o, 3)
    if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
    return o

def main():

    cur_state = state.load_data("state.p")


    ret = calc_full_info(cur_state, outlier_frac = .05, n_clusters=8)
    print("saving data to json")
    #json.encoder.FLOAT_REPR = lambda x: format(x, '.5f')
    with open("results_6.json",'w') as f:
        json.dump(round_floats(ret), f, indent=0)





if __name__ == "__main__":
    main()