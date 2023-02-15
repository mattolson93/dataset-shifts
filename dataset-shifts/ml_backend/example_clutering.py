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

from utils import get_scores, get_embeddings, get_scores_and_embeddings, get_neighbors,test_auc,get_dataset_img_info

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

def write_data(args, data, clear=False):
    filename = f"{args.net}_{args.activation_name}_{args.lr}.csv"
    if clear: open(filename, 'w').close()
    f=open(filename, "a+"); 
    f.write(data); 
    f.close()

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
    parser.add_argument('--dre_path', dest='dre_path',  type=str, default="./models")
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--data_path', dest='data_path', type=str, default='./data/celebA')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='./data/celebA/list_attr_celeba.txt')
    parser.add_argument('--remove_att', type=str, default="")
    parser.add_argument('--remove_att_val', type=int, default=-1)
    parser.add_argument('--trainset_total', type=int, default=10) # TODO: debug ? 10 : 50000
    parser.add_argument('--heldout_total', type=int, default=1000) # TODO: debug ? 1000 : 10000
    parser.add_argument('--outlier_total', type=int, default=100) # TODO: debug ? 100 : 1000
    return parser.parse_args()



def calc_neighbors(train_zs, eval_zs, k_closest=2):
    train_zs = torch.from_numpy(train_zs)
    eval_zs = torch.from_numpy(eval_zs)

    n_train = train_zs.shape[0]

    all_data = torch.cat([train_zs,eval_zs]).cuda().squeeze()
    print("getting all distances")

    all_distances = []
    for point in tqdm(all_data):
        all_distances.append(torch.norm(point - all_data, dim=1).cpu().numpy())
    all_distances = np.array(all_distances)
    #all_distances = np.array([torch.norm(point - all_data, dim=1).cpu().numpy() for point in all_data])

    for i in range(all_data.shape[0]): all_distances[i,i] = np.Inf
    k = k_closest
    topk_closest_test = []
    topk_closest_train = []
    for i in tqdm(range(all_data.shape[0])):
        k_closests_indices = np.argpartition(all_distances[i][:n_train], k)[:k] 
        topk_closest_train.append(k_closests_indices)

        k_closests_indices = np.argpartition(all_distances[i][n_train:], k)[:k] 
        topk_closest_test.append(k_closests_indices+n_train)

    return np.array(topk_closest_train), np.array(topk_closest_test)



class state:
    def __init__(self, train_scores, train_zs, inlier_scores, inlier_zs, outlier_scores, outlier_zs, top2_closest_train, all_paths, eval_scores, emb_zs):
        self.train_scores        = train_scores
        self.train_zs            = train_zs
        self.inlier_scores       = inlier_scores
        self.inlier_zs           = inlier_zs
        self.outlier_scores      = outlier_scores
        self.outlier_zs          = outlier_zs
        self.top2_closest_train  = top2_closest_train
        self.all_paths           = all_paths
        self.eval_scores         = eval_scores
        self.emb_zs              = emb_zs

    def get_data(self):
        return self.train_scores, self.train_zs, self.inlier_scores, self.inlier_zs, self.outlier_scores, self.outlier_zs, self.top2_closest_train, self.all_paths, self.eval_scores, self.emb_zs

    def save_data(self, fname):
        pickle.dump( self, open(fname, "wb" ) )

    def load_data(fname):
        return pickle.load( open( fname, "rb" ) )
        

def process_model_data(model, datasets):
    print("getting embeddings")
    #scores, labels, ret_paths, zs
    train_scores,   _, train_paths,   train_zs   = get_dataset_img_info(model, datasets.train_dataloader)
    inlier_scores,  _, inlier_paths,  inlier_zs  = get_dataset_img_info(model, datasets.inlier_dataloader)
    outlier_scores, _, outlier_paths, outlier_zs = get_dataset_img_info(model, datasets.outlier_dataloader)

    print("determining nearest neighbors")

    all_paths = np.concatenate([train_paths,inlier_paths,outlier_paths])

    eval_scores    = np.concatenate([inlier_scores, outlier_scores])
    eval_zs        = np.concatenate([inlier_zs, outlier_zs])
    auc            = test_auc(inlier_scores, outlier_scores)

    top2_closest_train, _ = calc_neighbors(train_zs, eval_zs, k_closest=2)


    print("running UMAP")
    trans = umap.UMAP(n_neighbors=15).fit(eval_zs)
    emb_zs = trans.transform(eval_zs)
    #embs_inlier     = trans.transform(inlier_zs)
    #embs_outlier    = trans.transform(outlier_zs)
    return state(train_scores, train_zs, inlier_scores, inlier_zs, outlier_scores, outlier_zs, top2_closest_train, all_paths, eval_scores, emb_zs)
    
    
    

def calc_cluster_info(cur_state, outlier_frac = .1, n_clusters=4):
    train_scores, _, _, _, _, _, top2_closest_train, all_paths, eval_scores, emb_zs = cur_state.get_data()

    print("running cluster")
    n_eval_to_select = int(emb_zs.shape[0] * outlier_frac)
    ind_top_outliers = np.argpartition(eval_scores, n_eval_to_select)[:n_eval_to_select]
    selected_zs = emb_zs[ind_top_outliers]

    #clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(eval_zs[ind_top_outliers])
    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(selected_zs)
    cluster_labels = clustering.labels_


    #seperate the eval data into the repsective clusters
    cluster_data = [[] for i in range(n_clusters) ]
    for ind, label in zip(ind_top_outliers, cluster_labels):
        full_ind = ind +  len(train_scores) #hardcoded assumption that our original data file is ordered: train, inlier, outlier
        cluster_data[label].append((full_ind,eval_scores[ind], all_paths[full_ind] ))
      
    #sort those clusters by score (and json friendly)
    for i in range(n_clusters):
        cluster_data[i].sort(key=lambda tup: tup[1])
        new_cluster_data = []
        for item in cluster_data[i]:
            new_cluster_data.append({"index": item[0].item(), "path":  item[2]})
        cluster_data[i] = new_cluster_data



    #for every item in the selected eval data, find the closest training point
    train_cluster_data = [[] for i in range(n_clusters) ]
    for i in range(n_clusters):
        for item in cluster_data[i]:
            #TODO: add a uniqueness constraint?
            clost_train_idx = top2_closest_train[item["index"]][0]
            train_cluster_data[i].append({"index": clost_train_idx.item(), "path":  all_paths[clost_train_idx]})
                                   
    d = {
        "test": cluster_data,
        "train": train_cluster_data
    }
    return d
    


def main():

    process_model = True
    if process_model:
        args = get_args()
        datasets = all_celeba(args)

        # Model
        print('==> Building model..')

        train_func, modelfunc  = model_dict[args.net]

        model_path = args.dre_path + args.net + "_dre.pth" #TODO OS path
        model = ComboModel(modelfunc(model_path, pretrained=False).cuda(), SugiyamaNet(2048).cuda()).cuda()
        model.load_state_dict(torch.load(model_path))
        model.eval()

        cur_state = process_model_data(model, datasets)

        cur_state.save_data("state.p")
    else:
        cur_state = state.load_data("state.p")


    ret = calc_cluster_info(cur_state, outlier_frac = .1, n_clusters=4)

    with open("cluster_data_1.json",'w') as f:
        json.dump(ret, f, indent=2)



class all_celeba():
    def __init__(self, args):
        trainset_total = args.trainset_total 
        heldout_total  = args.heldout_total 
        outlier_total  = args.outlier_total 
        inlier_total   = heldout_total - outlier_total
        batch_size = args.batch_size
       
        multi_atts = ['Eyeglasses','Wearing_Hat','Wearing_Necktie']
        multi_att_vals = [-1, -1, -1]
        if args.attrs[0] == multi_atts:
            exit("cant remove att to be classified")

        kwargs = {'num_workers': 8, 'pin_memory': True} if torch.cuda.is_available() else {}

        print(args.attrs, multi_atts, multi_att_vals, "loading dataset")
        
        train_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'train', args.attrs)

        heldout_outlier_data,       train_dataset = train_dataset.split_multi_attribute(multi_atts, multi_att_vals)

        
        #outlier_train,       train_dataset = train_dataset.split_attribute(args.remove_att, args.remove_att_val)
        heldout_inlier_data, train_dataset = train_dataset.split(inlier_total)
        train_dataset, _   = train_dataset.split(trainset_total)

        test_statistics  = CelebA.get_test_statistics(args.attr_path, multi_atts, multi_att_vals)
        
        heldout_outlier_data.match_statistics(test_statistics, outlier_total)

        assert len(train_dataset)        == trainset_total
        assert len(heldout_inlier_data)  == inlier_total
        assert len(heldout_outlier_data) == outlier_total
        # Data
        self.train_dataloader = data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=False, drop_last=False, **kwargs
        )

        self.inlier_dataloader = data.DataLoader(
            heldout_inlier_data, batch_size=args.batch_size,
            shuffle=False, drop_last=False, **kwargs
        )
        self.outlier_dataloader = data.DataLoader(
            heldout_outlier_data, batch_size=args.batch_size,
            shuffle=False, drop_last=False, **kwargs
        )

        self.eval_CelebA_data = CelebA.combine(heldout_inlier_data, heldout_outlier_data)
        self.eval_dataloader = data.DataLoader(
            self.eval_CelebA_data, batch_size=args.batch_size,
            shuffle=False, drop_last=False, **kwargs
        )
       

        self.perfect_user_train_dataloader = data.DataLoader(
            CelebA.combine(train_dataset, heldout_outlier_data), batch_size=args.batch_size,
            shuffle=False, drop_last=False, **kwargs
        )





if __name__ == "__main__":
    main()