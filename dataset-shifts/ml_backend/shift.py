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

import numba

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
import umap
import time
from scipy import stats

from models.CustomModel import ResNet18, vgg11, AutoEncoder, SugiyamaNet, ComboModel, VAE, SugiyamaNet
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
        "dre":        (train.dre_scratch,vgg11),
        "imagenet":   (train.imagenet, inception_v3)
    }

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_args():
    parser = argparse.ArgumentParser(description='PyTorch celeba weird Training')
    parser.add_argument('--attrs', dest='attrs', default=attrs_default, nargs='+', help='attributes to learn')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--img_size', default=64, type=int, help='img size')
    parser.add_argument('--epochs', default=20, type=int, help=' ')
    parser.add_argument('--batch_size', default=128, type=int, help='minibatch size')
    parser.add_argument('--data', default='CelebA', type=str,  choices=['CelebA', 'CelebA-HQ'])
    parser.add_argument('--net', default="classifier", type=str, choices=model_dict.keys())
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--data_path', dest='data_path', type=str, default='./data/celebA')
    parser.add_argument('--attr_path', dest='attr_path', type=str, default='./data/celebA/list_attr_celeba.txt')
    parser.add_argument('--remove_att', type=str, default="")
    parser.add_argument('--remove_att_val', type=int, default=-1)
    parser.add_argument('--trainset_total', type=int, default=50000)
    parser.add_argument('--heldout_total', type=int, default=10000)
    parser.add_argument('--outlier_total', type=int, default=1000)
    parser.add_argument("--save_dir", type=str,  default="pics")
    parser.add_argument("--do_train", action='store_true', help="  ")
    parser.add_argument("--do_dre", action='store_true', help="")
    parser.add_argument("--cluster_eval_type", type=str, default="", help="??")
    return parser.parse_args()



@torch.no_grad()
def calc_inception_distr(model, dataset):
    model.eval()
    device = next(model.parameters()).device
    all_logits = torch.empty(0).to(device)

    print("pre-parsing the training data for average distribution")
    for images, labels, _ in tqdm(dataset):
        all_logits = torch.cat([all_logits, model(images.to(device))], dim = 0)

    model.average_distr = torch.log_softmax(all_logits, dim=1).mean(0)


def main():
    args = get_args()
    datasets = all_celeba(args)
    trainset_total = args.trainset_total 
    heldout_total  = args.heldout_total 
    outlier_total  = args.outlier_total 
    inlier_total   = heldout_total - outlier_total

    # Model
    print('==> Building model..')

    train_func, modelfunc  = model_dict[args.net]

    model_path = args.net + ".pth"
    
    if args.net == 'imagenet':
        model = modelfunc(model_path).to(device)
        for name, param in model.named_parameters():
            param.requires_grad = False

        calc_inception_distr(model,  datasets.train_dataloader)
        bs = args.batch_size
    else:
        model = modelfunc().to(device)
        bs = 2048

        if args.do_train:
            optimizer  = optim.Adam(model.parameters(), lr=args.lr) if args.net !='dre' else optim.SGD(model.parameters(), lr=.01)
            for e in range(args.epochs):
                train_func(model, optimizer, datasets.train_dataloader, datasets.eval_dataloader)
            torch.save(model.state_dict(), model_path)
        else:
            model.load_state_dict(torch.load(model_path))
    model.eval()

    original_train_zs   = get_embeddings(model, datasets.train_dataloader)
    original_inlier_zs  = get_embeddings(model, datasets.inlier_dataloader)
    original_outlier_zs = get_embeddings(model, datasets.outlier_dataloader)

    if args.do_dre:
        model = get_2phase_dre(model,datasets, bs)
        model.eval()


    print("getting embeddings")
   
    train_scores,   train_zs   = get_scores_and_embeddings(model, datasets.train_dataloader)
    inlier_scores,  inlier_zs  = get_scores_and_embeddings(model, datasets.inlier_dataloader)
    outlier_scores, outlier_zs = get_scores_and_embeddings(model, datasets.outlier_dataloader)
    
    eval_scores    = np.concatenate([inlier_scores, outlier_scores])
    eval_zs        = np.concatenate([inlier_zs, outlier_zs])
    auc            = test_auc(inlier_scores, outlier_scores)


    top1k_scores = np.argpartition(eval_scores, outlier_total)[:outlier_total]
    inlier_total   =  heldout_total  - outlier_total
    outlier_selection_percent = (top1k_scores > inlier_total).sum() / outlier_total
    print(f"percentage of outlier selected: {outlier_selection_percent*100}")


    train_neighbors, eval_neighbors, train_subset = get_neighbors(model, datasets)


    if args.cluster_eval_type != "":
        print("kmeans or inception score or something")


   
    
    trans = umap.UMAP(n_neighbors=15,verbose=True).fit(eval_zs)
    embs_train      = trans.transform(train_zs[train_subset])
    embs_inlier     = trans.transform(inlier_zs)
    embs_outlier    = trans.transform(outlier_zs)

    total_points = embs_train.shape[0] + embs_inlier.shape[0] + embs_outlier.shape[0] 

    train_scores,   tlabels, t_paths, _ = get_dataset_img_info(model, datasets.train_dataloader)  #this is probably what I want
    inlier_scores,  ilabels, i_paths, _ = get_dataset_img_info(model, datasets.inlier_dataloader)  #this is probably what I want
    outlier_scores, olabels, o_paths, _ = get_dataset_img_info(model, datasets.outlier_dataloader)  #this is probably what I want

    import pdb; pdb.set_trace()


    indices     = np.array(["indices"]+list(range(total_points)))
    data_type   = np.array(["datatype"]+["train"] * len(train_subset) + ["inlier"]*embs_inlier.shape[0] + ["outlier"]*embs_outlier.shape[0])
    labels      = np.concatenate([["data_label"], tlabels[train_subset], ilabels, olabels])
    paths       = np.concatenate([["data_path"],  t_paths[train_subset], i_paths, o_paths])
    scores      = np.concatenate([["shift_score"],  train_scores[train_subset], inlier_scores, outlier_scores])
    og_embs     = np.concatenate([[["emb_x","emb_y"]], embs_train, embs_inlier, embs_outlier])
    close_train = np.concatenate([[[str(i)+"_closest_train" for i in range(1, 21)]], train_neighbors])
    close_eval  = np.concatenate([[[str(i)+"_closest_eval"  for i in range(1, 21)]], eval_neighbors])


    savedata =  np.vstack([indices,data_type,labels,paths,scores,og_embs.T,close_train.T,close_eval.T]).T
    filename = f'dre_{args.net}.csv' if args.do_dre else f'{args.net}.csv'
    np.savetxt(filename,savedata, delimiter=',', fmt='%s')


    print("here")

    exit()
    do_plots(15, .1)




    '''for i, top10 in enumerate(np.argpartition(eval_scores, 10)[:10]):
        prep_img, _, path = eval_CelebA_data.__getitem__(top10)
        get_scorecam_example(net, prep_img.to(device), path, os.path.join("imgs",f"top{i}_outlier"), dr_model=dr_model, target_layer=2, invert_sigmoid=True)

    for i, top10 in enumerate(np.argpartition(eval_scores, -10)[-10:]):
        prep_img, _, path = eval_CelebA_data.__getitem__(top10)
        get_scorecam_example(net, prep_img.to(device), path, os.path.join("imgs",f"top{i}_inlier"), dr_model=dr_model, target_layer=2, invert_sigmoid=False)
    '''

    
def do_plots(neighbors,dist):
    #x_emb_dtest = TSNE(n_components=2, perplexity=neighbors, verbose=True).fit_transform(dr_eval_zs.cpu().numpy())
    x_emb_dtest = umap.UMAP(n_neighbors=neighbors,verbose=True).fit_transform(dr_eval_zs.cpu().numpy())
    #x_emb_dtest_small = x_emb_dtest[(x_emb_dtest[:,0] >= 12) & (x_emb_dtest[:,1] <= 0)]
    #make_plot2d(x_emb_dtest_small, ["test"] * x_emb_dtest_small.shape[0],        "true labels",                             "test.png")
    
    #x_emb_test = TSNE(n_components=2, perplexity=neighbors, verbose=True).fit_transform(torch.cat([inlier_zs, outlier_zs]).cpu().numpy())
    x_emb_test = umap.UMAP(n_neighbors=neighbors, verbose=True).fit_transform(torch.cat([inlier_zs, outlier_zs]).cpu().numpy())


    #do pretty graphs

    bins = 8
    graph_scores = np.array((stats.mstats.rankdata(eval_scores) / len(eval_scores)) * bins).astype(int)
    labels_true = ['inlier'] * len(inlier_zs) + ['outlier'] * len(outlier_zs)
    labels_selected = ["not selected"] * len(eval_scores)
    labled_intersected = ['excluded inlier'] * len(inlier_zs) + ['excluded outlier'] * len(outlier_zs) 
    for s in top1k_scores:
        labels_selected[s] = "selected"
        if s > len(inlier_zs):
            labled_intersected[s] = 'intersection'

    make_plot2d(x_emb_dtest, labels_true,        "true labels",                             "true_labels_density.png")
    make_plot2d(x_emb_dtest[::-1], labels_true[::-1],        "true labels",                             "true_labels_density_flipped.png")
    make_plot2d(x_emb_dtest, labels_selected,    "labels selected by density_score",        "labled_selected_density.png")
    make_plot2d(x_emb_dtest, labled_intersected, "compare true labels to density selected", "labels_intersected_density.png")
    make_plot2d_scores(x_emb_dtest, labels_true, graph_scores, "density scores plot",       "score_density.png")
    
    
    make_plot2d(x_emb_test, labels_true,        "true labels",                             "true_labels_original.png")
    make_plot2d(x_emb_test[::-1], labels_true[::-1],        "true labels",                             "true_labels_original_flipped.png")
    make_plot2d(x_emb_test, labels_selected,    "labels selected by density_score",        "labled_selected_original.png")
    make_plot2d(x_emb_test, labled_intersected, "compare true labels to density selected", "labels_intersected_original.png")
    make_plot2d_scores(x_emb_test, labels_true, graph_scores, "density scores plot",       "score_plot_original.png")
    
 

def get_2phase_dre(net,datasets, batch_size = 2048):
    train_zs   = get_embeddings(net, datasets.train_dataloader)
    inlier_zs  = get_embeddings(net, datasets.inlier_dataloader)
    outlier_zs = get_embeddings(net, datasets.outlier_dataloader)
   
    epochs = 100
    size = net.get_latent_size()
    #compute dr data
    dr_train_loader        = torch.utils.data.DataLoader(train_zs, batch_size=batch_size, shuffle=True,drop_last=False)
    dr_test_loader         = torch.utils.data.DataLoader(torch.cat([inlier_zs, outlier_zs]), batch_size=batch_size, shuffle=True)
    dr_inlier_loader       = torch.utils.data.DataLoader(inlier_zs, batch_size=batch_size, shuffle=False, drop_last=False)
    dr_outlier_loader      = torch.utils.data.DataLoader(outlier_zs, batch_size=batch_size, shuffle=False, drop_last=False)

    torch.cuda.empty_cache()
    dr_model = train.dre(SugiyamaNet, size, dr_train_loader, dr_test_loader, dr_inlier_loader, dr_outlier_loader, epochs, device)

    return ComboModel(net, dr_model)





class all_celeba():
    def __init__(self, args):
        trainset_total = args.trainset_total 
        heldout_total  = args.heldout_total 
        outlier_total  = args.outlier_total 
        inlier_total   = heldout_total - outlier_total
        batch_size = args.batch_size
       
        
        if args.attrs[0] == args.remove_att:
            exit("cant remove att to be classified")

        kwargs = {'num_workers': 6, 'pin_memory': True} if torch.cuda.is_available() else {}

        print(args.attrs, args.remove_att, args.remove_att_val, "loading dataset")
        
        train_dataset = CelebA(args.data_path, args.attr_path, args.img_size, 'train', args.attrs)
        
        outlier_train,       train_dataset = train_dataset.split_attribute(args.remove_att, args.remove_att_val)
        heldout_inlier_data, train_dataset = train_dataset.split(inlier_total)
        train_dataset, _ = train_dataset.split(trainset_total)

        heldout_outlier_data, _            = outlier_train.split(outlier_total)

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
            shuffle=True, drop_last=False, **kwargs
        )

        self.perfect_user_train_dataloader = data.DataLoader(
            CelebA.combine(train_dataset, heldout_outlier_data), batch_size=args.batch_size,
            shuffle=False, drop_last=False, **kwargs
        )





if __name__ == "__main__":
    main()