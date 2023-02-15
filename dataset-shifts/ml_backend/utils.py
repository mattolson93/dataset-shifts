from __future__ import print_function
import argparse
import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


import matplotlib
matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt

import saliency
from importlib import reload

reload(saliency)
reload(saliency)

def get_roc(known_scores, unknown_scores, do_plot=False, **options):
    y_true = np.array([0] * len(known_scores) + [1] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    #fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = roc_auc_score(y_true, y_score)
    return auc_score

def get_score(preds, mode):
    if 'ce' in  mode:# 'confidence_threshold':
        return 1 - torch.max(torch.softmax(preds, dim=1), dim=1)[0].data.cpu().numpy()
    elif mode == 'augmented_classifier':
        return preds[:, -1].data.cpu().numpy()
        #return torch.softmax(preds, dim=1)[:, -1].data.cpu().numpy()
    elif mode == 'entropy':
        return -((preds * torch.log(preds)).sum(1)).data.cpu().numpy()
    elif mode in ['kliep', 'mse', '1vsall', 'hmax', 'hacked_mse','ulsif']:#'kliep_threshold':
        return 1-torch.max(preds, dim=1)[0].data.cpu().numpy()
    assert False

@torch.no_grad()
def test_open_set_performance(model, testing_dataset, openset_dataset, mode, get_acc=False):
    model.eval()
    device = next(model.parameters()).device
    known_scores = []
    unknown_scores = []
    total_correct = 0
    total = 0

    for images, labels in testing_dataset:
        logits = model(images.to(device))
        known_scores.extend(get_score(logits, mode))
        if get_acc:
            total_correct += torch.sum(logits.max(dim=1)[1].cpu() == labels)
            total+=logits.size(0)

    for images, labels in openset_dataset:
        logits = model(images.to(device))
        unknown_scores.extend(get_score(logits, mode))

    
    if get_acc: print('Test Accuracy: {}/{} ({:.03f})'.format(total_correct, total, float(total_correct) / total))

    auc = get_roc(known_scores, unknown_scores, mode)
    print(f'{auc:.4f} AUC SCORE.  Mode {mode}') 
    print(f'avg   known:  {np.mean(known_scores):.4f}~ {np.std(known_scores):.4f}')
    print(f'avg unknown:  {np.mean(unknown_scores):.4f}~ {np.std(unknown_scores):.4f}')
    return auc


  
@torch.no_grad()
def test_model(model, dataset):
    model.eval()
    device = next(model.parameters()).device

    total = 0
    total_correct = 0
    for images, labels in dataset:
        logits = model(images.to(device))
        correct = torch.sum(logits.max(dim=1)[1].cpu() == labels)
        total += len(images)
        total_correct += correct
    accuracy = float(total_correct) / total
    print('Test Accuracy: {}/{} ({:.03f})'.format(total_correct, total, accuracy))
    return accuracy



@torch.no_grad()
def get_scores(model, dataset):
    model.eval()
    device = next(model.parameters()).device

    cur_scores = []
    for batch in tqdm(dataset):
        inputs = batch[0] if type(batch) is list else batch

        score = model.get_score(inputs.to(device)).cpu().numpy()
        cur_scores.extend(score)

    return np.array(cur_scores).squeeze()

def test_auc(known_scores, unknown_scores):
    y_true = np.array([1] * len(known_scores) + [0] * len(unknown_scores))
    y_score = np.concatenate([known_scores, unknown_scores])
    #fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    
    print(f'{auc:.4f} AUC SCORE. ') 
    print(f'avg   known:  {np.mean(known_scores):.4f}~ {np.std(known_scores):.4f}')
    print(f'avg unknown:  {np.mean(unknown_scores):.4f}~ {np.std(unknown_scores):.4f}')
    return auc


@torch.no_grad()
def get_embeddings(model, dataset):
    model.eval()
    device = next(model.parameters()).device
    zs = torch.empty(0)

    for batch in tqdm(dataset):
        inputs = batch[0] if type(batch) is list else batch
        z = model.get_latent(inputs.to(device))
        zs = torch.cat([zs, z.cpu()], dim = 0)

    return zs

@torch.no_grad()
def get_scores_and_embeddings(model, dataset):
    model.eval()
    device = next(model.parameters()).device
    zs = torch.empty(0)

    cur_score = []
    for batch in tqdm(dataset):
        inputs = batch[0] if type(batch) is list else batch
        
        density_score = model.get_score(inputs.to(device)).cpu().numpy()
        cur_score.extend(density_score)

        z = model.z 
        zs = torch.cat([zs, z.cpu()], dim = 0)


    return np.array(cur_score).squeeze(), zs.numpy()

@torch.no_grad()
def get_scores_and_salientembeddings(model, dataset, layer=2):
    model.eval()
    device = next(model.parameters()).device
    zs = torch.empty(0)

    cur_score = []
    cam = saliency.ScoreCam(model, layer)
    for batch in tqdm(dataset):
        inputs = batch[0] if type(batch) is list else batch
        
        scores, z = cam.generate_cam(inputs.to(device), get_img=False)
        cur_score.extend(scores)

        zs = torch.cat([zs, z.cpu()], dim = 0)


    return cur_score, zs

@torch.no_grad()
def save_salient_imgs(model, dataset, outfile_base, layer=3):
    model.eval()
    device = next(model.parameters()).device

    cam = saliency.ScoreCam(model, layer)
    for batch in tqdm(dataset):
        inputs = batch[0] if type(batch) is list else batch

        cam.save_cams(inputs.to(device), batch[2], outfile_base)
        break





def get_dataset_img_info(net, img_dataset, get_salient_zs=False):
    device = next(net.parameters()).device
    
    net.eval()
    test_loss = 0
    correct = []
    total = 0
    ret_paths = []
    labels = []
    scores = []
    zs = torch.empty(0)

    with torch.no_grad():
        for batch_idx, (inputs, targets, paths) in enumerate(tqdm(img_dataset, desc="getting stats")):
            inputs, targetss = inputs.to(device), targets.to(device).type(torch.float)

            score = net.get_score(inputs)
            z = net.z


            zs = torch.cat([zs, z.cpu()], dim = 0)
            scores.extend(score.cpu().numpy())
            labels.extend(targets.cpu().numpy())
            ret_paths.extend(paths)

    scores = np.array(scores).squeeze()
    labels = np.array(labels).squeeze()
    ret_paths = np.array(ret_paths).squeeze()

    return scores, labels, ret_paths, zs.numpy()

def get_combomodel_dataset_info(net, img_dataset, get_salient_zs=False):
    device = next(net.parameters()).device
    
    net.eval()
    test_loss = 0
    correct = []
    total = 0
    ret_paths = []
    labels = []
    scores = []
    dre_zs = torch.empty(0)
    imgnet_zs = torch.empty(0)

    with torch.no_grad():
        for batch_idx, (inputs, targets, paths) in enumerate(tqdm(img_dataset, desc="getting stats")):
            inputs, targetss = inputs.to(device), targets.to(device).type(torch.float)

            score = net.get_score(inputs)

            dre_zs    = torch.cat([dre_zs, net.z.cpu()], dim = 0)
            imgnet_zs = torch.cat([imgnet_zs, net.cnn.z.cpu()], dim = 0)

            scores.extend(score.cpu().numpy())
            labels.extend(targets.cpu().numpy())
            ret_paths.extend(paths)

    scores = np.array(scores).squeeze()
    labels = np.array(labels).squeeze()
    ret_paths = np.array(ret_paths).squeeze()

    return scores, labels, ret_paths, imgnet_zs.numpy(), dre_zs.numpy()


    

def get_neighbors(net, datasets, k_closest=20,  tsub_total = 10000):
    train_zs   = get_embeddings(net, datasets.train_dataloader)
    inlier_zs  = get_embeddings(net, datasets.inlier_dataloader)
    outlier_zs = get_embeddings(net, datasets.outlier_dataloader)

    temp = np.arange(train_zs.shape[0])
    np.random.seed(0)
    np.random.shuffle(temp)
    t_ind = temp[:tsub_total] # get 10k random training points

    all_data = torch.cat([train_zs[t_ind],inlier_zs, outlier_zs]).cuda().squeeze()
    all_distances = np.array([torch.norm(point - all_data, dim=1).cpu().numpy() for point in all_data])

    for i in range(all_data.shape[0]): all_distances[i,i] = np.Inf

    k = k_closest
    topk_closest_test = []
    topk_closest_train = []
    for i in tqdm(range(all_data.shape[0])):
        k_closests_indices = np.argpartition(all_distances[i][:tsub_total], k)[:k] 
        topk_closest_train.append(k_closests_indices)

        k_closests_indices = np.argpartition(all_distances[i][tsub_total:], k)[:k] 
        topk_closest_test.append(k_closests_indices+tsub_total)


    return np.array(topk_closest_train), np.array(topk_closest_test), t_ind





def make_plot2d(x, y, title, outfile):
    sns.set(rc={'figure.figsize':(50,50)})
    palette = sns.color_palette("bright", len(set(y)))
    myplot = sns.scatterplot(x[:,0], x[:,1], hue=y, legend='full', palette=palette).set_title(title)
    fig = myplot.get_figure()
    fig.savefig(outfile)
    fig.clf()

def make_plot2d_scores(x, y, scores, title, outfile, reverse_colors=False):
    if x.shape[1] != 2: exit("bad x shape for 2d plot")
    sns.set(rc={'figure.figsize':(50.7,50.27)})
    palette = sns.cubehelix_palette(len(set(scores)))
    if reverse_colors: palette = palette[::-1]
    #palette = sns.color_palette("bright", len(set(scores)))
    myplot = sns.scatterplot(x[:,0], x[:,1], hue=scores, linewidth=0.25, alpha=.75, legend='full', palette=palette).set_title(title)
    fig = myplot.get_figure()
    fig.savefig(outfile)
    fig.clf()