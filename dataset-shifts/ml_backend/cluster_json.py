'''Train CIFAR10 with PyTorch.'''
import numpy as np
import torch
import os
import pickle
import json
import random
from tqdm import tqdm, trange
from sklearn.cluster import AgglomerativeClustering

from state import state


def calc_cluster_info2(astate, outlier_count = 10, n_clusters = 100, max_dist = 2.25, n_bins = 5, bin_type="score", max_imgs_in_bin=50, k_imgs_on_frontpage = 9 ):
    do_reverse=True
    
    eval_zs = astate.get_eval_zs()
    o = astate.train_offset


    clustering = AgglomerativeClustering(n_clusters=n_clusters).fit(eval_zs)
    cluster_labels = clustering.labels_#np.argsort(label_map)

    centriods = np.zeros((n_clusters, eval_zs.shape[1]))
    cluster_avg_score = np.zeros(n_clusters)


    for i, (z, lab) in enumerate(zip(eval_zs, cluster_labels)):
        ind = i + o
        centriods[lab] += z
        cluster_avg_score[lab] += astate.get_user_study_scores(ind)

    for i in range(n_clusters):
        n_items_in_cluster = (cluster_labels == i).sum()
        centriods[i]      /= n_items_in_cluster
        cluster_avg_score[i] /= n_items_in_cluster

    most_outlier_cluster_inds = cluster_avg_score.argsort()[-outlier_count:][::-1]

    #min_display_score = np.sort(astate.all_scores[astate.get_eval_inds()])[displayed_outlier_count]

    d_was_too_small = 0
    #all_used_ids = np.zeros(eval_zs.shape[0] + o, np.bool)
    #all_used_ids[ind_top_outliers] = 1
    ranking_cluster = []  # outlier count items
    all_used_inds_list = []
    
    for cluster_no in tqdm(most_outlier_cluster_inds):
        top_images = []
        more_test  = []
        cluster_centriod = centriods[cluster_no]
        cur_cluster_score = cluster_avg_score[cluster_no]

        evals_of_interest = []
        cur_scores = []
        for selected_index, i in enumerate(astate.get_eval_inds()):
            if cluster_labels[selected_index] != cluster_no: continue

            evals_of_interest.append(i)
            cur_scores.append(astate.all_scores[i])

            more_test.append(astate.get_simple_json(i))

        capped_imgs_ind = np.random.permutation(len(more_test))[:max_imgs_in_bin]
        all_cluster_imgs  = np.array(more_test)[capped_imgs_ind]
        evals_of_interest = np.array(evals_of_interest)[capped_imgs_ind]


        train_distances = torch.norm(torch.from_numpy(cluster_centriod).cuda() 
                             - torch.from_numpy(astate.get_train_zs()).cuda(), dim=1).cpu().numpy()
        
        trains_of_interest  = np.argsort(train_distances)[:all_cluster_imgs.shape[0]]
        indices_of_interest = np.concatenate([trains_of_interest, evals_of_interest])
        all_used_inds_list.append(indices_of_interest)

        similar_train = [astate.get_simple_json(i) for i in trains_of_interest]


        min_val = astate.get_user_study_scores()[indices_of_interest].min()
        max_val = astate.get_user_study_scores()[indices_of_interest].max()
        
        bin_step = ( max_val - min_val)/n_bins
        bin_range = []
        bin_range_str = []
        for j in range(n_bins):
            l = min_val + (bin_step*j)
            r = min_val + (bin_step*(j+1))
            val = f"{l:.3f}-{r:.3f}"
            bin_range.append(l)
            bin_range_str.append(val)
        bin_range.append(r)
        bin_range = bin_range[::-1]
        bin_range_str = bin_range_str[::-1]

        def next_bin(val, cur_bin, bin_step):
            return False if val >= (cur_bin - bin_step) else True


        close_test_dicts = sorted(all_cluster_imgs.tolist(), key = lambda dd: dd[bin_type], reverse=do_reverse)
        bin_ind = 0
        binned_test   = [[] for x in range(n_bins)]
        binned_train  = [[] for x in range(n_bins)]

        for d in close_test_dicts:
            if len(top_images) < k_imgs_on_frontpage:
                top_images.append(d.copy())

            while next_bin(d[bin_type] , bin_range[bin_ind] , bin_step):
                if bin_ind == n_bins-1: break
                
                bin_ind += 1
            #floating point math screws up the last element to add to the last bin
            binned_test[bin_ind].append(d)


        close_train_dicts = sorted(similar_train, key = lambda dd: dd[bin_type], reverse=do_reverse)
        bin_ind = 0
        for j, d in enumerate(close_train_dicts):
            while next_bin(d[bin_type] , bin_range[bin_ind] , bin_step):
                if bin_ind == n_bins-1: break
                
                bin_ind += 1

            #floating point math screws up the last element to add to the last bin
            binned_train[bin_ind].append(d)



        d = {
            'bin_range'        : bin_range_str,
            'top_images'       : sorted(top_images, key = lambda dd: dd[bin_type], reverse=do_reverse),
            "more_test"        : binned_test,
            "similar_train"    : binned_train,
            "cluster_avg_score": cur_cluster_score,
            "cluster_no"       : cluster_no
        }    

        ranking_cluster.append(d)

    sorted_ranking_cluster = sorted(ranking_cluster, key = lambda dd: dd["cluster_avg_score"], reverse=True)

    cluster_labeling_map = [None] * n_clusters
    for i, sorted_dicts in enumerate(sorted_ranking_cluster):
        cluster_labeling_map[sorted_dicts["cluster_no"]] = i +1
        sorted_dicts["cluster_no"] = i +1



    #most_outlier_cluster_inds
    print(f"{d_was_too_small} test items didn't have 10 nearest neighbros with distance of {max_dist}")

    ind_not_used_ids = np.ones(astate.all_scores.shape[0], np.bool)
    ind_not_used_ids[np.unique(np.concatenate(all_used_inds_list))] = 0

    rand_train_ind = np.where(np.copy((ind_not_used_ids[:o] == True)))[0]
    rand_test_ind  = np.where(np.copy((ind_not_used_ids[o:] == True)))[0]
    np.random.shuffle(rand_train_ind)
    np.random.shuffle(rand_test_ind)

    non_clusters_inds = np.concatenate([rand_train_ind[:1000], rand_test_ind[:1000]])

    all_test = []
    all_train = []
    for og_cluster, cluster_inds in zip(most_outlier_cluster_inds, all_used_inds_list):
        cluster_id = cluster_labeling_map[og_cluster]
        for ind in cluster_inds:
            d = astate.get_simple_json(ind, get_emb=True)
            d["cluster_no"] = cluster_id
            if ind < o:
                all_train.append(d)
            else:
                all_test.append(d)

    for ind in non_clusters_inds:
        d = astate.get_simple_json(ind, get_emb=True)
        d["cluster_no"] = -1
        if ind < o:
            all_train.append(d)
        else:
            all_test.append(d)


    ret = {
        "ranking_clusters": sorted_ranking_cluster,
        "all_test":     all_test,
        "all_train":    all_train
    }
    return ret


def round_floats(o):
    if isinstance(o, float): return round(o, 3)
    if isinstance(o, dict): return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [round_floats(x) for x in o]
    return o

def main():

    print("loading state")
    cur_state = state.load_data("state.p")
    cur_state.set_user_study_score()
    cur_state.set_space_type(og=False)

    print("processing")
    #ret = calc_cluster_info(cur_state, outlier_count = 1000)
    ret = calc_cluster_info2(cur_state, outlier_count = 10)
    print("saving data to json")
    #pickle.dump( ret, open("temp_deleteme.p", "wb" ) )
    #json.encoder.FLOAT_REPR = lambda x: format(x, '.5f')
    with open("dr_cluster2_results.json",'w') as f:
        ret = round_floats(ret)
        json.dump(ret, f, indent=0)





if __name__ == "__main__":
    main()