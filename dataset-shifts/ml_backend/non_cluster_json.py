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

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN


import os
from tqdm import tqdm, trange

import pickle
import json
from scipy import stats

from state import state

def calc_non_cluster_info(astate,  outlier_count = 200, max_dist = 1.25, n_bins = 5, bin_type="score"):
    eval_zs = astate.get_eval_zs()
    o = astate.train_offset

    ind_top_outliers = np.argsort(astate.get_eval_scores())[:outlier_count] + o
   
    d_was_too_small = outlier_count
    while d_was_too_small > (outlier_count/2):
        d_was_too_small = 0
        max_dist+=.25

        all_used_ids = np.zeros(eval_zs.shape[0] + o, np.bool)
        all_used_ids[ind_top_outliers] = 1
        ranking_flat = []  # outlier count items
        for i in tqdm(ind_top_outliers):
            eval_i = i - o

            test_distances  = torch.norm(torch.from_numpy(eval_zs[eval_i]).cuda() 
                                 - torch.from_numpy(eval_zs).cuda(), dim=1).cpu().numpy()
            test_distances[eval_i] = np.Inf
            test_tenth_smallest = test_distances[np.argpartition(test_distances, 9)[9]]
            test_fifty_largest = test_distances[np.argpartition(test_distances, 49)[49]]
            if test_tenth_smallest > max_dist: d_was_too_small+=1

            train_distances = torch.norm(torch.from_numpy(eval_zs[eval_i]).cuda() 
                                 - torch.from_numpy(astate.get_train_zs()).cuda(), dim=1).cpu().numpy()
            train_tenth_smallest = train_distances[np.argpartition(train_distances, 9)[9]]
            train_fifty_largest = train_distances[np.argpartition(train_distances, 49)[49]]



            #close_tests  = test_distances  <= max(max_dist,test_tenth_smallest)
            #close_trains = train_distances <= max(max_dist,train_tenth_smallest)
            close_tests  = test_distances  <= min(test_fifty_largest , max(max_dist,test_tenth_smallest))
            close_trains = train_distances <= min(train_fifty_largest, max(max_dist,train_tenth_smallest))

            indices_of_interest = np.concatenate([close_trains, close_tests])
            all_used_ids = np.logical_or(all_used_ids, indices_of_interest)

            if bin_type == "distance_value":
                do_reverse=False
                min_val = train_distances[indices_of_interest].min()
                max_val = train_distances[indices_of_interest].max()
                

                bin_step = (max_val - min_val)/n_bins
                start = min_val 
                stop  = max_val
                bin_range = np.arange(start, stop, bin_step)
                bin_range_str = []
                for j in range(n_bins):
                    l = min_val + (bin_step*j)
                    r = min_val + (bin_step*(j+1))
                    val = f"{l:.3f}-{r:.3f}"
                    bin_range_str.append(val)
                
                def next_bin(val, cur_bin, bin_step):
                    return 0 if val < cur_bin + bin_step else 1
            elif bin_type == "score":
                do_reverse=True
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


            similar_test  = [[] for x in range(n_bins)]
            close_test_dicts = []
            for close_test_ind in np.where(close_tests)[0]:
                d = astate.get_simple_json(close_test_ind+ o)
                d['distance_value'] = test_distances[close_test_ind].item()
                close_test_dicts.append(d)

            close_test_dicts = sorted(close_test_dicts, key = lambda dd: dd[bin_type], reverse=do_reverse)
            bin_ind = 0
            for d in close_test_dicts:
                while next_bin(d[bin_type] , bin_range[bin_ind] , bin_step):
                    if bin_ind == n_bins-1: break
                    bin_ind += 1
                #floating point math screws up the last element to add to the last bin
                similar_test[bin_ind].append(d)


            similar_train = [[] for x in range(n_bins)]
            close_train_dicts = []
            for close_train_ind in np.where(close_trains)[0]:
                d = astate.get_simple_json(close_train_ind)
                d['distance_value'] = train_distances[close_train_ind].item()
                close_train_dicts.append(d)

            close_train_dicts = sorted(close_train_dicts, key = lambda dd: dd[bin_type], reverse=do_reverse)
            bin_ind = 0
            for j, d in enumerate(close_train_dicts):
                while next_bin(d[bin_type] , bin_range[bin_ind] , bin_step):
                    if bin_ind == n_bins-1: break
                    bin_ind += 1

                #floating point math screws up the last element to add to the last bin
                similar_train[bin_ind].append(d)

            d = astate.get_simple_json(i)

            d['bin_range'] = bin_range_str
            d["similar_test"] =  similar_test
            d["similar_train"] =  similar_train

            ranking_flat.append(d)

        print(f"{d_was_too_small} test items didn't have 10 nearest neighbros with distance of {max_dist}")
        
    ind_not_used_ids = np.ones(astate.all_scores.shape[0], np.bool)
    ind_not_used_ids[all_used_ids] = 0

    rand_train_ind = np.where(np.copy((ind_not_used_ids[:o] == True)))[0]
    rand_test_ind  = np.where(np.copy((ind_not_used_ids[o:] == True)))[0]
    np.random.shuffle(rand_train_ind)
    np.random.shuffle(rand_test_ind)

    all_used_inds = np.union1d(np.where(all_used_ids)[0], rand_train_ind[:1000])
    all_used_inds = np.union1d(all_used_inds, rand_test_ind[:1000])

    all_test = []
    all_train = []
    for ind in tqdm(all_used_inds):
        d = astate.get_simple_json(ind, get_emb=True)
        if ind < o:
            all_train.append(d)
        else:
            all_test.append(d)

    all_test = sorted(all_test, key = lambda dd: dd["score"])

    d = {
        "ranking_flat": ranking_flat,
        "all_test":     all_test,
        "all_train":    all_train
    }
    return d
    
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
    ret = calc_non_cluster_info(cur_state, outlier_count = 200)
    print("saving data to json")
    #json.encoder.FLOAT_REPR = lambda x: format(x, '.5f')
    with open("dr_nearest_eye.json",'w') as f:
        json.dump(round_floats(ret), f, indent=0)





if __name__ == "__main__":
    main()