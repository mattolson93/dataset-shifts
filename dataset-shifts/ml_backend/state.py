'''Train CIFAR10 with PyTorch.'''
import numpy as np
import pickle
from pathlib import Path  

class state:
    def __init__(self, train_scores,  og_train_zs  , dr_train_zs, inlier_scores, og_inlier_zs , dr_inlier_zs, outlier_scores, og_outlier_zs, dr_outlier_zs, \
                 og_topk_closest_train, og_topk_closest_test,  dr_topk_closest_train, dr_topk_closest_test, all_paths, all_labels, all_shift_labels, og_emb_zs, dr_emb_zs , topk_closest_train_ratio):
        
        self.train_scores        = train_scores
        self.og_train_zs         = og_train_zs
        self.dr_train_zs         = dr_train_zs
        self.inlier_scores       = inlier_scores
        self.og_inlier_zs        = og_inlier_zs
        self.dr_inlier_zs        = dr_inlier_zs
        self.outlier_scores      = outlier_scores
        self.og_outlier_zs       = og_outlier_zs
        self.dr_outlier_zs       = dr_outlier_zs

        self.train_offset        = og_train_zs.shape[0]

        self.og_topk_closest_train  = og_topk_closest_train
        self.dr_topk_closest_train  = dr_topk_closest_train

        self.og_topk_closest_test   = og_topk_closest_test
        self.dr_topk_closest_test   = dr_topk_closest_test

        self.topk_closest_train_ratio = topk_closest_train_ratio

        self.all_og_zs     = np.concatenate([og_train_zs,og_inlier_zs,og_outlier_zs])
        self.all_dr_zs     = np.concatenate([dr_train_zs,dr_inlier_zs,dr_outlier_zs])
        self.all_scores    = np.concatenate([train_scores,inlier_scores,outlier_scores])
        self.all_rankings  = np.argsort(self.all_scores)
        
        self.all_paths           = all_paths
        self.all_labels          = all_labels
        self.all_shift_labels    = all_shift_labels
        self.og_emb_zs           = og_emb_zs
        self.dr_emb_zs           = dr_emb_zs

        self.set_user_study_score()
        self.set_space_type(og=False)

    def set_user_study_score(self):
        a = 1 - self.all_scores
        self.user_study_scores = (a - np.min(a))/np.ptp(a)


    def set_space_type(self, og=False):
        self.space_type_original = og
        if og:
            self.all_zs = np.vstack([self.og_train_zs, self.og_inlier_zs, self.og_outlier_zs])
        else:
            self.all_zs = np.vstack([self.dr_train_zs, self.dr_inlier_zs, self.dr_outlier_zs])

    def get_id(self, index):
        return int(self.all_paths[index][14:-4])

    def get_filename(self, index):
        return Path(self.all_paths[index]).name

    def get_user_study_scores(self, i=None):
        if i is None:
            return self.user_study_scores
        else:
            return self.user_study_scores[i]


    def get_train_zs(self): return self.all_zs[:self.train_offset]
    def get_eval_zs(self):  return self.all_zs[self.train_offset:]
    def get_eval_inds(self):  return np.arange(self.dr_inlier_zs.shape[0] + self.dr_outlier_zs.shape[0]) + self.train_offset


    def get_all_zs(self, i=None):

        if i is None:
            return self.all_zs
        else:
            return self.all_zs[i]

    def get_eval_scores(self):
        return self.all_scores[self.train_offset:]


    def get_data(self):
        exit("don't call the 'state.get_data' function")
        return self.train_scores, self.train_zs, self.inlier_scores, self.inlier_zs, self.outlier_scores, self.outlier_zs, self.top2_closest_train, self.all_paths, self.eval_scores, self.emb_zs

    def save_data(self, fname):
        pickle.dump( self, open(fname, "wb" ) )

    def load_data(fname):
        return pickle.load( open( fname, "rb" ) )

    def get_dataset_type(self, index):
        if index < self.dr_train_zs.shape[0]:
            return "train"
        elif index < self.dr_train_zs.shape[0] + self.dr_inlier_zs.shape[0]:
            return "inlier"
        elif index < self.dr_train_zs.shape[0] + self.dr_inlier_zs.shape[0] + self.dr_outlier_zs.shape[0]:
            return "outlier"
        else:
            exit("bad index into state class")


    def get_json_friendly(self,index):

        ret = {
            'id': index,
            'datasettype':  self.get_dataset_type(index),
            'label': self.all_labels[index],
            'shiftlabel': self.all_shift_labels[index],
            'path': self.all_paths[index],
            "density_space_rank": self.all_rankings[index].item(),

            "score": self.all_scores[index].item(),
            "og_emb_x": self.og_emb_zs[index][0].item(),
            "og_emb_y": self.og_emb_zs[index][1].item(),

            "dr_emb_x": self.dr_emb_zs[index][0].item(),
            "dr_emb_y": self.dr_emb_zs[index][1].item(),
            "og_closest_trains": self.og_topk_closest_train[index],
            "dr_closest_trains": self.dr_topk_closest_train[index],
            "og_closest_evals":  self.og_topk_closest_test[index],
            "dr_closest_evals":  self.dr_topk_closest_test[index],
            "close_og_far_dr":   self.topk_closest_train_ratio[index]
        }

        return ret

    


    def get_simple_json(self,index, get_emb=False):
        def redu_prec(num):
            p = 3
            x = float(10**p)
            return float(int(num * x) + 1) / x

        ret = {
            "id": self.get_id(index),
            "filename": self.get_filename(index),
            "score": self.get_user_study_scores(index).item(),
        }

        if get_emb:
            if self.space_type_original:
                ret["emb_x"] = self.og_emb_zs[index][0].item()
                ret["emb_y"] = self.og_emb_zs[index][1].item()
            else:
                ret["emb_x"] = self.dr_emb_zs[index][0].item()
                ret["emb_y"] = self.dr_emb_zs[index][1].item()

        return ret
     



    