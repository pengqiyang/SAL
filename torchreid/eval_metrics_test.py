from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import pdb
import os
import copy
from collections import defaultdict
import sys
import warnings
import shutil

try:
    from torchreid.eval_cylib.eval_metrics_cy import evaluate_cy
    IS_CYTHON_AVAI = True
    print("Using Cython evaluation code as the backend")
except ImportError:
    IS_CYTHON_AVAI = False
    warnings.warn("Cython evaluation is UNAVAILABLE, which is highly recommended")


def eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank):
    """Evaluation with cuhk03 metric
    Key: one image for each gallery identity is randomly sampled for each query identity.
    Random sampling is performed num_repeats times.
    """
    num_repeats = 10
    num_q, num_g = distmat.shape
    
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0. # number of valid query
    
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        #raw_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        kept_g_pids = g_pids[order][keep]
        g_pids_dict = defaultdict(list)
        for idx, pid in enumerate(kept_g_pids):
            g_pids_dict[pid].append(idx)

        cmc, AP = 0., 0.
        for repeat_idx in range(num_repeats):
            mask = np.zeros(len(raw_cmc), dtype=np.bool)
            for _, idxs in g_pids_dict.items():
                # randomly sample one image for each gallery person
                rnd_idx = np.random.choice(idxs)
                mask[rnd_idx] = True
            masked_raw_cmc = raw_cmc[mask]
            _cmc = masked_raw_cmc.cumsum()
            _cmc[_cmc > 1] = 1
            cmc += _cmc[:max_rank].astype(np.float32)
            # compute AP
            num_rel = masked_raw_cmc.sum()
            tmp_cmc = masked_raw_cmc.cumsum()
            tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
            tmp_cmc = np.asarray(tmp_cmc) * masked_raw_cmc
            AP += tmp_cmc.sum() / num_rel
        
        cmc /= num_repeats
        AP /= num_repeats
        all_cmc.append(cmc)
        all_AP.append(AP)
        num_valid_q += 1.

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP




def eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, q_path, g_path, q_attrs, g_attrs, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    1.快速版本的map计算有误
    2.没有滤除统一camer id
    3.注意将gallery中没有的id从query滤除
    """
    #pdb.set_trace()
    num_q, num_g = distmat.shape
    
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    #pdb.set_trace()
    indices = np.argsort(distmat, axis=1)
    #matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    matches =  (g_pids[indices[:,:30]] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    false_pid=[]
    num_valid_q = 0. # number of valid query
    g_path_save = []
    q_attr_save = []
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        '''
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        '''
        # compute cmc curve
        #raw_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        raw_cmc = matches[q_idx]
        temp_path = g_path[order]
        #temp_attr = q_attrs[order]
        
        
        
        '''
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            #clear_q_pid.append(q_idx)
            print("have")
            continue
        '''      
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        #pdb.set_trace()
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        #save the false image
        #print(cmc[0])
        if cmc[0]==1:
            #pdb.set_trace()
            
            q_attr_save.append(q_attrs[q_idx])
            g_path_save.append(temp_path[0])
            print(q_attrs[q_idx])
            print(temp_path[0])
            #print(temp_attr[0])
            #false_pid.append(q_pid)
            '''
            os.makedirs('/media/data1/pengqy/attributed_based_person_seach/SAL/results/'+str(num_valid_q))
            shutil.copy(q_path[q_idx], '/media/data1/pengqy/attributed_based_person_seach/SAL/results/'+str(num_valid_q)+'/query.jpg')
            shutil.copy(temp_path[0], '/media/data1/pengqy/attributed_based_person_seach/SAL/results/'+str(num_valid_q)+'/1.jpg')
            shutil.copy(temp_path[1], '/media/data1/pengqy/attributed_based_person_seach/SAL/results/'+str(num_valid_q)+'/2.jpg')
            shutil.copy(temp_path[2], '/media/data1/pengqy/attributed_based_person_seach/SAL/results/'+str(num_valid_q)+'/3.jpg')
            shutil.copy(temp_path[3], '/media/data1/pengqy/attributed_based_person_seach/SAL/results/'+str(num_valid_q)+'/4.jpg')
            shutil.copy(temp_path[4], '/media/data1/pengqy/attributed_based_person_seach/SAL/results/'+str(num_valid_q)+'/5.jpg')
            '''
        
        
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    np.save('src/pa100k_extra_predict_true_sems.npy', q_attr_save)
    np.save('src/pa100k_extra_predict_true_path.npy', g_path_save)
    print(len(g_path_save))
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
   
    return all_cmc, mAP



def eval_market1501_version1(distmat, q_pids, g_pids, q_camids, g_camids, q_path, g_path, q_attrs, g_attrs, max_rank):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    1.快速版本的map计算有误
    2.没有滤除统一camer id
    3.注意将gallery中没有的id从query滤除
    """
    num_q, num_g = distmat.shape
    
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    #pdb.set_trace()
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    
    #matches =  (g_pids[indices[:,:30]] == q_pids[:, np.newaxis]).astype(np.int32)
    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    false_pid=[]
    num_valid_q = 0. # number of valid query

    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        '''
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        '''
        # compute cmc curve
        #raw_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        raw_cmc = matches[q_idx]
        #temp_path = g_path[order][keep]
        #temp_attr = g_attrs[order][keep]
        
        
        
        
        if not np.any(raw_cmc):
            # this condition is true when query identity does not appear in gallery
            #clear_q_pid.append(q_idx)
            print("have")
            continue
              
        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)
        #save the false image

        
    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"
    #np.save('query_pid.npy', false_pid)
    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
   
    return all_cmc, mAP


def evaluate_py(distmat, q_pids, g_pids, q_camids, g_camids, q_path, g_path, q_attrs, g_attrs, max_rank, use_metric_cuhk03, test_origin=False):
    if use_metric_cuhk03:
        return eval_cuhk03(distmat, q_pids, g_pids, q_camids, g_camids, max_rank)
    else:
        if test_origin:
            return eval_market1501_version1(distmat, q_pids, g_pids, q_camids, g_camids, q_path, g_path,  q_attrs, g_attrs, max_rank)
        return eval_market1501(distmat, q_pids, g_pids, q_camids, g_camids, q_path, g_path,  q_attrs, g_attrs, max_rank)


def evaluate(distmat, q_pids, g_pids, q_camids, g_camids, q_path, g_path, q_attrs, g_attrs, max_rank=50, use_metric_cuhk03=False, use_cython=True, test_origin=False):
    if use_cython and IS_CYTHON_AVAI:
        return evaluate_cy(distmat, q_pids, g_pids, q_camids, g_camids, max_rank, use_metric_cuhk03)
    else:
        return evaluate_py(distmat, q_pids, g_pids, q_camids, g_camids, q_path, g_path, q_attrs, g_attrs, max_rank, use_metric_cuhk03, test_origin=test_origin)
        
