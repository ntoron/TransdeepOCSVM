"""
TRAIN GANOMALY

. Example: Run the following command from the terminal.
    run train.py                             \
        --model ganomaly                        \
        --dataset UCSD_Anomaly_Dataset/UCSDped1 \
        --batchsize 32                          \
        --isize 256                         \
        --nz 512                                \
        --ngf 64                               \
        --ndf 64
"""


##
# LIBRARIES
from __future__ import print_function

from options import Options
from lib.data import load_data
from lib.model import Ganomaly

import itertools
import numpy as np 
import pandas as pd
import torch 
from sklearn import metrics, svm, preprocessing
from sklearn.model_selection import train_test_split
import os
from sklearn.cluster import KMeans
import time
from sklearn.preprocessing import StandardScaler, Normalizer
from mmd_utilities import *


##
def train():
    """ Training
    """

    ##
    # ARGUMENTS
    opt = Options().parse()

    ##
    # LOAD DATA
    dataloader = load_data(opt)
    ##
    # LOAD MODEL
    model = Ganomaly(opt, dataloader)
    ##
    # TRAIN MODEL
    model.train()


    train_1 = model.train_final()
    train_1 = train_1.cpu().numpy() 

    test_1, y_true, y_true_original, auroc_value, auprc_value = model.test_final()
    test_1 = test_1.cpu().numpy() 
    y_true = y_true.cpu().numpy() 
    y_true_original = y_true_original.cpu().numpy()  


    test_path = os.path.join(opt.outf,  opt.dataset,'test', 'OCSVM','abnormal'+str(opt.abnormal_class),'seed'+str(opt.manualseed))
    if not os.path.isdir(test_path):
        os.makedirs(test_path)

    print("GANomaly AUROC: {}".format(auroc_value))
    np.save(test_path+'/ganomaly_aucroc.npy', auroc_value)


    for i in range(len(y_true)): 
        if y_true[i] == 1: 
            y_true[i] = 0
        else:   
            y_true[i] = 1 

    
    ################################ 

    cf = svm.OneClassSVM(gamma='scale', nu=0.1)
    train_ind = np.random.choice(train_1.shape[0], 10000, replace=False)
    cf.fit(train_1[train_ind, :]) 
    y_scores = cf.score_samples(test_1)
    y_scores = (y_scores - min(y_scores)) / (max(y_scores) - min(y_scores)) 
    
    auroc = metrics.roc_auc_score(y_true, y_scores) 
    print("HybridGAN AUROC: {}".format(auroc))
    np.save(test_path+'/svm_aucroc1.npy', auroc)
    np.save(test_path+'/svm_aucroc1_transduct_'+str(0)+'.npy', auroc)


    bandwidth=get_bandwidth(y_scores, test_1)

    for trans_iter in np.arange(0,30,1):

        optimal_threshold = find_optimal_threshold(y_scores=y_scores, train_1=train_1, test_1=test_1, y_true=y_true, train_ind=train_ind, test_path=test_path, bandwidth=bandwidth)
        abn_idx = np.where(y_scores<np.percentile(y_scores,optimal_threshold))
        abn_tst_latent = test_1[abn_idx]
        kmeans = KMeans(n_clusters=1, random_state=0).fit(abn_tst_latent)
        train_1 = np.concatenate((train_1,kmeans.transform(train_1)),axis=1)
        test_1 = np.concatenate((test_1,kmeans.transform(test_1)),axis=1)
        cf = svm.OneClassSVM(gamma='scale', nu=0.1)
        cf.fit(train_1[train_ind, :]) 
        y_scores = cf.score_samples(test_1)
        y_scores = (y_scores - min(y_scores)) / (max(y_scores) - min(y_scores)) 
        auroc = metrics.roc_auc_score(y_true, y_scores) 
        print("TransdeepOCSVM AUROC after {} iterations: {}".format(trans_iter+1, auroc))
        print("Optimal_threshold after {} iterations: {}".format(trans_iter+1, optimal_threshold[0]))
        np.save(test_path+'/svm_aucroc1_transduct_'+str(trans_iter+1)+'.npy', auroc)
        np.save(test_path+'/optimal_threshold_'+ str(trans_iter+1) +'.npy',optimal_threshold)


if __name__ == '__main__':
    train()
