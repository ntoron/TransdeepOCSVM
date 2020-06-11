#majority of code is borrowed from https://github.com/djsutherland/opt-mmd

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from sklearn import metrics, svm, preprocessing
from sklearn.model_selection import train_test_split
import os
from sklearn.cluster import KMeans
from scipy.stats import percentileofscore
import multiprocessing
from functools import partial
import time


def get_bandwidth(y_scores, z_test):

    abn_idx = np.where(y_scores<np.percentile(y_scores,50))
    abn_idx2 = np.where(y_scores>=np.percentile(y_scores,50))
    x_mmd_ind = np.random.choice(np.squeeze(np.array(abn_idx)), min(500, np.array(abn_idx).shape[1]), replace=False)
    X_test = z_test[x_mmd_ind]
    y_mmd_ind = np.random.choice(np.squeeze(np.array(abn_idx2)), min(500, np.array(abn_idx).shape[1]), replace=False)
    Y_test = z_test[y_mmd_ind]

    from sklearn.metrics.pairwise import euclidean_distances
    sub = lambda feats, n: feats[np.random.choice(
        feats.shape[0], min(feats.shape[0], n), replace=False)]
    median_samples=1000
    Z = np.r_[sub(X_test, median_samples // 2), sub(Y_test, median_samples // 2)]
    D2 = euclidean_distances(Z, squared=True)
    upper = D2[np.triu_indices_from(D2, k=1)]
    kernel_width = np.median(upper, overwrite_input=True)
    bandwidth = np.sqrt(kernel_width / 2)

    return (bandwidth)



def rbf_mmd2(X, Y, sigma=0, biased=True):
    gamma = 1 / (2 * sigma**2)

    XX = np.dot(X, X.T)
    XY = np.dot(X, Y.T)
    YY = np.dot(Y, Y.T)

    X_sqnorms = np.diag(XX)
    Y_sqnorms = np.diag(YY)

    K_XY = np.exp(-gamma * (
            -2 * XY + X_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))
    K_XX = np.exp(-gamma * (
            -2 * XX + X_sqnorms[:, np.newaxis] + X_sqnorms[np.newaxis, :]))
    K_YY = np.exp(-gamma * (
            -2 * YY + Y_sqnorms[:, np.newaxis] + Y_sqnorms[np.newaxis, :]))

    if biased:
        mmd2 = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean()
    else:
        m = K_XX.shape[0]
        n = K_YY.shape[0]

        mmd2 = ((K_XX.sum() - m) / (m * (m - 1))
              + (K_YY.sum() - n) / (n * (n - 1))
              - 2 * K_XY.mean())
    return (mmd2)


def _find_optimal_threshold(x, mmd_left, mmd_right):

    f1 = interp1d(x, np.array(mmd_left),axis=0, fill_value="extrapolate")
    f2 = interp1d(x, np.array(mmd_right),axis=0, fill_value="extrapolate")

    def findIntersection(fun1,fun2,x0):
        return fsolve(lambda x : fun1(x) - fun2(x),x0)

    optimal_threshold = findIntersection(f1,f2,0.0)

    return (optimal_threshold)

def get_mmd(perc_value, y_scores, train_1, test_1, y_true, train_ind, test_path, bandwidth):

    abn_idx = np.where(y_scores<np.percentile(y_scores,perc_value))
    abn_tst_latent = test_1[abn_idx]
    kmeans = KMeans(n_clusters=1, random_state=0).fit(abn_tst_latent)
    train_1_prime = np.concatenate((train_1,kmeans.transform(train_1)),axis=1)
    test_1_prime = np.concatenate((test_1,kmeans.transform(test_1)),axis=1)
    cf = svm.OneClassSVM(gamma='scale', nu=0.1)
    cf.fit(train_1_prime[train_ind, :]) 
    y_scores_tmp_grid = cf.score_samples(test_1_prime)
    y_scores_tmp_grid = (y_scores_tmp_grid - min(y_scores_tmp_grid)) / (max(y_scores_tmp_grid) - min(y_scores_tmp_grid)) 
    auroc = metrics.roc_auc_score(y_true, y_scores_tmp_grid) 
    auprc = metrics.average_precision_score(y_true, y_scores_tmp_grid) 
    np.save(test_path+'/svm_aucroc1_grid_'+str(perc_value)+'.npy', auroc)
    np.save(test_path+'/svm_aucprc1_grid_'+str(perc_value)+'.npy', auprc) 
    abn_idx_left = np.where(y_scores<np.percentile(y_scores,5))
    abn_idx_right = np.where(y_scores>=np.percentile(y_scores,80))
    abn_idx_current = np.where((y_scores>=np.percentile(y_scores,perc_value)) & (y_scores<np.percentile(y_scores,perc_value+5)))
    mmd_ind_left = np.random.choice(np.squeeze(np.array(abn_idx_left)), 500, replace=False)
    X_mmd_left = test_1[mmd_ind_left]
    np.save(test_path+'/y_true_x_mmd_left_ind_grid_'+str(perc_value)+'.npy', y_true[mmd_ind_left])
    mmd_ind_right = np.random.choice(np.squeeze(np.array(abn_idx_right)), 500, replace=False)
    X_mmd_right = test_1[mmd_ind_right]
    np.save(test_path+'/y_true_x_mmd_right_ind_grid_'+str(perc_value)+'.npy', y_true[mmd_ind_right])
    mmd_ind_current=np.random.choice(np.squeeze(np.array(abn_idx_current)), 500, replace=False)
    X_mmd_current = test_1[mmd_ind_current]
    np.save(test_path+'/y_true_x_mmd_current_ind_grid_'+str(perc_value)+'.npy', y_true[mmd_ind_current])
    mmd_output_left = rbf_mmd2(X_mmd_left, X_mmd_current,sigma=bandwidth,biased=True)
    mmd_output_right = rbf_mmd2(X_mmd_right, X_mmd_current,sigma=bandwidth,biased=True)

    np.save(test_path+'/mmd_grid_left_'+str(perc_value)+ '.npy', mmd_output_left)
    np.save(test_path+'/mmd_grid_right_'+str(perc_value)+ '.npy', mmd_output_right)

    return (mmd_output_left, mmd_output_right)


def find_optimal_threshold(y_scores, train_1, test_1, y_true, train_ind, test_path, bandwidth):
    t0=time.time()
    
    a_pool = multiprocessing.Pool()
    result= a_pool.map(partial(get_mmd, y_scores=y_scores, train_1=train_1, test_1=test_1, y_true=y_true, train_ind=train_ind, test_path=test_path, bandwidth=bandwidth), np.arange(5,90,5))
    result=np.array([np.array(x) for x in result])
    mmd_left_all=result[:,0]
    mmd_right_all=result[:,1]
    x = np.arange(5,90,5) + 2.5
    optimal_threshold = _find_optimal_threshold(x,mmd_left_all, mmd_right_all)

    return (optimal_threshold)
