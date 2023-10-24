'''
This code is modified from Mix-n-Match-Calibration to implement Bin-Mean-Shift.
https://github.com/zhang64-llnl/Mix-n-Match-Calibration/blob/master/util_calibration.py
@inproceedings{zhang2020mix,
  author={Zhang, Jize and Kailkhura, Bhavya and Han, T},
  booktitle={International Conference on Machine Learning (ICML)},
  title = {Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning},
  year = {2020},
}


'''
import numpy as np
from scipy.optimize import minimize 
from sklearn.metrics import log_loss
import pandas as pd
import time, pdb
from sklearn.metrics import log_loss, brier_score_loss
import sklearn.metrics as metrics
from sklearn.preprocessing import label_binarize
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import KBinsDiscretizer

import sys
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
# Imports to get "utility" package
# sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
# from utility.unpickle_probs import unpickle_probs
# from utility.evaluation import ECE, MCE

from sklearn.cluster import KMeans


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    
    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """

    # old
    # e_x = np.exp(x - np.max(x))
    # # e_x = np.exp(x)
    # out = e_x / e_x.sum(axis=1, keepdims=1)
    # # if np.isnan(out).sum() > 0:
    # #     print('has nan:', np.isnan(out).sum())
    # #     exit()

    # old 
    # out = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]

    # new
    x_ts = torch.tensor(x)
    return F.softmax(x_ts, dim=1).numpy()


class MultiProximityIsotonicRegression():
    """
    1. use proximity to cluster data into several bins
    2. use multi-isotonic-regression to calibrate each bin
    
    multi-class isotonic regression adopted from https://github.com/zhang64-llnl/Mix-n-Match-Calibration/blob/e41afbaf8181a0bd2fb194f9e9d30bcbe5b7f6c3/util_calibration.py
    """
    
    def __init__(self, proximity_bin=10) -> None:
        self.proximity_bin = proximity_bin
        self.calibrators = [IsotonicRegression(out_of_bounds='clip') for i in range(proximity_bin)]
        
    def get_bin_edges_by_kmeans(self, proximity):
        # set initialization to kmeans @ TODO not necessary
        n_bins = self.proximity_bin 
        column = proximity
        col_min, col_max = column.min(), column.max()
        uniform_edges = np.linspace(col_min, col_max, n_bins + 1)
        init = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5
        # 1D k-means procedure
        km = KMeans(n_clusters=self.proximity_bin, init=init, n_init=1)

        centers = km.fit(column[:, None]).cluster_centers_[:, 0]

        # Must sort, centers may be unsorted even with sorted init
        centers.sort()
        bin_edges = (centers[1:] + centers[:-1]) * 0.5
        bin_edges = np.r_[col_min, bin_edges, col_max]
        return bin_edges


    def fit_transform(self, logit, proximity, label):
        # logit: [samples, classes]
        # label: [samples, classes]
        
        bin_edges = self.get_bin_edges_by_kmeans(proximity)
        bin_no = np.searchsorted(bin_edges[1:-1], proximity, side="right")
        
        # prepare data
        n_classes = logit.shape[1]
        if len(label.shape) == 1:
            if n_classes == 2:
                one_hot_encoded_labels = np.zeros((len(label), n_classes))
                one_hot_encoded_labels[np.arange(len(label)), label.flatten()] = 1
                label = one_hot_encoded_labels
            elif n_classes > 2:
                label = label_binarize(label, classes=np.arange(n_classes))   # one-hot encoding [samples, classes]
        p = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]   # softmax [samples, classes]
        
        y_ = np.stack([self.calibrators[no].fit_transform(p[idx, :].flatten(), (label[idx,: ].flatten())) for idx,no in enumerate(bin_no)], axis=0)
        p = y_.reshape(logit.shape) + 1e-9 * p
        
        return p
    
    def transform(self, logit, proximity):
        
        bin_edges = self.get_bin_edges_by_kmeans(proximity)
        bin_no = np.searchsorted(bin_edges[1:-1], proximity, side="right")
        
        p = np.exp(logit)/np.sum(np.exp(logit),1)[:,None] 
        y_ = np.stack([self.calibrators[no].predict(p[idx].flatten()) for idx,no in enumerate(bin_no)], axis=0)
        p = y_.reshape(logit.shape) + 1e-9 * p
        return p
    
    
    
class BinMeanShift():
    """
    This is a plug-and-play proximity-based method for binning-based calibration algorithms (e.g. isotonic regression).
        1. use proximity to cluster data into several bins
        2. use binning-based calibration algorithm (e.g. multi-isotonic-regression) to calibrate each bin
    
    """
    
    def __init__(self, method_name, method, bin_strategy='quantile', normalize_conf=False, proximity_bin=10, **kwargs) -> None:
        self.method_name = method_name
        self.proximity_bin = proximity_bin
        self.bin_strategy = bin_strategy
        self.normalize_conf = normalize_conf
        self.calibrators = [method(**kwargs) for i in range(proximity_bin)]
        
    def get_bin_edges_by_kmeans(self, proximity):
        # set initialization to kmeans @ TODO not necessary
        n_bins = self.proximity_bin 
        column = proximity
        col_min, col_max = column.min(), column.max()
        uniform_edges = np.linspace(col_min, col_max, n_bins + 1)
        init = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5
        # 1D k-means procedure
        km = KMeans(n_clusters=self.proximity_bin, init=init, n_init=1)

        centers = km.fit(column[:, None]).cluster_centers_[:, 0]

        # Must sort, centers may be unsorted even with sorted init
        centers.sort()
        bin_edges = (centers[1:] + centers[:-1]) * 0.5
        bin_edges = np.r_[col_min, bin_edges, col_max]
        return bin_edges
    
    def get_bin_edges_by_quantile(self, proximity):
        quantiles = np.linspace(0, 100, self.proximity_bin + 1)
        bin_edges = np.asarray(np.percentile(proximity, quantiles))
        
        # # Remove bins whose width are too small (i.e., <= 1e-8)
        # mask = np.ediff1d(bin_edges, to_begin=np.inf) > 1e-8
        # bin_edges = bin_edges[mask]
        # if len(bin_edges) - 1 != self.proximity_bin:
        #     print("Bins whose width are too small (i.e., <= 1e-8) are removed. Consider decreasing the number of bins.")
        #     # TODO what if there are multiple bins whose width are too small? then the number of bins will be smaller than self.proximity_bin - 1 
        #     self.proximity_bin = len(bin_edges) - 1
            
        return bin_edges

    def get_bin_edges_by_uniform(self, proximity):
        col_min, col_max = proximity.min(), proximity.max()
        bin_edges = np.linspace(col_min, col_max, self.proximity_bin + 1)
        
        # # Remove bins whose width are too small (i.e., <= 1e-8)
        # mask = np.ediff1d(bin_edges, to_begin=np.inf) > 1e-8
        # bin_edges = bin_edges[mask]
        # if len(bin_edges) - 1 != self.proximity_bin:
        #     print("Bins whose width are too small (i.e., <= 1e-8) are removed. Consider decreasing the number of bins.")
        #     # TODO what if there are multiple bins whose width are too small? then the number of bins will be smaller than self.proximity_bin - 1 
        #     self.proximity_bin = len(bin_edges) - 1
        return bin_edges


    def fit_transform(self, logit, proximity, label):
        # logit: [samples, classes]
        # label: [samples, classes]
        if self.bin_strategy == 'quantile':
            self.bin_edges = self.get_bin_edges_by_quantile(proximity)
        elif self.bin_strategy == 'kmeans':
            self.bin_edges = self.get_bin_edges_by_kmeans(proximity)  
        elif self.bin_strategy == 'uniform':
            self.bin_edges = self.get_bin_edges_by_uniform(proximity)  
            
        bin_no = np.searchsorted(self.bin_edges[1:-1], proximity, side="right")
        
        # compute every bins' samples' indices
        conf_indices = []
        for b in range(self.proximity_bin):  # max(bin_no) is the number of bins
            indices = [i for i, x in enumerate(bin_no) if x == b]  # get indices where bin_no is b
            conf_indices.append(indices)
            
        if self.method_name in ['histogram_binning', 'isotonic_regression']:
            logit = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]
        probs = np.concatenate([self.calibrators[no].fit_transform(logit[idx], label[idx]) for no,idx in enumerate(conf_indices)])

        if self.normalize_conf:
            probs = probs / np.sum(probs, axis=1)[:, None]
        
        index = np.argsort(np.concatenate(conf_indices))
        probs = probs[index]
        
        return probs
    
    def transform(self, logit, proximity):

        bin_no = np.searchsorted(self.bin_edges[1:-1], proximity, side="right")
        
        # compute every bins' samples' indices
        conf_indices = []
        for b in range(self.proximity_bin):  # max(bin_no) is the number of bins
            indices = [i for i, x in enumerate(bin_no) if x == b]  # get indices where bin_no is b
            conf_indices.append(indices)

        if self.method_name in ['histogram_binning', 'isotonic_regression']:
            logit = np.exp(logit)/np.sum(np.exp(logit),1)[:,None]        
        probs = np.concatenate([self.calibrators[no].transform(logit[idx]) for no,idx in enumerate(conf_indices)])
        
        index = np.argsort(np.concatenate(conf_indices))
        probs = probs[index]
        
        return probs

