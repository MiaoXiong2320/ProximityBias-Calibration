'''
    https://github.com/markus93/NN_calibration/blob/eb235cdba006882d74a87114a3563a9efca691b7/scripts/utility/evaluation.py
    https://github.com/markus93/NN_calibration/blob/master/scripts/calibration/cal_methods.py
    
    This file contains the code for evaluation metrics:
    - ECE 
    - MCE
    - Dist-aware ECE
    - Dist-aware MCE
    ...
'''

import numpy as np
from scipy.optimize import minimize 
from sklearn.metrics import log_loss
import pandas as pd
import time, pdb
from sklearn.metrics import log_loss, brier_score_loss
import sklearn.metrics as metrics
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import average_precision_score, roc_auc_score, auc
import sys
from os import path
# from KDEpy import FFTKDE

import torch
import torch.nn as nn
import torch.nn.functional as F



def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin



   
def ECE(conf, pred, gt, conf_bin_num = 10):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  
        
    Returns:
        ece: expected calibration error
    """
    df = pd.DataFrame({'ys':gt, 'conf':conf, 'pred':pred})
    df['correct'] = (df.pred == df.ys).astype('int')
    

    bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
    df['conf_bin'] = df['conf'].apply(lambda x: np.digitize(x, bin_bounds))
    # df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='uniform').fit_transform(conf[:, np.newaxis])
    
    # groupy by knn + conf
    group_acc = df.groupby(['conf_bin'])['correct'].mean()
    group_confs = df.groupby(['conf_bin'])['conf'].mean()
    counts = df.groupby(['conf_bin'])['conf'].count()
    ece = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
        
    return ece
     
def PIECE(conf, knndist, pred, gt, dist_bin_num =10, conf_bin_num = 10, knn_strategy='quantile'):

    """
    Proximity Informed Expected Calibration Error 
    
    Args:
        conf (numpy.ndarray): list of confidences
        knndist (numpy.ndarray): list of distances of which a sample to its K nearest neighbors
        pred (numpy.ndarray): list of predictions
        gt (numpy.ndarray): list of true labels
        dist_bin_num: (float): the number of bins for knndist
        conf_bin_size: (float): size of one bin (0,1)  
        
    Returns:
        ece: expected calibration error
    """
    
    
    df = pd.DataFrame({'ys':gt, 'knndist':knndist, 'conf':conf, 'pred':pred})
    df['correct'] = (df.pred == df.ys).astype('int')
    df['knn_bin'] = KBinsDiscretizer(n_bins=dist_bin_num, encode='ordinal',strategy=knn_strategy).fit_transform(knndist[:, np.newaxis])
    
    bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
    df['conf_bin'] = df['conf'].apply(lambda x: np.digitize(x, bin_bounds))
    # df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='uniform').fit_transform(conf[:, np.newaxis])
    
    # groupy by knn + conf
    group_acc = df.groupby(['knn_bin', 'conf_bin'])['correct'].mean()
    group_confs = df.groupby(['knn_bin', 'conf_bin'])['conf'].mean()
    counts = df.groupby(['knn_bin', 'conf_bin'])['conf'].count()
    ece = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
    
    # group by only knn
    # group_acc = df.groupby(['knn_bin'])['correct'].mean()
    # group_confs = df.groupby(['knn_bin'])['conf'].mean()
    # counts = df.groupby(['knn_bin'])['conf'].count()
    # ece = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
    
    
    # n = len(conf)
    # ece = 0  # Starting error
    # upper_bounds = np.arange(conf_bin_size, 1+conf_bin_size, conf_bin_size)  # Get bounds of bins
    # for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
    #     acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-conf_bin_size, conf_thresh, conf, pred, gt)        
    #     ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE
        
    return ece


def MCE(conf, pred, gt, conf_bin_num = 10):

    """
    Maximal Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  
        
    Returns:
        mce: maximum calibration error
    """
    df = pd.DataFrame({'ys':gt, 'conf':conf, 'pred':pred})
    df['correct'] = (df.pred == df.ys).astype('int')

    bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
    df['conf_bin'] = df['conf'].apply(lambda x: np.digitize(x, bin_bounds))
    # df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='uniform').fit_transform(conf[:, np.newaxis])
    
    # groupy by knn + conf
    group_acc = df.groupby(['conf_bin'])['correct'].mean()
    group_confs = df.groupby(['conf_bin'])['conf'].mean()
    counts = df.groupby(['conf_bin'])['conf'].count()
    mce = (np.abs(group_acc - group_confs) * counts / len(df)).max()
        
    return mce



def AdaptiveECE(conf, pred, gt, conf_bin_num=10):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  
        
    Returns:
        ace: expected calibration error
    """
    df = pd.DataFrame({'ys':gt, 'conf':conf, 'pred':pred})
    df['correct'] = (df.pred == df.ys).astype('int')
    df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='quantile').fit_transform(conf[:, np.newaxis])
    
    # groupy by knn + conf
    group_acc = df.groupby(['conf_bin'])['correct'].mean()
    group_confs = df.groupby(['conf_bin'])['conf'].mean()
    counts = df.groupby(['conf_bin'])['conf'].count()
    ace = (np.abs(group_acc - group_confs) * counts / len(df)).sum()
        
    return ace
   


def mirror_1d(d, xmin=None, xmax=None):
    """If necessary apply reflecting boundary conditions."""
    if xmin is not None and xmax is not None:
        xmed = (xmin+xmax)/2
        return np.concatenate(((2*xmin-d[d < xmed]).reshape(-1,1), d, (2*xmax-d[d >= xmed]).reshape(-1,1)))
    elif xmin is not None:
        return np.concatenate((2*xmin-d, d))
    elif xmax is not None:
        return np.concatenate((d, 2*xmax-d))
    else:
        return d


def evaluate(probs, preds, y_true, knndist, verbose = False, normalize = False, conf_bins = 15, knn_bins=15):
    """
    test_prob_score, test_preds, test_ys, test_knndists
    Evaluate model using various scoring measures: Error Rate, ECE, MCE, NLL, Brier Score
    
    NLL is implemented using the log_loss function from sklearn.metrics
    
    Params:
        probs (samples, classes): probabilities for all the classes
        knndist: a list containing distances of which a sample to its K nearest neighbors (samples,)
        y_true: a list containing the actual class labels
        verbose: (bool) are the scores printed out. (default = False)
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized.
        conf_bins: (int) - into how many bins are probabilities divided (default = 15)
        knn_bins: (int) - into how many bins are knndists divided (default = 15)
        
    Returns:
        (error, ece, mce, loss, brier), returns various scoring measures
    """
    if normalize:
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        # Check if everything below or equal to 1?
        
    confs = probs[range(probs.shape[0]), preds]
    
    # Calculate ECE
    ece = ECE(confs, preds, y_true, conf_bin_num = conf_bins)
    mce = MCE(confs, preds, y_true, conf_bin_num = conf_bins)
    da_ece = PIECE(confs, knndist, preds, y_true, dist_bin_num = knn_bins, conf_bin_num = conf_bins)
    ace = AdaptiveECE(confs, preds, y_true, conf_bin_num = conf_bins)
    

    if verbose:
        print("ECE:", ece)
        print("MCE:", mce)
        print("DA-ECE:", da_ece)
        print("ACE:", ace)
    
    return (ece, mce, ace, da_ece)



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

    # return out

    # new
    x_ts = torch.tensor(x)
    return F.softmax(x_ts, dim=1).numpy()
    


################## FOR MISCLASSIFICATION ##################

metrics_to_use = ['accuracy', 'auc_roc', 'ap_success', 'ap_errors', "fpr_at_95tpr"] 


def get_misclassification_scores(pred, target, confidence, metrics=metrics_to_use, split="train"):
    """_summary_

    Args:
        pred (N,): _description_
        target (N,): _description_
        confidence (N,): _description_
        split (str, optional): _description_. Defaults to "train".

    Returns:
        _type_: _description_
    """
    
    accurate = (pred == target)
    errors = (pred != target)
    confs = confidence
    accuracy = accurate.mean().item()
    
    # accurate = np.reshape(accurate, newshape=(len(accurate), -1)).flatten()
    # errors = np.reshape(errors, newshape=(len(errors), -1)).flatten()
    # confs = np.reshape(confs, newshape=(len(confs), -1)).flatten()

    scores = {}
    if "accuracy" in metrics:
        scores[f"{split}/accuracy"] = {"value": accuracy, "string": f"{accuracy:05.2%}"}
        
    if "auc_roc" in metrics:
        if len(np.unique(accurate)) == 1:
            auc_score = 1
        else:
            auc_score = roc_auc_score(accurate, confs)
        scores[f"{split}/auc_roc"] = {"value": auc_score, "string": f"{auc_score:05.2%}"}
        
    if "ap_success" in metrics:
        ap_success = average_precision_score(accurate, confs)
        scores[f"{split}/ap_success"] = {"value": ap_success, "string": f"{ap_success:05.2%}"}
        
    if "accuracy_success" in metrics:
        accuracy_success = np.round(confs[accurate == 1]).mean()
        scores[f"{split}/accuracy_success"] = {
            "value": accuracy_success,
            "string": f"{accuracy_success:05.2%}",
        }
        
    if "ap_errors" in metrics:
        ap_errors = average_precision_score(errors, -confs)
        scores[f"{split}/ap_errors"] = {"value": ap_errors, "string": f"{ap_errors:05.2%}"}
        
    if "accuracy_errors" in metrics:
        # choose all false prediction's softmax score, compute average -> accuracy = 1 - averge
        accuracy_errors = 1.0 - np.round(confs[errors == 1]).mean()
        scores[f"{split}/accuracy_errors"] = {
            "value": accuracy_errors,
            "string": f"{accuracy_errors:05.2%}",
        }
        
    if "fpr_at_95tpr" in metrics:
        for i, delta in enumerate(np.arange(
            confs.min(),
            confs.max(),
            (confs.max() - confs.min()) / 10000,
        )):
            tpr = len(confs[(accurate == 1) & (confs >= delta)]) / len(
                confs[(accurate == 1)]
            )
            # if i%100 == 0:
            #     print(f"Threshold:\t {delta:.6f}")
            #     print(f"TPR: \t\t {tpr:.4%}")
            #     print("------")
            if 0.9505 >= tpr >= 0.9495:
                print(f"Nearest threshold 95% TPR value: {tpr:.6f}")
                print(f"Threshold 95% TPR value: {delta:.6f}")
                fpr = len(
                    confs[(errors == 1) & (confs >= delta)]
                ) / len(confs[(errors == 1)])
                scores[f"{split}/fpr_at_95tpr"] = {"value": fpr, "string": f"{fpr:05.2%}"}
                break
            
    if "aurc" in metrics:
        risks, coverages = [], []
        for delta in sorted(set(confs))[:-1]:
            coverages.append((confs > delta).mean())
            selected_accurate = accurate[confs > delta]
            risks.append(1. - selected_accurate.mean())
        aurc = auc(coverages, risks)
        eaurc = aurc - ((1. - accuracy) + accuracy*np.log(accuracy))
        scores[f"{split}/aurc"] = {"value": aurc, "string": f"{aurc*1000:01.2f}"}
        scores[f"{split}/e-aurc"] = {"value": eaurc, "string": f"{eaurc*1000:01.2f}"}
        
    return scores
