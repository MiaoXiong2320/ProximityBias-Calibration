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



def DA_MCE(conf, knndist, pred, gt, dist_bin_num =10, conf_bin_num = 10, knn_strategy='quantile'):

    """
    Distance-Aware Maximal Calibration Error
    
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
    df['knn_bin'] = KBinsDiscretizer(n_bins=dist_bin_num, encode='ordinal',strategy='quantile').fit_transform(knndist[:, np.newaxis])
    
    bin_bounds = np.linspace(0, 1, conf_bin_num + 1)[1:-1]
    df['conf_bin'] = df['conf'].apply(lambda x: np.digitize(x, bin_bounds))    
    # df['conf_bin'] = KBinsDiscretizer(n_bins=conf_bin_num, encode='ordinal',strategy='uniform').fit_transform(conf[:, np.newaxis])
    
    # group by knn + conf
    group_acc = df.groupby(['knn_bin', 'conf_bin'])['correct'].mean()
    group_confs = df.groupby(['knn_bin', 'conf_bin'])['conf'].mean()
    damce = (np.abs(group_acc - group_confs)).max()
    
    return damce

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
   


def ToplabelECE(conf, gt, pred=None, conf_bin_num=15):
    """
    Top Label Expected Calibration Error
    Adapted from https://github.com/aigen/df-posthoc-calibration/blob/main/assessment.py
    
    Args:
        if pred is not None:
            conf (numpy.ndarray): list of confidences (n_samples)
            gt (numpy.ndarray): list of true labels (n_samples)
            pred (numpy.ndarray): list of predictions (n_samples)
        elif pred is None:
            conf (numpy.ndarray): logits (n_samples, n_classes)
            gt (numpy.ndarray): list of true labels (n_samples)
        bin_size: (float): size of one bin (0,1)  
        
    Returns:
        mce: maximum calibration error
    """    
    if pred is not None:
        conf = conf.squeeze()
        pred = pred.squeeze()
        gt = gt.squeeze()
        assert(np.size(conf.shape) == 1), "Check dimensions of input matrices"
        assert(conf.shape == pred.shape), "Check dimensions of input matrices"
        assert(gt.shape == pred.shape), "Check dimensions of input matrices"
        # assert(np.min(gt) >= 1), "Labels should be numbered 1 ... L"
        
        labels = np.unique(pred)
        
        tl_ece = 0
        for l in labels:
            l_inds = np.argwhere(pred == l)
            class_ece = ECE(conf[l_inds].squeeze(-1), pred[l_inds].squeeze(-1), gt[l_inds].squeeze(-1), conf_bin_num = conf_bin_num)
            tl_ece += l_inds.size * class_ece
            
        tl_ece = tl_ece / pred.size
        
        return tl_ece
    
    else:
        gt = gt.squeeze()
        # assert(np.min(gt) >= 1), "Labels should be numbered 1 ... L"
        assert(np.size(conf.shape) == 2), "Prediction matrix should be 2 dimensional"
        assert(gt.size == conf.shape[0]), "Check dimensions of input matrices"
        
        confidences = np.max(softmax(conf, axis=-1), axis=1)
        pred = np.argmax(conf, axis=1)
        
        return ToplabelECE(gt, confidences, pred, conf_bin_num)






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


def KDE_ECE(p, label, p_int=None, order=1):
    """
    p is the confidence: [n_samples, n_classes]
    label is the one-hot label: [n_samples, n_classes]
    """

    # points from numerical integration
    if p_int is None:
        p_int = np.copy(p)

    p = np.clip(p,1e-256,1-1e-256)
    p_int = np.clip(p_int,1e-256,1-1e-256)
    
    
    x_int = np.linspace(-0.6, 1.6, num=2**14)
    
    
    N = p.shape[0]

    # this is needed to convert labels from one-hot to conventional form
    label_index = np.array([np.where(r==1)[0][0] for r in label])
    with torch.no_grad():
        # number of classes is larger than 2
        if p.shape[1] !=2:
            p_new = torch.from_numpy(p)
            p_b = torch.zeros(N,1)
            label_binary = np.zeros((N,1))
            for i in range(N):
                pred_label = int(torch.argmax(p_new[i]).numpy())
                if pred_label == label_index[i]:
                    label_binary[i] = 1
                p_b[i] = p_new[i,pred_label]/torch.sum(p_new[i,:])  
        else:
            p_b = torch.from_numpy((p/np.sum(p,1)[:,None])[:,1])
            label_binary = label_index
                
    method = 'triweight'
    
    dconf_1 = (p_b[np.where(label_binary==1)].reshape(-1,1)).numpy()
    kbw = np.std(p_b.numpy())*(N*2)**-0.2
    kbw = np.std(dconf_1)*(N*2)**-0.2
    # Mirror the data about the domain boundary
    low_bound = 0.0
    up_bound = 1.0
    dconf_1m = mirror_1d(dconf_1,low_bound,up_bound)
    # Compute KDE using the bandwidth found, and twice as many grid points
    pp1 = FFTKDE(bw=kbw, kernel=method).fit(dconf_1m).evaluate(x_int)
    pp1[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
    pp1[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
    pp1 = pp1 * 2  # Double the y-values to get integral of ~1
    
    
    p_int = p_int/np.sum(p_int,1)[:,None]
    N1 = p_int.shape[0]
    with torch.no_grad():
        p_new = torch.from_numpy(p_int)
        pred_b_int = np.zeros((N1,1))
        if p_int.shape[1]!=2:
            for i in range(N1):
                pred_label = int(torch.argmax(p_new[i]).numpy())
                pred_b_int[i] = p_int[i,pred_label]
        else:
            for i in range(N1):
                pred_b_int[i] = p_int[i,1]

    low_bound = 0.0
    up_bound = 1.0
    pred_b_intm = mirror_1d(pred_b_int,low_bound,up_bound)
    # Compute KDE using the bandwidth found, and twice as many grid points
    pp2 = FFTKDE(bw=kbw, kernel=method).fit(pred_b_intm).evaluate(x_int)
    pp2[x_int<=low_bound] = 0  # Set the KDE to zero outside of the domain
    pp2[x_int>=up_bound] = 0  # Set the KDE to zero outside of the domain
    pp2 = pp2 * 2  # Double the y-values to get integral of ~1

    
    if p.shape[1] !=2: # top label (confidence)
        perc = np.mean(label_binary)
    else: # or joint calibration for binary cases
        perc = np.mean(label_index)
            
    integral = np.zeros(x_int.shape)
    reliability= np.zeros(x_int.shape)
    for i in range(x_int.shape[0]):
        conf = x_int[i]
        if np.max([pp1[np.abs(x_int-conf).argmin()],pp2[np.abs(x_int-conf).argmin()]])>1e-6:
            accu = np.min([perc*pp1[np.abs(x_int-conf).argmin()]/pp2[np.abs(x_int-conf).argmin()],1.0])
            if np.isnan(accu)==False:
                integral[i] = np.abs(conf-accu)**order*pp2[i]  
                reliability[i] = accu
        else:
            if i>1:
                integral[i] = integral[i-1]

    ind = np.where((x_int >= 0.0) & (x_int <= 1.0))
    return np.trapz(integral[ind],x_int[ind])/np.trapz(pp2[ind],x_int[ind])




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
    # new_preds is the prediction of the model after the calibration
    new_preds = np.argmax(probs, axis=1)
    
    accuracy = metrics.accuracy_score(y_true, preds) * 100
    new_accuracy = metrics.accuracy_score(y_true, new_preds) * 100
    accuracy_change = new_accuracy - accuracy
    
    # Calculate ECE
    ece = ECE(confs, preds, y_true, conf_bin_num = conf_bins)
    mce = MCE(confs, preds, y_true, conf_bin_num = conf_bins)
    da_ece = PIECE(confs, knndist, preds, y_true, dist_bin_num = knn_bins, conf_bin_num = conf_bins)
    da_mce = DA_MCE(confs, knndist, preds, y_true, dist_bin_num = knn_bins, conf_bin_num = conf_bins)
    ace = AdaptiveECE(confs, preds, y_true, conf_bin_num = conf_bins)
    toplabel_ece = ToplabelECE(confs, y_true, preds, conf_bin_num = conf_bins)
    
    
    # nll loss
    # true_confs = probs[range(probs.shape[0]), y_true]
    loss = log_loss(y_true=y_true, y_pred=probs, labels=range(probs.shape[1]))
    
    # brier score
    y_true_onehot = np.zeros(probs.shape)
    y_true_onehot[np.arange(y_true.size), y_true] = 1
    brier = brier_score_loss(y_true_onehot.flatten(), probs.flatten())
    
    auc = roc_auc_score(y_true == preds, confs)

    # y_true_onehot = np.zeros((y_true.size, y_true.max() + 1))
    # y_true_onehot[np.arange(y_true.size), y_true] = 1
    # brier = brier_score_loss(y_true_onehot.flatten(), probs.flatten())

    if verbose:
        print("Accuracy change:", accuracy_change)
        print("ECE:", ece)
        print("MCE:", mce)
        print("DA-ECE:", da_ece)
        print("DA-MCE:", da_mce)
        print("ACE:", ace)
        print("Loss:", loss)
        print("Toplabel-ECE:", toplabel_ece)
        print("brier:", brier)
        print("AUCROC:", auc)
    
    return (accuracy_change, ece, mce, ace, toplabel_ece, da_ece, da_mce, loss, brier, auc)



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
