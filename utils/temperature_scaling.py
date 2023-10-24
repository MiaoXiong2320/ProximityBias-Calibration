'''
    https://github.com/markus93/NN_calibration/blob/eb235cdba006882d74a87114a3563a9efca691b7/scripts/utility/evaluation.py
    https://github.com/markus93/NN_calibration/blob/master/scripts/calibration/cal_methods.py
    
    This file contains the code for the classic calibration methods and metrics:
    - temperature scaling
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
from sklearn.metrics import log_loss, brier_score_loss, mean_absolute_error
import sklearn.metrics as metrics
from sklearn.preprocessing import KBinsDiscretizer
import sys
from os import path
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.nn.functional as F
# Imports to get "utility" package
# sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
# from utility.unpickle_probs import unpickle_probs
# from utility.evaluation import ECE, MCE


def temperature_scale(logits, temp):
    temperature = temp.unsqueeze(1).expand(logits.size(0), logits.size(1)).cuda()
    return logits / temperature

def search_temperature(logits, targets):
    criterion = nn.CrossEntropyLoss()
    nlls = []
    temps = []
    logits = torch.tensor(logits).float().cuda()
    targets = torch.tensor(targets).long().cuda()
    for i in range(1, 500):
        temp = torch.ones(1) * i/100
        temps.append(temp.item())
        nlls.append(criterion(temperature_scale(logits, temp).cuda(), targets).item())
    best_idx = np.argmin(nlls)
    best_temp = temps[best_idx]
    best_nll = nlls[best_idx]
    print('\nSearched Temperature on Validation Data: ', round(best_temp,4), 'nll: ', best_nll)
    return best_temp, best_nll


def _compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, correct):
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
    filtered_tuples = [x for x in zip(correct, conf) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == 1])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[1] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin

def eval_ece(conf, corrects, bin_size = .1):
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins

    n = len(conf)
    ece = 0  # Starting error
    
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = _compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, corrects)        
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE
        
    return ece

def eval_mce(conf, corrects, bin_size = 0.1):

    """
    Maximal Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        mce: maximum calibration error
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    
    cal_errors = []
    
    for conf_thresh in upper_bounds:
        acc, avg_conf, _ = _compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, corrects)
        cal_errors.append(np.abs(acc-avg_conf))
        
    return max(cal_errors)


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
    


class HistogramBinning():
    """
    Histogram Binning as a calibration method. The bins are divided into equal lengths.
    
    The class contains two methods:
        - fit(probs, true), that should be used with validation data to train the calibration model.
        - predict(probs), this method is used to calibrate the confidences.
    """
    
    def __init__(self, M=15):
        """
        M (int): the number of equal-length bins used
        """
        self.bin_size = 1./M  # Calculate bin size
        self.conf = []  # Initiate confidence list
        self.upper_bounds = np.arange(self.bin_size, 1+self.bin_size, self.bin_size)  # Set bin bounds for intervals

    
    def _get_conf(self, conf_thresh_lower, conf_thresh_upper, probs, true):
        """
        Inner method to calculate optimal confidence for certain probability range
        
        Params:
            - conf_thresh_lower (float): start of the interval (not included)
            - conf_thresh_upper (float): end of the interval (included)
            - probs : list of probabilities.
            - true : list with true labels, where 1 is positive class and 0 is negative).
        """

        # Filter labels within probability range
        filtered = [x[0] for x in zip(true, probs) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
        nr_elems = len(filtered)  # Number of elements in the list.

        if nr_elems < 1:
            return 0
        else:
            # In essence the confidence equals to the average accuracy of a bin
            conf = sum(filtered)/nr_elems  # Sums positive classes
            return conf
    

    def fit(self, probs, true):
        """
        Fit the calibration model, finding optimal confidences for all the bins.
        
        Params:
            probs: probabilities of data
            true: true labels of data
        """

        conf = []

        # Got through intervals and add confidence to list
        for conf_thresh in self.upper_bounds:
            temp_conf = self._get_conf((conf_thresh - self.bin_size), conf_thresh, probs = probs, true = true)
            conf.append(temp_conf)

        self.conf = conf


    # Fit based on predicted confidence
    def predict(self, probs):
        """
        Calibrate the confidences
        
        Param:
            probs: probabilities of the data (shape [samples, classes])
            
        Returns:
            Calibrated probabilities (shape [samples, classes])
        """

        # Go through all the probs and check what confidence is suitable for it.
        for i, prob in enumerate(probs):
            idx = np.searchsorted(self.upper_bounds, prob)
            probs[i] = self.conf[idx]

        return probs
        
        
class TemperatureScaling():
    
    def __init__(self, temp = 1, maxiter = 50, solver = "BFGS"):
        """
        Initialize class
        
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
    
    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x)
        loss = log_loss(y_true=true, y_pred=scaled_probs, labels=range(probs.shape[1]))
        if np.isnan(loss).sum() > 0:
            print('np.isnan(loss).sum()', np.isnan(loss).sum())
        return loss
    
    # Find the temperature
    def fit(self, logits, true):
        """
        Trains the model and finds optimal temperature
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: true labels(n_samples,).
            
        Returns:
            the results of optimizer after minimizing is finished.
        """
        # self.n_classes = logits.shape[1]
        # if len(true.shape) == 1:
        #     true = label_binarize(true, classes=np.arange(self.n_classes))
        
        true = true.flatten() # Flatten y_val
        opt = minimize(self._loss_fun, x0 = 1, args=(logits, true), options={'maxiter':self.maxiter}, method = self.solver)
        self.temp = opt.x[0]
        print('finished temp', self.temp)
        
        return opt
        
    def predict(self, logits, temp = None):
        """
        Scales logits based on the temperature and returns calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        if not temp:
            return softmax(logits/self.temp)
        else:
            return softmax(logits/temp)


class MyConf():
    def __init__(self) -> None:
        self.__name__ = 'MyConf'
        
    def fit_transform(self, logit, label):
        return softmax(logit)
    
    def transform(self, logit):
        return softmax(logit)

class MyTemperatureScaling():
    def __init__(self) -> None:
        self.__name__ = 'MyTemperatureScaling'
        self.calibrator = TemperatureScaling(maxiter=100)
        
    def fit_transform(self, logit, label):
        # logit: [samples, classes]
        # label: [samples, classes]

        self.calibrator.fit(logit, label)
        return self.calibrator.predict(logit)
    
    def transform(self, logit):
        return self.calibrator.predict(logit)

class TemperatureScalingWithDistAware():
    
    def __init__(self, num_neigh, maxiter = 50, solver = "BFGS"):
        """
        Initialize class
        
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.K = num_neigh
        self.temp = np.ones(3) # np.ones(self.K+1)
        self.maxiter = maxiter
        self.solver = solver

    def predict(self, logits, dists, temp = None):
        """
        Scales logits based on the temperature and returns calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            dists: distance to K nearest neighbors (shape [samples, K])
            temp: if not set use temperatures find by model or previously set.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        if temp is None:
            temp = self.temp

        # a^T d @TODO this method is ineffective
        # temp_ssl = np.matmul(dists, temp[0:-1, np.newaxis]) + temp[-1] 
        
        # use mean and variance as the information
        means = np.mean(dists, axis=-1)
        vars = np.var(dists, axis=-1)
        dummy = np.ones_like(vars)
        vecs = np.concatenate([means[:, np.newaxis], vars[:, np.newaxis], dummy[:, np.newaxis]], axis=-1)
        
        temp_ssl = np.matmul(vecs, temp[:, np.newaxis])
        
        return softmax(logits * temp_ssl)
    
    
    def _loss_fun(self, temp, logits, true, dists):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(logits, dists, temp)
        
        loss = log_loss(y_true=true, y_pred=scaled_probs)
 
        return loss
    
    # Find the temperature
    def fit(self, logits, labels, dists):
        """
        Trains the model and finds optimal temperature
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            labels: true labels. (shape [samples,])
            
        Returns:
            the results of optimizer after minimizing is finished.
        """

        # gts = labels.flatten() # Flatten y_val
       
        fit_nums = 3 # dists.shape[1] + 1 

        opt = minimize(self._loss_fun, x0 = np.ones(fit_nums)/2, args=(logits, labels, dists), options={'maxiter':self.maxiter}, method = self.solver)
        self.temp = opt.x
        print("learned temperature parameters: ", self.temp)
        
        return opt


