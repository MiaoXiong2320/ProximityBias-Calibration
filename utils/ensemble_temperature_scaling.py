'''
https://github.com/zhang64-llnl/Mix-n-Match-Calibration/blob/master/util_calibration.py
@inproceedings{zhang2020mix,
  author={Zhang, Jize and Kailkhura, Bhavya and Han, T},
  booktitle={International Conference on Machine Learning (ICML)},
  title = {Mix-n-Match: Ensemble and Compositional Methods for Uncertainty Calibration in Deep Learning},
  year = {2020},
}
This code is modified from Mix-n-Match-Calibration to implement ensemble temperature calibration.


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

import sys
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
# Imports to get "utility" package
# sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
# from utility.unpickle_probs import unpickle_probs
# from utility.evaluation import ECE, MCE


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
    
def mse_t(t, *args):
    ## find optimal temperature with MSE loss function

    logit, label = args
    logit = logit/t
    n = np.sum(np.exp(logit),1)  
    p = np.exp(logit)/n[:,None]
    mse = np.mean((p-label)**2)
    return mse


def ll_t(t, *args):
    ## find optimal temperature with Cross-Entropy loss function

    logit, label = args
    logit = logit/t
    n = np.sum(np.exp(logit),1)  
    p = np.clip(np.exp(logit)/n[:,None],1e-20,1-1e-20)
    N = p.shape[0]
    ce = -np.sum(label*np.log(p))/N
    return ce
      
class EnsembleTemperatureScaling():
    
    def __init__(self, temp = 1, loss_func='mse', solver = "L-BFGS-B"):
        """
        Initialize class
        
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.weight = None
        
        self.loss_func = loss_func
        self.solver = solver # BFGS
        


    def _log_loss_fun(self, x, logits, true):
        # x is the temperature variable to optimize
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_logits = self.temp_calibrate(logits, x)
        loss = log_loss(y_true=true, y_pred=scaled_logits, eps=1e-20)
        if np.isnan(loss).sum() > 0:
            print('np.isnan(loss).sum()', np.isnan(loss).sum())
        return loss

    def _mse_loss_func(self, x, logits, true):
        # x is the temperature variable to optimize
        # true: one-hot encoding [samples, classes]
        ## find optimal temperature with MSE loss function

        scaled_logits = self.temp_calibrate(logits, x)
        n = np.sum(np.exp(scaled_logits),1)  
        p = np.exp(scaled_logits)/n[:,None]
        mse = np.mean((p-true)**2)
        return mse


    
    # Find the temperature
    def _fit_temperature(self, logits, true):
        """
        Trains the model and finds optimal temperature
        
        Params:
            logits (shape [samples, classes]): the output from neural network for each class
            true: one-hot-encoding of true labels.
            
        Returns:
            the results of optimizer after minimizing is finished.
        """
        bnds = ((0.05, 5.0),)
        if self.loss_func == 'ce':  
            opt = minimize(ll_t, x0 = 1.0, args=(logits, true), method = self.solver, bounds=bnds, tol=1e-12)
        elif self.loss_func == 'mse':
            opt = minimize(mse_t, x0 = 1.0, args=(logits, true), method = self.solver, bounds=bnds, tol=1e-12)            
        self.temp = opt.x[0]
        print('ETS finished temp', self.temp)
       
        return self.temp
    
    def mse_w(self, w, *args):
        ## find optimal weight coefficients with MSE loss function

        p0, p1, p2, label = args
        p = w[0]*p0+w[1]*p1+w[2]*p2
        p = p/np.sum(p,1)[:,None]
        mse = np.mean((p-label)**2)   
        return mse


    def ll_w(self, w, *args):
        ## find optimal weight coefficients with Cros-Entropy loss function

        p0, p1, p2, label = args
        p = (w[0]*p0+w[1]*p1+w[2]*p2)
        N = p.shape[0]
        ce = -np.sum(label*np.log(p))/N
        return ce


    ##### Ftting Enseble Temperature Scaling
    def _fit_weights(self, logit, label, temp):
        # softmax
        p1 = softmax(logit)
        # calibrated softmax
        p0 = softmax(logit/temp)
        # label smooth
        n_class = logit.shape[1]
        p2 = np.ones_like(p0)/n_class
        

        bnds_w = ((0.0, 1.0),(0.0, 1.0),(0.0, 1.0),) # the variable should in this range [0,1]
        def my_constraint_fun(x): return np.sum(x)-1 # ensure the sum of w1+w2+w3=1
        constraints = { "type":"eq", "fun":my_constraint_fun}
        if self.loss_func == 'ce':
            w = minimize(self.ll_w, (1.0, 0.0, 0.0) , args = (p0,p1,p2,label), method='SLSQP', constraints = constraints, bounds=bnds_w, tol=1e-12, options={'disp': True})
        if self.loss_func == 'mse':
            w = minimize(self.mse_w, (1.0, 0.0, 0.0) , args = (p0,p1,p2,label), method='SLSQP', constraints = constraints, bounds=bnds_w, tol=1e-12, options={'disp': True})
        
        self.weight = w.x
        print("weight = " +str(self.weight))
        return self.weight

    ##### Calibration: Ensemble Temperature Scaling
    def fit(self, logit, label):
        # label is one-hot encoding
        n_classes = logit.shape[1]
        if len(label.shape) == 1:
            if n_classes == 2:
                one_hot_encoded_labels = np.zeros((len(label), n_classes))
                one_hot_encoded_labels[np.arange(len(label)), label.flatten()] = 1
                label = one_hot_encoded_labels
            elif n_classes > 2:
                label = label_binarize(label, classes=np.arange(n_classes))
        
        # these two terms can be estimated separately
        temp = self._fit_temperature(logit, label)
        weight = self._fit_weights(logit, label, temp)
        return 
    
    def fit_transform(self, logit, label):
        self.fit(logit, label)
        return self.transform(logit)
    
    def transform(self, logit):
        """
        Scales logits based on the ensemble composed temperature and returns calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        temp = self.temp
        w = self.weight
        n_class = logit.shape[1]
        
        # softmax
        p1 = softmax(logit)
        # calibrated softmax
        p0 = softmax(logit/temp)
        # label smooth
        n_class = logit.shape[1]
        p2 = np.ones_like(p0)/n_class
        
        p = w[0]*p0 + w[1]*p1 +w[2]*p2
        return p 
        
        
    def temp_calibrate(self, logits, temp = None):
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
        
    
class MultiIsotonicRegression():
    """multi-class isotonic regression adopted from https://github.com/zhang64-llnl/Mix-n-Match-Calibration/blob/e41afbaf8181a0bd2fb194f9e9d30bcbe5b7f6c3/util_calibration.py"""
    
    def __init__(self) -> None:
        self.__name__ = 'MultiIsotonicRegression'
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        
    def fit_transform(self, logit, label):
        # logit: [samples, classes]
        # label: [samples, classes]
        
        n_classes = logit.shape[1]
        if len(label.shape) == 1:
            if n_classes == 2:
                one_hot_encoded_labels = np.zeros((len(label), n_classes))
                one_hot_encoded_labels[np.arange(len(label)), label.flatten()] = 1
                label = one_hot_encoded_labels
            elif n_classes > 2:
                label = label_binarize(label, classes=np.arange(n_classes))
        
        p = np.exp(logit)/np.sum(np.exp(logit),1)[:,None] 
        y_ = self.calibrator.fit_transform(p.flatten(), (label.flatten()))
        p = y_.reshape(logit.shape) + 1e-9 * p
        
        return p
    
    def transform(self, logit):
        p = np.exp(logit)/np.sum(np.exp(logit),1)[:,None] 
        y_ = self.calibrator.predict(p.flatten())
        p = y_.reshape(logit.shape) + 1e-9 * p
        return p
        


