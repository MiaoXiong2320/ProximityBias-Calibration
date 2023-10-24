'''
This file contains the code of the density estimation calibration method. 
'''
import sys, math
import numpy as np
import pandas as pd
import time, pdb
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.preprocessing import normalize


import torch
import torch.nn as nn
import torch.nn.functional as F
# Imports to get "utility" package
# sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
# from utility.unpickle_probs import unpickle_probs
# from utility.evaluation import ECE, MCE

import statsmodels.api as sm
from scipy.special import softmax
        
# import KDEpy as kde
from scipy.stats import gaussian_kde
from sklearn.neighbors import KernelDensity


class DensityRatioCalibration():
    
    def __init__(self):
        pass 
        
        
    # Find the temperature
    def fit(self, probs, preds, true, proximity, bandwidth='normal_reference'):
        """
        Train the density estimator for correctly classified samples and misclassified samples
        1. split samples into correctly classified and misclassified
        3. count their numbers and show the ratio   
        4. learn the conditional distribution of the confidence given the proximity -> the distributions are represented as <dens_true, dens_false>
        5. Compute acc_ratio = p(correct=false, d) / p(correct=true, d) for every bins -> samples in the bins are assigned the corresponding acc_ratio
        6. Compute the calibration score for each sample
        
        Params:
            probs: the confidenc vector of every classes (shape [samples, classes])
            preds: the predicted class for each sample (shape [samples, ])
            true: true labels (shape [samples,])
            proximity: the exponential function of the negative average distance to K nearest neighbors (shape [samples,])
            
        Returns:
            None
        """
        assert np.all(probs >= 0) and np.all(probs <= 1), "All elements in 'probs' should be in the range [0, 1]."
        
        confs = np.max(probs, axis=-1)
            
        val_df = pd.DataFrame({'ys':true, 'proximity':proximity, 'conf':confs, 'pred':preds})
        
        val_df['correct'] = (val_df.pred == val_df.ys).astype('int')
        
        val_df_true = val_df[val_df['correct'] == 1]
        val_df_false = val_df[val_df['correct'] == 0]
        

        indep = val_df_true['proximity']
        dep = val_df_true['conf']
        self.dens_true = sm.nonparametric.KDEMultivariate(data=[dep, indep], var_type='cc', bw=bandwidth)

        indep = val_df_false['proximity']
        dep = val_df_false['conf']
        self.dens_false = sm.nonparametric.KDEMultivariate(data=[dep, indep], var_type='cc', bw=bandwidth)
        
        self.false_true_ratio = (val_df.pred != val_df.ys).sum() / (val_df.pred == val_df.ys).sum()
        
        print('Density Estimation Done.')      
  

    def predict(self, probs, proximities):
        """
        use Bayes' rule to compute the posterior probability
        
        p(\hat{y}=y | h(x), d)=\frac{p(h(x), d|\hat{y}=y)} {p(h(x), d|\hat{y}=y) + p(h(x), d|\hat{y} \neq y) \cdot \frac{p(\hat{y} \neq y)}{p(\hat{y}=y)}}
        
        Params:
            probs: the confidenc vector of every classes (shape [samples, classes])
            preds: the predicted class for each sample (shape [samples, ])
            true: true labels (shape [samples,])
            proximity:  the exponential function of the negative average distance to K nearest neighbors (shape [samples,])
            
        Returns:
            calibrated probabilities (nd.array with shape [samples,])
        """
 
        assert np.all(probs >= 0) and np.all(probs <= 1), "All elements in 'probs' should be in the range [0, 1]."
        
        preds = np.argmax(probs, axis=-1)
        confs = np.max(probs, axis=-1) 
            
        data = np.array([confs, proximities]).T # shape [samples, 2]
        conf_reg_true = self.dens_true.pdf(data) # shape [samples,]
        conf_reg_false = self.dens_false.pdf(data)
        
        # eps is to avoid division by 0
        eps = 1e-10
        conf_calibrated = conf_reg_true / np.maximum(conf_reg_true + conf_reg_false * self.false_true_ratio, eps)
    
        
        # Normalize the rest of the values in each row to sum to 1-conf_max
        mask = np.ones(probs.shape, dtype=bool)
        mask[range(probs.shape[0]), preds] = False
        probs = probs * mask
        probs = probs * ((1 - conf_calibrated) / probs.sum(axis=-1))[:, np.newaxis]
        
        # Add the calibrated confidence to the predicted class
        probs[range(probs.shape[0]), preds] = conf_calibrated # dtype64 -> dtype32

        return probs
    
    

def mirror_1d(d, xmin=None, xmax=None):
    """
    If necessary apply reflecting boundary conditions.
    input data d: [samples,]
    """
    if xmin is not None and xmax is not None:
        xmed = (xmin+xmax)/2
        return np.concatenate(((2*xmin-d[d < xmed]), d, (2*xmax-d[d >= xmed])))
    elif xmin is not None:
        return np.concatenate((2*xmin-d, d))
    elif xmax is not None:
        return np.concatenate((d, 2*xmax-d))
    else:
        return d
    
def mirror_1d_along_axis(data, axis=0, xmin=0, xmax=1):
    """
    data: [samples, 2] (in our case dim_features=2)
    data1 is the confidence; [samples,]
    data2 is the proximity; [samples,]
    """
    # 1. mirror data1
    if xmin is not None and xmax is not None:
        xmed = (xmin + xmax)/2
        d_left = np.copy(data[data[:, axis] < xmed])
        d_right = np.copy(data[data[:, axis] >= xmed])
        d_left[:, axis] = 2*xmin - d_left[:, axis]
        d_right[:, axis] = 2*xmax - d_right[:, axis]
        data_mirror = np.concatenate((d_left, data, d_right), axis=0)
        return data_mirror
    elif xmin is not None:
        d_left = np.copy(data)
        d_left[:, axis] = 2*xmin - d_left[:, axis]
        return np.concatenate((d_left, data), axis=0)
    elif xmax is not None:
        d_right = np.copy(data)
        d_right[:, axis] = 2*xmax - d_right[:, axis]
        return np.concatenate((data, d_right), axis=0)
    else:
        return data

def mirror_2d(data, xmin=0, xmax=1, ymin=0, ymax=None):
    """ 
    first axis: conf; second axis: proximity
    data: [samples, dim_features] (in our case dim_features=2)
    """
    data_mirror1 = mirror_1d_along_axis(data, axis=0, xmin=xmin, xmax=xmax)
    data_mirror2 = mirror_1d_along_axis(data_mirror1, axis=1, xmin=ymin, xmax=ymax)
    return data_mirror2
    
    
    
class CustomizedDensityRatioCalibration():
    
    """customized version """
    
    def __init__(self, kernel, kernel_func, mirror=False, bandwidth=0.1, norm=2):
        self.kernel = kernel
        self.kernel_func = kernel_func
        self.bandwidth = bandwidth
        self.norm = norm
        self.mirror = mirror
        
        
    # Find the temperature
    def fit(self, logits, preds, true, proximity, is_conf=True):
        """
        Train the density estimator for correctly classified samples and misclassified samples
        1. split samples into correctly classified and misclassified
        3. count their numbers and show the ratio   
        4. learn the conditional distribution of the confidence given the proximity -> the distributions are represented as <dens_true, dens_false>
        5. Compute acc_ratio = p(correct=false, d) / p(correct=true, d) for every bins -> samples in the bins are assigned the corresponding acc_ratio
        6. Compute the calibration score for each sample
        
        Params:
            if is_conf == true:
                logits: the output from neural network for each class (shape [samples, classes])
            else:
                logits: confidence scores (shape [samples,])
            
            preds: the predicted class for each sample (shape [samples, ])
            true: true labels (shape [samples,])
            proximity:  the exponential function of the negative average distance to K nearest neighbors (shape [samples,])
            
        Returns:
            None
        """
        # if is_conf == true, think the logits are actually confidences; otherwise compute confidence scores
        if is_conf:
            confs = logits
        else:
            confs = np.max(softmax(logits, axis=-1), axis=-1) # TODO
            
        val_df = pd.DataFrame({'ys':true, 'proximity':proximity, 'conf':confs, 'pred':preds})
        
        val_df['correct'] = (val_df.pred == val_df.ys).astype('int')
        
        
        val_df_true = val_df[val_df['correct'] == 1]
        val_df_false = val_df[val_df['correct'] == 0]
        
        # mirror the data to avoid boundary effects
        low_bound = 0.0
        up_bound = 1.0 
            
        true_proximity = val_df_true['proximity'].to_numpy()
        true_conf = val_df_true['conf'].to_numpy()
        true_data = np.array([true_conf, true_proximity]).T
        
        
        false_proximity = val_df_false['proximity'].to_numpy()
        false_conf = val_df_false['conf'].to_numpy()
        false_data = np.array([false_conf, false_proximity]).T # (n_samples, n_features)
        
        if self.mirror:
            true_data = mirror_2d(true_data, xmin=low_bound, xmax=up_bound, ymin=0.0, ymax=None)
            false_data = mirror_2d(false_data, xmin=low_bound, xmax=up_bound, ymin=0.0, ymax=None)
        
        
        if self.kernel == 'scipy_gaussian_kde':
            # data shape: shape (# of dims, # of data).
            self.dens_true = gaussian_kde(true_data.T, bw_method=self.bandwidth)
            self.dens_false = gaussian_kde(false_data.T, bw_method=self.bandwidth)
        
        elif self.kernel == 'sklearn_kde':
            # (n_samples, n_features)
            self.dens_true = KernelDensity(kernel=self.kernel_func, bandwidth=self.bandwidth).fit(true_data)
            self.dens_false = KernelDensity(kernel=self.kernel_func, bandwidth=self.bandwidth).fit(false_data)
            
        
        # elif self.kernel == 'NaiveKDE':
        #     # true_data shape: (num_samples, 2)
        #     self.dens_true = kde.NaiveKDE(kernel=self.kernel_func, bw=self.bandwidth, norm=self.norm).fit(true_data)
        #     self.dens_false = kde.NaiveKDE(kernel=self.kernel_func, bw=self.bandwidth, norm=self.norm).fit(false_data)
            
        # elif self.kernel == 'FFTKDE':
        #     # true_data shape: (num_samples, 2)
        #     self.dens_true = kde.FFTKDE(kernel=self.kernel_func, bw=self.bandwidth, norm=self.norm).fit(true_data)
        #     self.dens_false = kde.FFTKDE(kernel=self.kernel_func, bw=self.bandwidth, norm=self.norm).fit(false_data)
            
        elif self.kernel == 'KDEMultivariate':
            self.dens_true = sm.nonparametric.KDEMultivariate(data=true_data, var_type='cc', bw=self.bandwidth)
            self.dens_false = sm.nonparametric.KDEMultivariate(data=false_data, var_type='cc', bw=self.bandwidth)
        else:
            raise NotImplementedError
        
        self.false_true_ratio = (val_df.pred != val_df.ys).sum() / (val_df.pred == val_df.ys).sum()
        
        self.get_bw()
        
        print('Density Estimation Done.')   
    
    def dens_true_pdf(self, logits, proximities, is_conf=True):
        """get the pdf for correctly classified samples """
        if is_conf:
            confs = logits
        else:
            confs = np.max(softmax(logits, axis=-1), axis=-1)
            
        data = np.array([confs, proximities]).T
        
        if self.kernel in ['NaiveKDE', 'TreeKDE', 'FFTKDE']:
            # Sort the data in one of the dimensions
            sorted_indices = np.argsort(data[:, 0])
            sorted_data = data[sorted_indices, :]
            
            conf_reg_true = self.dens_true.evaluate(sorted_data)
            conf_reg_false = self.dens_false.evaluate(sorted_data)
            
            # Use the original indices to sort the density values back into the original order
            conf_reg_true = conf_reg_true[np.argsort(sorted_indices), :]
            conf_reg_false = conf_reg_false[np.argsort(sorted_indices), :]

        elif self.kernel == 'KDEMultivariate':
            conf_reg_true = self.dens_true.pdf(data)
            # conf_reg_false = self.dens_false.pdf(data)
        
        elif self.kernel == 'scipy_gaussian_kde':
            conf_reg_true = self.dens_true.pdf(data.T)
            # conf_reg_false = self.dens_false.pdf(data.T)
        
        elif self.kernel == 'sklearn_kde':
            conf_reg_true = np.exp(self.dens_true.score_samples(data))
            # conf_reg_false = self.dens_false.score_samples(data)
        else:
            raise NotImplementedError
        
        if self.mirror:
            conf_reg_true[confs<0.0] = 0  # Set the KDE to zero outside of the domain
            conf_reg_true[confs>1.0] = 0
            # conf_reg_false[confs<0.0] = 0
            # conf_reg_false[confs>1.0] = 0
        
            # # Double the density to get integral of ~1 -> two dimension 4
            conf_reg_true = conf_reg_true * 4 
            # conf_reg_false = conf_reg_false * 4
        
        return conf_reg_true
        
    def dens_false_pdf(self, logits, proximities, is_conf=True):
        """get the pdf for correctly classified samples """
        if is_conf:
            confs = logits
        else:
            confs = np.max(softmax(logits, axis=-1), axis=-1)
            
        data = np.array([confs, proximities]).T
        
        if self.kernel in ['NaiveKDE', 'TreeKDE', 'FFTKDE']:
            # Sort the data in one of the dimensions
            sorted_indices = np.argsort(data[:, 0])
            sorted_data = data[sorted_indices, :]
            
            conf_reg_true = self.dens_true.evaluate(sorted_data)
            conf_reg_false = self.dens_false.evaluate(sorted_data)
            
            # Use the original indices to sort the density values back into the original order
            conf_reg_true = conf_reg_true[np.argsort(sorted_indices), :]
            conf_reg_false = conf_reg_false[np.argsort(sorted_indices), :]

        elif self.kernel == 'KDEMultivariate':
            # conf_reg_true = self.dens_true.pdf(data)
            conf_reg_false = self.dens_false.pdf(data)
        
        elif self.kernel == 'scipy_gaussian_kde':
            # conf_reg_true = self.dens_true.pdf(data.T)
            conf_reg_false = self.dens_false.pdf(data.T)
        
        elif self.kernel == 'sklearn_kde':
            # conf_reg_true = self.dens_true.score_samples(data)
            conf_reg_false = np.exp(self.dens_false.score_samples(data))
        else:
            raise NotImplementedError
        
        if self.mirror:
            # conf_reg_true[confs<0.0] = 0  # Set the KDE to zero outside of the domain
            # conf_reg_true[confs>1.0] = 0
            conf_reg_false[confs<0.0] = 0
            conf_reg_false[confs>1.0] = 0
        
            # # Double the density to get integral of ~1 -> two dimension 4
            # conf_reg_true = conf_reg_true * 4 
            conf_reg_false = conf_reg_false * 4
        
        return conf_reg_false
    
    def get_bw(self):
  
        if self.kernel in ['KDEMultivariate', 'FFTKDE', 'NaiveKDE']:
            self.dens_true_bw = self.dens_true.bw
            self.dens_false_bw = self.dens_false.bw
        elif self.kernel == 'scipy_gaussian_kde':
            self.dens_true_bw = self.dens_true.factor
            self.dens_false_bw = self.dens_false.factor
        elif self.kernel == 'sklearn_kde':
            self.dens_true_bw = self.dens_true.bandwidth
            self.dens_false_bw = self.dens_false.bandwidth  
        
    def predict(self, logits, proximities, is_conf=True):
        """
        use Bayes' rule to compute the posterior probability
        
        p(\hat{y}=y | h(x), d)=\frac{p(h(x), d|\hat{y}=y)} {p(h(x), d|\hat{y}=y) + p(h(x), d|\hat{y} \neq y) \cdot \frac{p(\hat{y} \neq y)}{p(\hat{y}=y)}}
        
        Params:
            if is_conf == true:
                logits: the output from neural network for each class (shape [samples, classes])
            else:
                logits: confidence scores (shape [samples,])
            
            preds: the predicted class for each sample (shape [samples, ])
            true: true labels (shape [samples,])
            proximity:  the exponential function of the negative average distance to K nearest neighbors (shape [samples,])
            
        Returns:
            calibrated probabilities (nd.array with shape [samples,])
        """
        # if is_conf == true, think the logits are actually confidences; otherwise compute confidence scores
        if is_conf:
            confs = logits
        else:
            confs = np.max(softmax(logits, axis=-1), axis=-1)
            
        data = np.array([confs, proximities]).T
        
        if self.kernel in ['NaiveKDE', 'TreeKDE', 'FFTKDE']:
            # Sort the data in one of the dimensions
            sorted_indices = np.argsort(data[:, 0])
            sorted_data = data[sorted_indices, :]
            
            conf_reg_true = self.dens_true.evaluate(sorted_data)
            conf_reg_false = self.dens_false.evaluate(sorted_data)
            
            # Use the original indices to sort the density values back into the original order
            conf_reg_true = conf_reg_true[np.argsort(sorted_indices), :]
            conf_reg_false = conf_reg_false[np.argsort(sorted_indices), :]

        elif self.kernel == 'KDEMultivariate':
            conf_reg_true = self.dens_true.pdf(data)
            conf_reg_false = self.dens_false.pdf(data)
        
        elif self.kernel == 'scipy_gaussian_kde':
            conf_reg_true = self.dens_true.pdf(data.T)
            conf_reg_false = self.dens_false.pdf(data.T)
        
        elif self.kernel == 'sklearn_kde':
            conf_reg_true = np.exp(self.dens_true.score_samples(data))
            conf_reg_false = np.exp(self.dens_false.score_samples(data))
        else:
            raise NotImplementedError
        
        if self.mirror:
            conf_reg_true[confs<0.0] = 0  # Set the KDE to zero outside of the domain
            conf_reg_true[confs>1.0] = 0
            conf_reg_false[confs<0.0] = 0
            conf_reg_false[confs>1.0] = 0
        
            # # Double the density to get integral of ~1 -> two dimension 4
            conf_reg_true = conf_reg_true * 4 
            conf_reg_false = conf_reg_false * 4
        
    
        # eps is to avoid division by 0
        eps = 1e-10
        conf_calibrated = conf_reg_true / np.maximum(conf_reg_true + conf_reg_false * self.false_true_ratio, eps)
        
        probs = softmax(logits, axis=-1)
        preds = np.argmax(probs, axis=-1)
        # Normalize the rest of the values in each row to sum to 1-conf_max
        mask = np.ones(probs.shape, dtype=bool)
        mask[range(probs.shape[0]), preds] = False
        probs = probs * mask
        probs = probs * ((1 - conf_calibrated) / probs.sum(axis=-1))[:, np.newaxis]
        
        # Add the calibrated confidence to the predicted class
        probs[range(probs.shape[0]), preds] = conf_calibrated # dtype64 -> dtype32
        
        return probs
    
