"""
This code is used to evaluate the performance of several calibration methods on every model.

The calibration measure includes:
- ECE (Expected Calibration Error)
- AdaptiveECE (Adaptive Expected Calibration Error)
- DistAwareECE (Distance-Aware Expected Calibration Error)

Calibration methods include:
- Temperature Scaling
- Parameterized Temperature Scaling
- Density Estimation Calibration
- Histogram Binning
"""

#%%
import torch
import pandas as pd
import seaborn as sns
import faiss
from argparse import ArgumentParser
from sklearn.preprocessing import KBinsDiscretizer
from scipy.special import softmax
import argparse
import os, time, pdb, sys
import os.path as osp
import json
import shutil
import random
import numpy as np
from sklearn.metrics import classification_report
import matplotlib 
import pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from temperature_scaling import *
from netcal.presentation import ReliabilityDiagram
from netcal.metrics import ECE
from netcal.scaling import TemperatureScaling
from netcal.binning import HistogramBinning

from utils.visualize_helper import density_map_plot

def check_manual_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed) # for cpu
    torch.cuda.manual_seed(seed) # for single GPU
    torch.cuda.manual_seed_all(seed) # for all GPUs
    print("Using seed: {seed}".format(seed=seed))
    

#%%
parser = ArgumentParser()

# parser.add_argument("--data_dir", type=str, default="cifar10_data")
# parser.add_argument("--download_weights", type=int, default=0, choices=[0, 1])
# parser.add_argument("--test_phase", type=int, default=0, choices=[0, 1])
# parser.add_argument("--dev", type=int, default=0, choices=[0, 1])
# parser.add_argument( "--logger", type=str, default="tensorboard", choices=["tensorboard", "wandb"] )

# parser.add_argument("--classifier", type=str, default="resnet18")
# parser.add_argument("--pretrained", type=int, default=0, choices=[0, 1])

# parser.add_argument("--precision", type=int, default=32, choices=[16, 32])
# parser.add_argument("--batch_size", type=int, default=256)
# parser.add_argument("--max_epochs", type=int, default=100)
# parser.add_argument("--num_workers", type=int, default=8)
# parser.add_argument("--gpu_id", type=str, default="3")

# parser.add_argument("--learning_rate", type=float, default=1e-2)
# parser.add_argument("--pin_mem", type=bool, default=False)
# parser.add_argument("--weight_decay", type=float, default=1e-2)

parser.add_argument("--normalize", type=bool, default=True)

parser.add_argument("--num_neighbors", type=int, default=10)
parser.add_argument("--random_seed", type=int, default=2022)
parser.add_argument("--distance_measure", type=str, default="L2") # L2, cosine, IVFFlat, IVFPQ

parser.add_argument('--model', 
                    # default='eca_nfnet_l1',
                    # default='convnext_xlarge_384_in22ft1k',
                    # default='tf_efficientnet_b4_ns',
                    # default='efficientnet_b3',
                    # default='tf_efficientnet_b3',
                    default='resnet18',
                    # default='mobilenetv3_small_050',
                    # default='swsl_resnext101_32x8d',
                    # default='beit_large_patch16_384',
                    # default='mobilevitv2_075',
                    # default='repvgg_b0',
                    # default='volo_d5_512'
)


args = parser.parse_args()

check_manual_seed(args.random_seed)

os.environ['CUDA_VISIBLE_DEVICES'] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
############## LOAD MODEL #######################
K = args.num_neighbors
print("Loading model: {}".format(args.model))
# save data
load_dir = "pytorch_image_models/intermediate_output/imagenet/"
ys, zs, logits, confs, preds = pickle.load(open(osp.join(load_dir, 'out_{}.p'.format(args.model)), 'rb'))
val_acc = (ys == preds).mean()
print('Val acc: {:.4f}, mean conf: {:.4f}'.format(val_acc, confs.mean()))
max_logits = np.max(logits, axis=1)
confs_vec = softmax(logits, axis=-1)
if args.normalize:
    zs = zs / np.linalg.norm(zs, axis=1, keepdims=True)


img_dir = "plots/{:d}_{}".format(int((ys == preds).mean()*1000), args.model)
os.makedirs(img_dir, exist_ok=True)


# split dataset into two parts
try:
    permute_idx = np.load(osp.join(img_dir, "val_test_idx.npy"))
except:
    permute_idx = np.random.permutation(ys.shape[0])    
    np.save(osp.join(img_dir, "val_test_idx.npy"), permute_idx)
val_idx = permute_idx[0:int(ys.shape[0]/2)]
test_idx = permute_idx[int(ys.shape[0]/2):]
val_ys, val_zs, val_logits, val_preds, val_confs = ys[val_idx], zs[val_idx], logits[val_idx], preds[val_idx], confs[val_idx]
test_ys, test_zs, test_logits, test_preds, test_confs = ys[test_idx], zs[test_idx], logits[test_idx], preds[test_idx], confs[test_idx]


# initialize a KDTree / or other search engine
dim = val_zs.shape[1]
if args.distance_measure == "L2":
    index = faiss.IndexFlatL2(dim) # val_zs.shape[1]: len_feature
elif args.distance_measure == "cosine":
    index = faiss.IndexFlatIP(dim)
elif args.distance_measure == "IVFFlat":
    nlist = 100
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    index.nprobe = 10 # number of clusters; default to be 1; if nprobe=nlist -> exact search 
    index.train(val_zs) # need training 
elif args.distance_measure == "IVFPQ":
    nlist = 100  # number of clusters
    m = 8        # compressed into 8 bit
    quantizer = faiss.IndexFlatL2(dim) # define the quantizer 
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)  # 8 specifies that each sub-vector is encoded as 8 bits
    index.nprobe = 10 # number of clusters; default to be 1; if nprobe=nlist -> exact search 
    index.train(val_zs) # need training 
else:
    raise NotImplementedError

# add data to the KDTree
index.add(val_zs)
# search neigh_dists, neigh_indices
D, I = index.search(val_zs, K+1)
val_dists = D[:, 1:]
test_dists, I = index.search(test_zs, K)


#################### CALIBRATION METHOD ############################

val_knndists = np.mean(val_dists, axis=1)
test_knndists = np.mean(test_dists, axis=1)
# val_confs = np.max(softmax(val_logits, axis=-1), axis=-1)
# val_preds = np.argmax(val_logits, axis=-1)

compare_methods = ['custom_density_estimation']
# compare_methods = ['conf', 'pts', 'pts_conf', 'pts_knndist']
test_results = {}
if 'conf' in compare_methods:
    test_results['conf'] =  test_confs
    np.save(osp.join(img_dir, "test_conf.npy"), test_confs)
    

if 'temperature_scaling' in compare_methods:

    from utils.temperature_scaling import  TemperatureScaling

    TS_calibrator = TemperatureScaling(maxiter=100)
    TS_calibrator.fit(val_logits, val_ys)
    best_temp = TS_calibrator.temp 

    probs_val = TS_calibrator.predict(val_logits) 
    probs_test = TS_calibrator.predict(test_logits)
    confs_ts_val = np.max(probs_val,axis=-1)
    confs_ts_test = np.max(probs_test,axis=-1)
    
    test_results['temperature_scaling'] =  confs_ts_test
    np.save(osp.join(img_dir, "test_temperature_scaling.npy"), confs_ts_test)
    
if 'histogram_binning' in compare_methods:
    from netcal.binning import HistogramBinning
    """one-verus-all histogram binning: for every class, learn a binary calirator; for every class, learn 10 bins' accuracies"""

    histbin = HistogramBinning(bins=10)
    probs_val = histbin.fit_transform(softmax(val_logits, axis=-1), val_ys)
    probs_test = histbin.transform(softmax(test_logits, axis=-1))
    
    calib_confs_val = np.max(probs_val,axis=-1)
    confs_hist_test = np.max(probs_test,axis=-1)
    
    test_results['histogram_binning'] =  confs_hist_test
    np.save(osp.join(img_dir, "test_histogram_binning.npy"), confs_hist_test)

if 'isotonic_regression' in compare_methods:
    from netcal.binning import IsotonicRegression
    
    calibrator = IsotonicRegression() # input (softmax) shape (n_samples, [n_classes])
    probs_val = calibrator.fit_transform(softmax(val_logits, axis=-1), val_ys)
    probs_test = calibrator.transform(softmax(test_logits, axis=-1))
    
    calib_confs_val = np.max(probs_val,axis=-1)
    calib_confs_test = np.max(probs_test,axis=-1)
    
    test_results['isotonic_regression'] =  calib_confs_test
    np.save(osp.join(img_dir, "test_isotonic_regression.npy"), calib_confs_test)   

if 'custom_density_estimation' in compare_methods:
    from utils.density_ratio_calibration import CustomizeKDECal
    
    kernel = 'sklearn_kde'
    kernel_func = 'gaussian'
    bandwidth = 0.04
    mirror = False
    calibrator = CustomizeKDECal(kernel=kernel, kernel_func=kernel_func, bandwidth=bandwidth, mirror=mirror) 
    
    calibrator.fit(val_confs, val_preds, val_ys, val_knndists, is_conf=True)
    conf_reg_val = calibrator.predict(val_confs, val_knndists, is_conf=True)
    conf_reg_test = calibrator.predict(test_logits, test_knndists, is_conf=False)
    
    test_results['custom_density_estimation'] = conf_reg_test
    np.save(osp.join(img_dir, "test_custom_density_estimation_{}.npy".format(args.distance_measure)), conf_reg_test)

if 'density_estimation' in compare_methods:
    from utils.density_ratio_calibration import DensityRatioCalibration

    DER_calibrator = DensityRatioCalibration()
    DER_calibrator.fit(val_confs, val_preds, val_ys, val_knndists, is_conf=True)
    conf_reg_val = DER_calibrator.predict(val_confs, val_knndists, is_conf=True)
    conf_reg_test = DER_calibrator.predict(test_confs, test_knndists, is_conf=True)
    
    test_results['density_estimation'] = conf_reg_test
    np.save(osp.join(img_dir, "test_density_estimation_{}.npy".format(args.distance_measure)), conf_reg_test)

if 'pts' in compare_methods:

    conf_save_file = osp.join(img_dir, "test_pts.npy")
    if osp.exists(conf_save_file):
        test_results['pts'] = np.load(conf_save_file)
    else:
        from utils.parameterized_temp_scaling import ParameterizedTemperatureScaling

        PTS_calibrator = ParameterizedTemperatureScaling(
                epochs=100, # stepsize = 100,000
                lr=0.00005,
                batch_size=1000,
                nlayers=2,
                n_nodes=5,
                length_logits=1000,
                top_k_logits=10
        )
        PTS_calibrator.tune(val_logits, val_ys)
        conf_pts_val = PTS_calibrator.calibrate(val_logits).max(axis=-1)
        conf_pts_test = PTS_calibrator.calibrate(test_logits).max(axis=-1)
        
        test_results['pts'] =  conf_pts_test
        np.save(osp.join(img_dir, "test_pts.npy"), conf_pts_test)
    
if 'pts_conf' in compare_methods:

    from utils.parameterized_temp_scaling import ParameterizedTemperatureScaling

    PTS_conf_calibrator = ParameterizedTemperatureScaling(
            epochs=100, # stepsize = 100,000
            lr=0.00005,
            batch_size=1000,
            nlayers=2,
            n_nodes=5,
            length_logits=1000,
            top_k_logits=10
    )
    val_probs = softmax(val_logits, axis=-1)
    test_probs = softmax(test_logits, axis=-1)
    PTS_conf_calibrator.tune(val_probs, val_ys)
    conf_pts_conf_val = PTS_conf_calibrator.calibrate(val_probs).max(axis=-1)
    conf_pts_conf_test = PTS_conf_calibrator.calibrate(test_probs).max(axis=-1)
    
    test_results['pts_conf'] =  conf_pts_conf_test 
    np.save(osp.join(img_dir, "test_pts_conf.npy"), conf_pts_conf_test)

if 'pts_knndist' in compare_methods:
    
    from utils.parameterized_temp_scaling import ParameterizedNeighborTemperatureScaling

    PTS_knndist_calibrator = ParameterizedNeighborTemperatureScaling(
            epochs=100, # stepsize = 100,000
            lr=0.00005,
            batch_size=1000,
            nlayers=2,
            n_nodes=5,
            length_logits=1000,
            top_k_logits=10,
            top_k_neighbors=args.num_neighbors
    )
    PTS_knndist_calibrator.tune(val_logits, val_dists, val_ys)
    conf_pts_knndist_val = PTS_knndist_calibrator.calibrate(val_logits, val_dists).max(axis=-1)
    conf_pts_knndist_test = PTS_knndist_calibrator.calibrate(test_logits, test_dists).max(axis=-1)
    
    test_results['pts_knndist'] =  conf_pts_knndist_test  
    np.save(osp.join(img_dir, "test_pts_knndist.npy"), conf_pts_knndist_test) 

if 'ensemble_ts' in compare_methods:
    from utils.ensemble_temperature_scaling import EnsembleTemperatureScaling
    calibrator = EnsembleTemperatureScaling() # input (softmax) shape (n_samples, [n_classes])
    probs_val = calibrator.fit_transform(val_logits, val_ys)
    probs_test = calibrator.transform(test_logits)
    
    calib_confs_val = np.max(probs_val,axis=-1)
    calib_confs_test = np.max(probs_test,axis=-1)
    
    test_results['ensemble_ts'] =  calib_confs_test
    np.save(osp.join(img_dir, "test_ensemble_ts.npy"), calib_confs_test)  
    
if 'multi_isotonic_regression' in compare_methods:
    from utils.ensemble_temperature_scaling import MultiIsotonicRegression

    calibrator = MultiIsotonicRegression() # input (softmax) shape (n_samples, [n_classes])
    probs_val = calibrator.fit_transform(val_logits, val_ys)
    probs_test = calibrator.transform(test_logits)
    
    calib_confs_val = np.max(probs_val,axis=-1)
    calib_confs_test = np.max(probs_test,axis=-1)
    
    test_results['multi_isotonic_regression'] =  calib_confs_test
    np.save(osp.join(img_dir, "test_multi_isotonic_regression.npy"), calib_confs_test)
# %%
#################### TESTING ECE LOSS ############################
"""
The header is:
compare_methods = ['conf', 'temperature_scaling', 'density_estimation', 'pts', 'histogram_binning', 'isotonic_regression']:
model,val_acc,conf_ece,conf_mce,conf_ace,conf_top_ece,conf_da_ece,conf_da_mce,temp_ece,temp_mce,temp_ace,temp_top_ece,temp_da_ece,temp_da_mce,conf_reg_ece,conf_reg_mce,conf_reg_ace,conf_reg_top_ece,conf_reg_da_ece,conf_reg_da_mce,conf_pts_ece,conf_pts_mce,conf_pts_ace,conf_pts_top_ece,conf_pts_da_ece,conf_pts_da_mce,conf_hist_ece,conf_hist_mce,conf_hist_ace,conf_hist_top_ece,conf_hist_da_ece,conf_hist_da_mce,conf_isotonic_ece,conf_isotonic_mce,conf_isotonic_ace,conf_isotonic_top_ece,conf_isotonic_da_ece,conf_isotonic_da_mce,


res_dir/metrics_append_knnbin10_confbin15.csv
compare_methods = ['conf', 'temperature_scaling', 'density_estimation', 'pts', 'histogram_binning', 'isotonic_regression', 'ensemble_ts', 'multi_isotonic_regression']
model,val_acc,conf_ece,conf_mce,conf_ace,conf_top_ece,conf_da_ece,conf_da_mce,temp_ece,temp_mce,temp_ace,temp_top_ece,temp_da_ece,temp_da_mce,conf_reg_ece,conf_reg_mce,conf_reg_ace,conf_reg_top_ece,conf_reg_da_ece,conf_reg_da_mce,conf_pts_ece,conf_pts_mce,conf_pts_ace,conf_pts_top_ece,conf_pts_da_ece,conf_pts_da_mce,conf_hist_ece,conf_hist_mce,conf_hist_ace,conf_hist_top_ece,conf_hist_da_ece,conf_hist_da_mce,conf_isotonic_ece,conf_isotonic_mce,conf_isotonic_ace,conf_isotonic_top_ece,conf_isotonic_da_ece,conf_isotonic_da_mce,conf_ets_ece,conf_ets_mce,conf_ets_ace,conf_ets_top_ece,conf_ets_da_ece,conf_ets_da_mce,conf_multi_isotonic_ece,conf_multi_isotonic_mce,conf_multi_isotonic_ace,conf_multi_isotonic_top_ece,conf_multi_isotonic_da_ece,conf_multi_isotonic_da_mce,

compare_methods = ['pts']
model,val_acc,conf_reg_ece,conf_reg_mce,conf_reg_ace,conf_reg_top_ece,conf_reg_da_ece,conf_reg_da_mce
"""
# TODO test how many knnbins are needed and suitable
knnbin = 10
confbin = 15
file_name_to_save = "res_dir/metrics_custom_kde_{}_knnbin{}_confbin{}.csv".format(args.distance_measure, knnbin, confbin)


# with open(file_name_to_save, "a") as f:
#     f.write(args.model + ',' + str(val_acc))
            
from utils.metrics import evaluate_with_da_preds

for method in compare_methods:
    test_conf_score = test_results[method]

    ece, mce, ace, toplabel_ece, da_ece, da_mce = evaluate_with_da_preds(test_conf_score, test_preds, test_ys, test_knndists, verbose = False, normalize = False, conf_bins = confbin, knn_bins=knnbin)

    with open(file_name_to_save, "a") as f:
        write_list = [args.model, val_acc, "kde", kernel, kernel_func, bandwidth, ece, mce, ace, toplabel_ece, da_ece, da_mce, mirror]
        f.write(','.join(str(x).replace(",", "") for x in write_list) + ',' )
        f.write("\n")



################## TEST PROXIMITY BIAS MITIGATION VISUAlIZATION ############################
# uncomment to save the proximity bias plot
# test_df = pd.DataFrame({'ys':test_ys, 'knndist':test_knndists, 'pred':test_preds})

# test_df['correct'] = (test_df.pred == test_df.ys).astype('int')

# test_df['knn_bin'] = KBinsDiscretizer(n_bins=6, encode='ordinal').fit_transform(test_knndists.reshape(-1, 1))
# # draw the plot on test dataset
# group_correct = test_df.groupby('knn_bin')['correct'].mean()
# group_knn = test_df.groupby('knn_bin')['knndist'].mean()


# plt.figure()
# colors = plt.cm.BuPu(np.linspace(0, 0.5, 2))
# plt.plot(group_knn, group_correct, 'bx-', label='acc')
# # plt.plot(group_knn, group_confs, 'go-', label='conf')

# # compare_methods = ['conf', 'temperature_scaling', 'density_estimation', 'pts']
# # label_names = ['conf', 'conf_temp', 'ours', 'conf_pts']
# label_names = ['conf', 'pts', 'pts_conf', 'pts_knndist']
# markers = ['go-', 'y+-', 'r^-', 'c+-']
# for method, marker, label in zip(compare_methods, markers, label_names):
#     test_df[method] = test_results[method]
#     group_confs_reg = test_df.groupby('knn_bin')[method].mean()
#     plt.plot(group_knn, group_confs_reg, marker, label=label)

# plt.title("{}".format(args.model))
# plt.legend()
# plt.grid(True)
# plt.xlabel("avg-to-knn distance")
# plt.show()
# plt.savefig(osp.join(img_dir, "pts_proximity_bias_{}_K{}".format(args.model, K)), dpi=300)

