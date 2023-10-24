"""
Dataset: Inaturalist 2021
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
import pandas as pd
# import torch
import faiss
from argparse import ArgumentParser
from sklearn.preprocessing import KBinsDiscretizer
from scipy.special import softmax
import os, pdb
import os.path as osp
import random
import numpy as np
import matplotlib 
import pickle
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from temperature_scaling import *
from netcal.scaling import TemperatureScaling
from netcal.binning import HistogramBinning


def check_manual_seed(seed):
    random.seed(seed)
    # torch.manual_seed(seed) # for cpu
    # torch.cuda.manual_seed(seed) # for single GPU
    # torch.cuda.manual_seed_all(seed) # for all GPUs
    print("Using seed: {seed}".format(seed=seed))
    

#%%
parser = ArgumentParser()

parser.add_argument("--data_dir", type=str, default="/home/miao/repo2022/yeeef/newt/benchmark/output")
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
                    default='resnet50_from_scratch_inat2021_val',
                    # default='mobilenetv3_small_050',
                    # default='swsl_resnext101_32x8d',
                    # default='beit_large_patch16_384',
                    # default='mobilevitv2_075',
                    # default='repvgg_b0',
                    # default='volo_d5_512'
)


args = parser.parse_args()

check_manual_seed(args.random_seed)

# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%
############## LOAD MODEL #######################
K = args.num_neighbors
print("Loading model: {}".format(args.model))
# save data
# ys: [num_samples,]; logits: [num_samples, num_classes] ;zs: [num_samples, dim_features] (last_second_feature); confs: [num_samples, ]; preds: [num_samples,]
# ys: ground truth label; same shape with preds
# confs = np.max(logits, axis=-1)
# preds = np.argmax(logits, axis=-1)
ys, zs, logits, confs, preds = pickle.load(open(osp.join(args.data_dir, 'out_{}.p'.format(args.model)), 'rb'))
# TODO
if 'svhn' in args.model:
    confs = np.max(confs, axis=-1)
    
num_classes = logits.shape[1]
val_acc = (ys == preds).mean()
print('Val acc: {:.4f}, mean conf: {:.4f}'.format(val_acc, confs.mean()))

if args.normalize:
    zs = zs / np.linalg.norm(zs, axis=1, keepdims=True)
print("zs.shape: {}".format(zs.shape))

img_dir = "plots/{:d}_{}".format(int((ys == preds).mean()*1000), args.model)
os.makedirs(img_dir, exist_ok=True)

# split dataset into two parts
try:
    permute_idx = np.load(osp.join(img_dir, "val_test_idx_{}.npy".format(args.random_seed)))
except:
    permute_idx = np.random.permutation(ys.shape[0])    
    np.save(osp.join(img_dir, "val_test_idx_{}.npy".format(args.random_seed)), permute_idx)

val_idx = permute_idx[0:int(ys.shape[0]/4)]
test_idx = permute_idx[int(ys.shape[0]/4):]
val_ys, val_zs, val_logits, val_preds, val_confs = ys[val_idx], zs[val_idx], logits[val_idx], preds[val_idx], confs[val_idx]
test_ys, test_zs, test_logits, test_preds, test_confs = ys[test_idx], zs[test_idx], logits[test_idx], preds[test_idx], confs[test_idx]
val_probs = softmax(val_logits, axis=-1)
test_probs = softmax(test_logits, axis=-1)



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
val_proximity = np.exp(-val_dists)
test_proximity = np.exp(-test_dists)

val_knndists = np.mean(val_proximity, axis=1)
test_knndists = np.mean(test_proximity, axis=1)

#%%
#################### CALIBRATION METHOD ############################

compare_methods = ['conf', 'temperature_scaling', 'pts', 'pts_knndist', 'ensemble_ts', 'multi_isotonic_regression']
# ['conf', 'temperature_scaling', 'density_estimation', 'pts', 'pts_knndist', 'histogram_binning', 'isotonic_regression', 'ensemble_ts', 'multi_isotonic_regression']
#  ['histogram_binning', 'isotonic_regression', 'multi_isotonic_regression'] , 'multi_isotonic_regression_conf_proximity', 'multi_proximity_isotonic_regression'
# compare_methods = ['conf', 'pts', 'pts_conf', 'pts_knndist']

test_results = {'val':{}, 'test':{}}
if 'conf' in compare_methods:
    test_results['val']['conf'] =  val_probs
    test_results['test']['conf'] =  test_probs

if 'multi_isotonic_regression_conf_proximity' in compare_methods:
    from utils.multi_proximity_isotonic import MultiIsotonicRegression_conf_proximity_wrapper
    
    proximity_bin = 5
    calibrator = MultiIsotonicRegression_conf_proximity_wrapper(proximity_bin=proximity_bin)
    # input (softmax) shape (n_samples, [n_classes])
    probs_val = calibrator.fit_transform(val_logits, val_knndists, val_ys)
    probs_test = calibrator.transform(test_logits, test_knndists)
    
    test_results['val']['multi_isotonic_regression_conf_proximity'] =  probs_val
    test_results['test']['multi_isotonic_regression_conf_proximity'] =  probs_test
    
      
if 'multi_proximity_isotonic_regression' in compare_methods:
    from utils.multi_proximity_isotonic import MultiProximityIsotonicRegression, MultiProximityIsotonicRegression_wrapper
    
    proximity_bin = 5
    calibrator = MultiProximityIsotonicRegression_wrapper(proximity_bin=proximity_bin)
    # input (softmax) shape (n_samples, [n_classes])
    probs_val = calibrator.fit_transform(val_logits, val_knndists, val_ys)
    probs_test = calibrator.transform(test_logits, test_knndists)
    
    # calib_confs_val = np.max(probs_val,axis=-1)
    # calib_confs_test = np.max(probs_test,axis=-1)
    
    test_results['val']['multi_proximity_isotonic_regression'] =  probs_val
    test_results['test']['multi_proximity_isotonic_regression'] =  probs_test


if 'temperature_scaling' in compare_methods:

    from utils.temperature_scaling import  TemperatureScaling

    TS_calibrator = TemperatureScaling(maxiter=100)
    TS_calibrator.fit(val_logits, val_ys)
    best_temp = TS_calibrator.temp 

    probs_val = TS_calibrator.predict(val_logits) 
    probs_test = TS_calibrator.predict(test_logits)
    # confs_ts_val = np.max(probs_val,axis=-1)
    # confs_ts_test = np.max(probs_test,axis=-1)
    test_results['val']['temperature_scaling'] =  probs_val
    test_results['test']['temperature_scaling'] =  probs_test
    
    
if 'histogram_binning' in compare_methods:
    from netcal.binning import HistogramBinning
    """one-verus-all histogram binning: for every class, learn a binary calirator; for every class, learn 10 bins' accuracies"""

    histbin = HistogramBinning(bins=10)
    probs_val = histbin.fit_transform(softmax(val_logits, axis=-1), val_ys)
    probs_test = histbin.transform(softmax(test_logits, axis=-1))
    
    # confs_hist_val = np.max(probs_val,axis=-1)
    # confs_hist_test = np.max(probs_test,axis=-1)
    
    test_results['val']['histogram_binning'] =  probs_val
    test_results['test']['histogram_binning'] =  probs_test
    

if 'isotonic_regression' in compare_methods:
    from netcal.binning import IsotonicRegression
    
    calibrator = IsotonicRegression() # input (softmax) shape (n_samples, [n_classes])
    probs_val = calibrator.fit_transform(softmax(val_logits, axis=-1), val_ys)
    probs_test = calibrator.transform(softmax(test_logits, axis=-1))
    
    # calib_confs_val = np.max(probs_val,axis=-1)
    # calib_confs_test = np.max(probs_test,axis=-1)

    test_results['val']['isotonic_regression'] =  probs_val
    test_results['test']['isotonic_regression'] =  probs_test



# if 'custom_density_estimation' in compare_methods:
#     # TODO
#     from utils.density_ratio_calibration import CustomizeKDECal
    
#     kernel = 'sklearn_kde'
#     kernel_func = 'gaussian'
#     bandwidth = 0.04
#     mirror = False
#     calibrator = CustomizeKDECal(kernel=kernel, kernel_func=kernel_func, bandwidth=bandwidth, mirror=mirror) 
    
#     calibrator.fit(val_confs, val_preds, val_ys, val_knndists, is_conf=True)
#     conf_reg_val = calibrator.predict(val_confs, val_knndists, is_conf=True)
#     conf_reg_test = calibrator.predict(test_logits, test_knndists, is_conf=False)
    
#     test_results['val']['custom_density_estimation'] = conf_reg_val
#     test_results['test']['custom_density_estimation'] = conf_reg_test
#     np.save(osp.join(img_dir, "test_custom_density_estimation_{}.npy".format(args.distance_measure)), conf_reg_test)  
    

if 'density_estimation' in compare_methods:
    from utils.density_ratio_calibration import DensityRatioCalibration

    DER_calibrator = DensityRatioCalibration()
    DER_calibrator.fit(val_logits, val_preds, val_ys, val_knndists)
    prob_reg_val = DER_calibrator.predict(val_logits, val_knndists)
    prob_reg_test = DER_calibrator.predict(test_logits, test_knndists)
    
    test_results['val']['density_estimation'] = prob_reg_val 
    test_results['test']['density_estimation'] = prob_reg_test
    

if 'pts' in compare_methods:


    from utils.parameterized_temp_scaling import ParameterizedTemperatureScaling

    PTS_calibrator = ParameterizedTemperatureScaling(
            epochs=100, # stepsize = 100,000
            lr=0.00005,
            batch_size=1000,
            nlayers=2,
            n_nodes=5,
            length_logits=num_classes,
            top_k_logits=10
    )
    PTS_calibrator.tune(val_logits, val_ys)
    prob_pts_val = PTS_calibrator.calibrate(val_logits)
    prob_pts_test = PTS_calibrator.calibrate(test_logits)
    
    test_results['val']['pts'] =  prob_pts_val
    test_results['test']['pts'] =  prob_pts_test
    
    
if 'pts_conf' in compare_methods:
    # use confidence as input to PTS rather than logits 

    from utils.parameterized_temp_scaling import ParameterizedTemperatureScaling

    PTS_conf_calibrator = ParameterizedTemperatureScaling(
            epochs=100, # stepsize = 100,000
            lr=0.00005,
            batch_size=1000,
            nlayers=2,
            n_nodes=5,
            length_logits=num_classes,
            top_k_logits=10
    )
    PTS_conf_calibrator.tune(val_probs, val_ys)
    conf_pts_conf_val = PTS_conf_calibrator.calibrate(val_probs)
    conf_pts_conf_test = PTS_conf_calibrator.calibrate(test_probs)
    
    test_results['val']['pts_conf'] =  conf_pts_conf_val 
    test_results['test']['pts_conf'] =  conf_pts_conf_test 

if 'pts_knndist' in compare_methods:
    
    from utils.parameterized_temp_scaling import ParameterizedNeighborTemperatureScaling

    top_k_neighbors = 5
    assert top_k_neighbors <= args.num_neighbors
    PTS_knndist_calibrator = ParameterizedNeighborTemperatureScaling(
            epochs=100, # stepsize = 100,000
            lr=0.00005,
            batch_size=1000,
            nlayers=2,
            n_nodes=5,
            length_logits=num_classes,
            top_k_logits=5,
            top_k_neighbors=top_k_neighbors
    )
    PTS_knndist_calibrator.tune(val_logits, val_proximity, val_ys)
    conf_pts_knndist_val = PTS_knndist_calibrator.calibrate(val_logits, val_proximity)
    conf_pts_knndist_test = PTS_knndist_calibrator.calibrate(test_logits, test_proximity)
    
    test_results['val']['pts_knndist'] =  conf_pts_knndist_val
    test_results['test']['pts_knndist'] =  conf_pts_knndist_test  

if 'ensemble_ts' in compare_methods:
    from utils.ensemble_temperature_scaling import EnsembleTemperatureScaling
    calibrator = EnsembleTemperatureScaling() # input (softmax) shape (n_samples, [n_classes])
    probs_val = calibrator.fit_transform(val_logits, val_ys)
    probs_test = calibrator.transform(test_logits)
    
    # calib_confs_val = np.max(probs_val,axis=-1)
    # calib_confs_test = np.max(probs_test,axis=-1)
    
    test_results['val']['ensemble_ts'] =  probs_val 
    test_results['test']['ensemble_ts'] =  probs_test 
    
    
if 'multi_isotonic_regression' in compare_methods:
    from utils.ensemble_temperature_scaling import MultiIsotonicRegression

    calibrator = MultiIsotonicRegression() # input (softmax) shape (n_samples, [n_classes])
    probs_val = calibrator.fit_transform(val_logits, val_ys)
    probs_test = calibrator.transform(test_logits)
    
    # calib_confs_val = np.max(probs_val,axis=-1)
    # calib_confs_test = np.max(probs_test,axis=-1)
    
    test_results['val']['multi_isotonic_regression'] =  probs_val
    test_results['test']['multi_isotonic_regression'] =  probs_test
    
    
########################## KDE PLUG_PlAY ########################################
kde_method = ['conf', 'temperature_scaling', 'pts', 'pts_knndist', 'ensemble_ts']

for method in kde_method:
    if method not in compare_methods:
        continue
    val_prob_score = test_results['val'][method]
    test_prob_score = test_results['test'][method]
    
    from utils.density_ratio_calibration import DensityRatioCalibration

    DER_calibrator = DensityRatioCalibration()
    DER_calibrator.fit(val_prob_score, val_preds, val_ys, val_knndists)
    prob_reg_test = DER_calibrator.predict(test_prob_score, test_knndists)
    
    test_results['test'][method+'_kde'] = prob_reg_test
    compare_methods.append(method+'_kde') 
    
########################## KDE PLUG_PlAY ########################################
binning_method =['histogram_binning', 'isotonic_regression', 'multi_isotonic_regression'] # 'multi_isotonic_regression', 'histogram_binning', 'isotonic_regression'
from utils.multi_proximity_isotonic import BinMeanShift
from utils.ensemble_temperature_scaling import MultiIsotonicRegression
from netcal.binning import IsotonicRegression, HistogramBinning

proximity_bin = 5
for method in binning_method:
    if method not in compare_methods:
        continue
    
    if method == 'histogram_binning':
        calibrator = BinMeanShift('histogram_binning', HistogramBinning, bin_strategy='quantile', normalize_conf=False, proximity_bin=proximity_bin, bins=10)
    elif method == 'isotonic_regression':
        calibrator = BinMeanShift('isotonic_regression', IsotonicRegression, bin_strategy='quantile', normalize_conf=False, proximity_bin=proximity_bin)
    elif method == 'multi_isotonic_regression':
        calibrator = BinMeanShift('multi_isotonic_regression', MultiIsotonicRegression, bin_strategy='quantile', normalize_conf=False, proximity_bin=proximity_bin)
    
    prob_reg_val = calibrator.fit_transform(val_logits, val_knndists, val_ys)
    prob_reg_test = calibrator.transform(test_logits, test_knndists)
    
    # TODO: change kde to other names
    test_results['test'][method+'_kde'] = prob_reg_test
    compare_methods.append(method+'_kde') 
    
# %%
#################### TESTING ECE LOSS ############################
# TODO test how many knnbins are needed and suitable
knnbin = 10
confbin = 15
file_name_to_save = "res_dir/metrics_table_inaturalist_knnbin{}_confbin{}.csv".format(knnbin, confbin)

            
from utils.metrics import evaluate

for method in compare_methods:
    test_prob_score = test_results['test'][method]

    for conf_normalize in [True, False]:
        accuracy_change, ece, mce, ace, toplabel_ece, da_ece, da_mce, loss, brier = evaluate(test_prob_score, test_preds, test_ys, test_knndists, verbose = False, normalize = conf_normalize, conf_bins = confbin, knn_bins=knnbin)

        with open(file_name_to_save, "a") as f:
            write_list = [args.model, val_acc, method, args.random_seed, args.distance_measure, accuracy_change, ece, mce, ace, toplabel_ece, da_ece, da_mce, loss, brier, confbin, knnbin, proximity_bin, conf_normalize]
            entry = ','.join([str(item) for item in write_list])
            f.write(entry)
            f.write("\n")

#%%
################# TEST PROXIMITY BIAS MITIGATION VISUAlIZATION ############################
test_df = pd.DataFrame({'ys':test_ys, 'knndist':test_knndists, 'pred':test_preds})

test_df['correct'] = (test_df.pred == test_df.ys).astype('int')
test_df['knn_bin'] = KBinsDiscretizer(n_bins=6, encode='ordinal').fit_transform(test_knndists.reshape(-1, 1))
# draw the plot on test dataset
group_correct = test_df.groupby('knn_bin')['correct'].mean()
group_knn = test_df.groupby('knn_bin')['knndist'].mean()

markers = ['bo-', 'y+--', 'g^--', 'c+--', 'm^--', 'b+--', 'k^--', 'g+--', 'y^--', 'r+--', 'c^--', 'm+--', 'b^--', 'k+--', 'g^--']
#%%
## conf + temp + acc vs proximity
plt.figure()
colors = plt.cm.BuPu(np.linspace(0, 0.5, 2))
plt.plot(group_knn, group_correct, 'rx-', label='acc')

part_methods = ['conf', 'temperature_scaling']
for method, marker in zip(part_methods, markers):
    test_df[method] = test_results['test'][method][range(test_preds.shape[0]), test_preds]
    group_confs_reg = test_df.groupby('knn_bin')[method].mean()
    plt.plot(group_knn, group_confs_reg, marker, label=method)

plt.title("{}".format(args.model))
plt.legend()
plt.grid(True)
plt.xlabel("proximity")
plt.show()
plt.savefig(osp.join(img_dir, "proximity_bias_{}_K{}".format(args.model, K)), dpi=300)


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

