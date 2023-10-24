import matplotlib.pyplot as plt
import os, time, pdb, sys
import pandas as pd
import seaborn as sns
import numpy as np
import os.path as osp

def density_map_plot(val_ys, val_knndists, val_confs, val_preds, calibrator, img_dir, method_name):
    
    # visualize the distribution of correctly classified & incorrectly classified samples using MSP        
    val_df_msp = pd.DataFrame({'ys':val_ys, 'knndist':val_knndists, 'conf':val_confs, 'pred':val_preds})
    val_df_msp['correct'] = (val_df_msp.pred == val_df_msp.ys).astype('int')
    val_df_msp_true = val_df_msp[val_df_msp['correct'] == 1]
    val_df_msp_false = val_df_msp[val_df_msp['correct'] == 0]

    plt.figure(figsize=(12,15))
    plt.subplot(3, 2, 1)
    plt.scatter(val_df_msp_true['knndist'], val_df_msp_true['conf'], s=2, alpha=0.5)
    plt.title("correct classified")
    plt.ylabel("conf")
    plt.xlabel("knndists")

    plt.subplot(3, 2, 2)
    plt.scatter(val_df_msp_false['knndist'], val_df_msp_false['conf'], s=2, alpha=0.5)
    plt.title("wrongly classified")
    plt.xlabel("knndists")
    # plt.savefig(osp.join(img_dir, "pd_distribution_msp"), dpi=300)

    """
    02. density map for the correctly & incorrectly classified samples
    """
    # Now plot the decision boundary using a fine mesh as input to a
    # filled contour plot

    # Build a coarser grid to plot a set of ensemble classifications
    # to show how these are different to what we see in the decision
    # surfaces. These points are regularly space and do not have a
    # black outline
    plot_step_coarser = 0.01
    x_min, x_max = val_df_msp['knndist'].min() - 0.1, val_df_msp['knndist'].max() + 0.1
    y_min, y_max = val_df_msp['conf'].min() - 0.1, val_df_msp['conf'].max() + 0.1
    xx_coarser, yy_coarser = np.meshgrid(
        np.arange(x_min, x_max, plot_step_coarser),
        np.arange(y_min, y_max, plot_step_coarser))


    plt.subplot(3, 2, 3)
    Z_points_coarser = calibrator.dens_true_pdf(yy_coarser.ravel(), xx_coarser.ravel(), is_conf=True).reshape(xx_coarser.shape)
    cs_points = plt.scatter(xx_coarser, yy_coarser,
                            c=Z_points_coarser, cmap=plt.cm.RdYlBu,
                            edgecolors="none")
    plt.colorbar(cs_points)
    plt.title("correct classified")


    plt.subplot(3, 2, 4)
    Z_points_coarser = calibrator.dens_false_pdf(yy_coarser.ravel(), xx_coarser.ravel(), is_conf=True).reshape(xx_coarser.shape)
    cs_points = plt.scatter(xx_coarser, yy_coarser,
                            c=Z_points_coarser, cmap=plt.cm.RdYlBu,
                            edgecolors="none")
    plt.colorbar(cs_points)
    plt.title("wrongly classified")


    plt.subplot(3, 2, 5)
    Z_points_coarser = calibrator.predict(yy_coarser.ravel(),xx_coarser.ravel(), is_conf=True).reshape(xx_coarser.shape)
    cs_points = plt.scatter(xx_coarser, yy_coarser,
                            c=Z_points_coarser, cmap=plt.cm.RdYlBu,
                            edgecolors="none")
    plt.colorbar(cs_points)
    plt.title("output calibrated confidence")
    plt.savefig(osp.join(img_dir, "density_map_msp_grid_{}".format(method_name)), dpi=300)

    #%%
    # this is used to visualize the density map for the correctly & incorrectly classified samples
    plt.figure(figsize=(15,12))
    plt.subplot(2, 2, 1)
    Z_points_coarser = calibrator.dens_true_pdf(val_confs, val_knndists, is_conf=True).reshape(val_knndists.shape)
    cs_points = plt.scatter(val_knndists,val_confs,
                            c=Z_points_coarser, cmap=plt.cm.RdYlBu,
                            edgecolors="none")
    plt.colorbar(cs_points)
    plt.title("correct classified")


    plt.subplot(2, 2, 2)
    Z_points_coarser = calibrator.dens_false_pdf(val_confs, val_knndists, is_conf=True).reshape(val_knndists.shape)
    cs_points = plt.scatter(val_knndists,val_confs,
                            c=Z_points_coarser, cmap=plt.cm.RdYlBu,
                            edgecolors="none")
    plt.colorbar(cs_points)
    plt.title("wrongly classified")


    plt.subplot(2, 2, 3)
    Z_points_coarser = calibrator.predict(val_confs,val_knndists, is_conf=True).reshape(val_knndists.shape)
    cs_points = plt.scatter(val_knndists,val_confs,
                            c=Z_points_coarser, cmap=plt.cm.RdYlBu,
                            edgecolors="none")
    plt.colorbar(cs_points)
    plt.title("output calibrated confidence")

    plt.subplot(2, 2, 4)
    cs_points = plt.scatter(val_knndists,val_confs,
                            c=(val_ys==val_preds), cmap=plt.cm.RdYlBu,
                            edgecolors="none", alpha=0.3)
    plt.colorbar(cs_points)
    plt.title("is correctly classified?")

    plt.savefig(osp.join(img_dir, "density_map_msp_trainingpoints_{}".format(method_name)), dpi=300)