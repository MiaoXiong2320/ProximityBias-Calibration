
import numpy as np
from sklearn.cluster import KMeans


def get_bin_edges_by_kmeans(proximity, proximity_bin):
    # set initialization to kmeans @ TODO not necessary
    n_bins = proximity_bin 
    column = proximity
    col_min, col_max = column.min(), column.max()
    uniform_edges = np.linspace(col_min, col_max, n_bins + 1)
    init = (uniform_edges[1:] + uniform_edges[:-1])[:, None] * 0.5
    # 1D k-means procedure
    km = KMeans(n_clusters=proximity_bin, init=init, n_init=1)

    centers = km.fit(column[:, None]).cluster_centers_[:, 0]

    # Must sort, centers may be unsorted even with sorted init
    centers.sort()
    bin_edges = (centers[1:] + centers[:-1]) * 0.5
    bin_edges = np.r_[col_min, bin_edges, col_max]
    return bin_edges

def get_bin_edges_by_quantile(proximity, proximity_bin):
    quantiles = np.linspace(0, 100, proximity_bin + 1)
    bin_edges = np.asarray(np.percentile(proximity, quantiles))
    
    # # Remove bins whose width are too small (i.e., <= 1e-8)
    # mask = np.ediff1d(bin_edges, to_begin=np.inf) > 1e-8
    # bin_edges = bin_edges[mask]
    # if len(bin_edges) - 1 != proximity_bin:
    #     print("Bins whose width are too small (i.e., <= 1e-8) are removed. Consider decreasing the number of bins.")
    #     # TODO what if there are multiple bins whose width are too small? then the number of bins will be smaller than proximity_bin - 1 
    #     proximity_bin = len(bin_edges) - 1
        
    return bin_edges

def get_bin_edges_by_uniform(proximity, proximity_bin):
    col_min, col_max = proximity.min(), proximity.max()
    bin_edges = np.linspace(col_min, col_max, proximity_bin + 1)
    
    # # Remove bins whose width are too small (i.e., <= 1e-8)
    # mask = np.ediff1d(bin_edges, to_begin=np.inf) > 1e-8
    # bin_edges = bin_edges[mask]
    # if len(bin_edges) - 1 != proximity_bin:
    #     print("Bins whose width are too small (i.e., <= 1e-8) are removed. Consider decreasing the number of bins.")
    #     # TODO what if there are multiple bins whose width are too small? then the number of bins will be smaller than proximity_bin - 1 
    #     proximity_bin = len(bin_edges) - 1
    return bin_edges