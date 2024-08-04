# input: 
# data - initial dataset 
# data_mc - generated MC dataset
# var_lst - name of variables in data
# method - name of selected method ('kNN', 'PPD', 'LD', 'KB', 'MS')
# T_value, p_value = good_fits(dataset, dataset_mc, ['x', 'y'], 'PPD')
# output:
# T_value and p_value for each variable from var_lst calculated using point-to-point-dissimilarity method

from ROOT import * 
import numpy as np
import os
from   ostap.fitting.ds2numpy  import ds2numpy
import tracemalloc
from scipy.spatial.distance import cdist
import concurrent.futures
import traceback
from numba import njit
from joblib import Parallel, delayed
from scipy.spatial import KDTree
from scipy.stats import mannwhitneyu, gaussian_kde

# Interface for classes that implement comparison methods
class SimilarityInterface:
    def compare_datasets(self, dataset1, dataset2):
        raise NotImplementedError("This method should be overridden by subclasses")

    def calculate_permuted_statistic(self, dataset1, dataset2):
        raise NotImplementedError("This method should be overridden by subclasses")

# Class that implements the PPD method for comparing datasets
class PPDSimilarity(SimilarityInterface):
    def __init__(self, var_lst, chunk_size=1000):
        self.var_lst = var_lst
        self.chunk_size = chunk_size

    # Calculation of the T-value according to the formula from the article
    @staticmethod
    @njit
    def calculate_T(sum_nd, sum_mc, nd, nmc):
        return (1 / (nd**2)) * sum_nd - (1 / (nd * nmc)) * sum_mc

    # Efficient distance calculation with chunking for large datasets
    @staticmethod
    @njit
    def calculate_chunk_distances(chunk, data2_vars):
        distances = np.sqrt(np.sum((chunk[:, np.newaxis, :] - data2_vars[np.newaxis, :, :])**2, axis=2)).flatten()
        return np.sum(distances)

    # if len(data_rd) * len(data_mc) > 10^7, using cdist method to calculate distance in ppd,
    # else separeate data and calculate distance
    def calculate_distance(self, data1, data2):
        data1_vars = np.column_stack([data1[var] for var in self.var_lst])
        data2_vars = np.column_stack([data2[var] for var in self.var_lst])

        if len(data1) * len(data2) > 1e7: # Arbitrary threshold for switching methods
            n1 = data1_vars.shape[0]
            dists = 0.0
            for i in range(0, n1, self.chunk_size):
                chunk = data1_vars[i:i + self.chunk_size]
                dists += self.calculate_chunk_distances(chunk, data2_vars)
            return dists
        else:
            distances = cdist(data1_vars, data2_vars, 'euclidean').flatten()
            return np.sum(distances)

    # Implementation of the permutation test for each variable
    def compare_datasets(self, dataset1, dataset2):
        # Pre-calculate distances between original data and MC data
        distances_data = self.calculate_distance(dataset1, dataset1)
        distances_mc_data = self.calculate_distance(dataset1, dataset2)
        T_value = self.calculate_T(distances_data, distances_mc_data, len(dataset1), len(dataset2))
        return T_value

    # Combination and random selection of original data and MC data
    def calculate_permuted_statistic(self, data, mc_data):
        permuted_data, permuted_mc_data = self.permute_data(data, mc_data)
        permuted_distances_data = self.calculate_distance(permuted_data, permuted_data)
        permuted_distances_mc_data = self.calculate_distance(permuted_data, permuted_mc_data)
        permuted_T_value = self.calculate_T(permuted_distances_data, permuted_distances_mc_data, len(permuted_data), len(permuted_mc_data))
        return permuted_T_value

    # Combination and random selection of original data and MC data
    @staticmethod
    def permute_data(data, mc_data):
        combined_data = np.concatenate([data, mc_data])
        np.random.shuffle(combined_data)
        permuted_data = combined_data[:len(data)]
        permuted_mc_data = combined_data[len(data):]
        return permuted_data, permuted_mc_data

# Class for choosing a comparison method and performing calculations
class GOFMethods:
    def __init__(self, method, var_lst=None, chunk_size=1000):
        if method == 'PPD':
            self.similarity_method = PPDSimilarity(var_lst, chunk_size)
        else:
            raise ValueError("Invalid method specified")

    def compare_datasets(self, dataset1, dataset2):
        return self.similarity_method.compare_datasets(dataset1, dataset2)

    def calculate_permuted_statistics(self, dataset1, dataset2, n_permutations=25):
        # add parallel processing
        permuted_values = Parallel(n_jobs=-1, backend="loky")(
            delayed(self.similarity_method.calculate_permuted_statistic)(dataset1, dataset2) for _ in range(n_permutations)
        )
        return np.array(permuted_values)

    def calculate_p_value(self, observed_value, permuted_values):
        return np.mean(permuted_values < observed_value)


def good_fits(data, data_mc=[], var_lst=[], method='PPD', chunk_size=1000, n_permutations=25):
    if not isinstance(var_lst, list) or not all(isinstance(var, str) for var in var_lst):
        raise TypeError("Var list must be a list of strings")
    if method not in ['PPD']:
        raise ValueError("Unknown method. Use 'PPD'")

    ds = ds2numpy(data, var_lst)
    ds_mc = ds2numpy(data_mc, var_lst)

    gof = GOFMethods(method, var_lst=var_lst, chunk_size=chunk_size)
    observed_value = gof.compare_datasets(ds, ds_mc)

    permuted_values = gof.calculate_permuted_statistics(ds, ds_mc, n_permutations)
    p_value = gof.calculate_p_value(observed_value, permuted_values)
    return observed_value, p_value


#########################################################################################
#########################################################################################
#################################### Test Results #######################################

# PPD: len(rd) = 10^2, len(mc) = 10^4, p-value = 0.1, time = 12.3 sec, max cpu = 16.4 MB
# PPD: len(rd) = 10^4, len(mc) = 10^4, p-value = 0.6, time = 23.1 sec, max cpu = 250.3 MB

#########################################################################################
#########################################################################################
