# input: 
# data - initial dataset 
# data_mc - generated MC dataset
# var_lst - name of variables in data
# method - name of selected method ('kNN', 'dism')
# T_value, p_value = good_fits(dataset, dataset_mc, ['x', 'y'], 'kNN')
# output:
# T_value and p_value for each variable from var_lst calculated using point-to-point-dissimilarity method

from ROOT import * 
import numpy as np
import os
from   ostap.fitting.ds2numpy  import ds2numpy


def distance_to_nearest_neighbor(data):
    from scipy.spatial import cKDTree

    tree = cKDTree(data) # creating k-tree
    distances, _ = tree.query(data, k=2) # calculating the distance to the nearest neighbor

    return distances[:, 1]
    

# Calculation of the T-value according to the formula from the article
def calculate_T(psi_distances_data, psi_distances_mc_data, nd, nmc):
    term1 = np.sum(psi_distances_data)
    term2 = np.sum(psi_distances_mc_data)
    return (1 / (nd**2)) * term1 - (1 / (nd * nmc)) * term2

# Combination and random selection of original data and MC data
def permute_data(data, mc_data):
    combined_data = np.concatenate([data, mc_data])
    np.random.shuffle(combined_data)
    permuted_data = combined_data[:len(data)]
    permuted_mc_data = combined_data[len(data):]
    return permuted_data, permuted_mc_data

# Definition of the psi function (psi(x) = e^(-x^2) / (2*sigma^2))
# The value of sigma is chosen: sigma = 0.01
def psi(x, sigma=0.01):
    return np.exp(-x**2 / (2 * sigma**2))

# Calculation distance between two points from datasets
def calculate_distance(data1, data2, var_lst):
    diff_squared = 0
    for var in var_lst:
        diff_squared += (data1[var][:, None] - data2[var])**2
    return np.sqrt(diff_squared)

def dissimilarity_method(data, mc_data, var_lst, sigma=0.01, n_permutations=1000):
    nd = len(data)
    nmc = len(mc_data)

    # Pre-calculate distances between original data and MC data
    distances_data = calculate_distance(data, data, var_lst)
    distances_mc_data = calculate_distance(data, mc_data, var_lst)

    # Pre-calculate psi function values for all distances
    psi_distances_data = psi(distances_data)
    psi_distances_mc_data = psi(distances_mc_data)

    observed_T = calculate_T(psi_distances_data, psi_distances_mc_data, nd, nmc)

    # Implementation of the permutation test for each variable
    # Repeated n times to obtain multiple T-value instances,
    # to get the p_value, the condition T < T_perm must be satisfied
    permuted_T_values = np.zeros(n_permutations)
    for i in range(n_permutations):
        permuted_data, permuted_mc_data = permute_data(data, mc_data)
        permuted_distances_data = calculate_distance(permuted_data, permuted_data, var_lst)
        permuted_distances_mc_data = calculate_distance(permuted_data, permuted_mc_data, var_lst)
        permuted_T_values[i] = calculate_T(psi(permuted_distances_data), psi(permuted_distances_mc_data), nd, nmc)

    # Calculate p-value as the fraction of cases where the sum of T-values for permuted data
    # is less than the sum of T-values for observed data
    p_value = np.mean(permuted_T_values < observed_T)

    return observed_T, p_value


def good_fits(data, data_mc = [], var_lst = [], method = 'dism'):

    if method.__eq__('kNN'):
        return distance_to_nearest_neighbor(data)

    elif method.__eq__('dism'):
        ds = ds2numpy(data, var_lst)
        ds_mc = ds2numpy(data_mc, var_lst)
        T_value, p_value = dissimilarity_method(ds, ds_mc, var_lst)
        return T_value, p_value
    else:
        raise ValueError("Uknown method")
