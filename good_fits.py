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
import tracemalloc

# decorator that tracks the maximum memory usage
# and returnss information about the location of the maximum memory usage
def memory_test(func):
    def wrapper(*args, **kwargs):
        # start tracking memory usage
        tracemalloc.start()
        # call the original function with the provided arguments
        result = func(*args, **kwargs)
        snapshot = tracemalloc.take_snapshot()
        # get statistics of memory usage for each line of code
        top_stats = snapshot.statistics('lineno')

        # find the maximum memory usage and 
        # line of code with the maximum memory usage
        max_memory = max(stat.size for stat in top_stats)
        max_memory_line = next(stat for stat in top_stats if stat.size == max_memory)
        print("Max memory usage:", max_memory, "byte")
        print("Max memory usage location:", max_memory_line.traceback.format())

        # Stop tracking memory usage
        tracemalloc.stop()
        return result
    return wrapper


@memory_test
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
#def calculate_distance(data1, data2, var_lst):
#    num_points = len(data1)
#    distances = np.zeros(num_points)
#    for i in range(num_points):
#        diff_squared = np.sum([(data1[i][var] - data2[i][var])**2 for var in var_lst])
#        distances[i] = np.sqrt(diff_squared)
#    return distances

from scipy.spatial.distance import cdist
# Calculation distance between two points from datasets
def calculate_distance(data1, data2, var_lst):
    data1_vars = np.column_stack([data1[var] for var in var_lst])
    data2_vars = np.column_stack([data2[var] for var in var_lst])

    distances = cdist(data1_vars, data2_vars, 'euclidean').flatten()
    return distances



@memory_test
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

    # Calculate p-value as the fraction of cases 
    # where the sum of T-values for permuted data
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
