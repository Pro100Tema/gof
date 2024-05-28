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
from scipy.spatial.distance import cdist
import concurrent.futures
from numba import jit

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
def calculate_T(sum_nd, sum_mc, nd, nmc):
    return (1 / (nd**2)) * sum_nd - (1 / (nd * nmc)) * sum_mc

## Combination and random selection of original data and MC data
def permute_data(data, mc_data):
    combined_data = np.concatenate([data, mc_data])
    np.random.shuffle(combined_data)
    permuted_data = combined_data[:len(data)]
    permuted_mc_data = combined_data[len(data):]
    return permuted_data, permuted_mc_data

# calculation ditance between points of datasets
def calculate_distance_iterative(data1, data2, var_lst):
    sum_distances = 0
    for i in range(len(data1)):
        for j in range(len(data2)):
            sum_distances += np.linalg.norm([data1[i][var] - data2[j][var] for var in var_lst])
    return sum_distances

# using cdist for calculate distance of large datasets (len > 10000)
def calculate_distance(data1, data2, var_lst):
    data1_vars = np.column_stack([data1[var] for var in var_lst])
    data2_vars = np.column_stack([data2[var] for var in var_lst])
    distances = cdist(data1_vars, data2_vars, 'euclidean').flatten()
    return np.sqrt(np.sum(distances))

# Implementation of the permutation test for each variable
# Repeated n times to obtain multiple T-value instances,
# to get the p_value, the condition T < T_perm must be satisfied
def calculate_permuted_T(data, mc_data, var_lst, large_data=False):
    permuted_data, permuted_mc_data = permute_data(data, mc_data)
    if large_data:
        permuted_distances_data = calculate_distance_iterative(permuted_data, permuted_data, var_lst)
        permuted_distances_mc_data = calculate_distance_iterative(permuted_data, permuted_mc_data, var_lst)
    else:
        permuted_distances_data = calculate_distance(permuted_data, permuted_data, var_lst)
        permuted_distances_mc_data = calculate_distance(permuted_data, permuted_mc_data, var_lst)
    permuted_T_value = calculate_T(permuted_distances_data, permuted_distances_mc_data, len(permuted_data), len(permuted_mc_data))
    return permuted_T_value

def dissimilarity_method(data, mc_data, var_lst, n_permutations=25):
    nd = len(data)
    nmc = len(mc_data)

    large_data = nd < 10000 or nmc < 10000

    if large_data:
        distances_data = calculate_distance_iterative(data, data, var_lst)
        distances_mc_data = calculate_distance_iterative(data, mc_data, var_lst)
    else:
        distances_data = calculate_distance(data, data, var_lst)
        distances_mc_data = calculate_distance(data, mc_data, var_lst)

    observed_T = calculate_T(distances_data, distances_mc_data, nd, nmc)
    
    # add parallel process to speed up permutation test work 
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = [executor.submit(calculate_permuted_T, data, mc_data, var_lst, large_data) for _ in range(10)]
        permuted_T_values = np.array([task.result() for task in concurrent.futures.as_completed(tasks)])

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
