# input: 
# data - initial dataset 
# data_mc - generated MC dataset
# var_lst - name of variables in data
# method - name of selected method ('kNN', 'PPD', 'LD', 'KB')
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


def distance_to_nearest_neighbor(data):
    from scipy.spatial import cKDTree

    tree = cKDTree(data) # creating k-tree
    distances, _ = tree.query(data, k=2) # calculating the distance to the nearest neighbor

    return distances[:, 1]

# Calculation of the T-value according to the formula from the article
@njit
def calculate_T(sum_nd, sum_mc, nd, nmc):
    return (1 / (nd**2)) * sum_nd - (1 / (nd * nmc)) * sum_mc

# Combination and random selection of original data and MC data
def permute_data(data, mc_data):
    combined_data = np.concatenate([data, mc_data])
    np.random.shuffle(combined_data)
    permuted_data = combined_data[:len(data)]
    permuted_mc_data = combined_data[len(data):]
    return permuted_data, permuted_mc_data

# Efficient distance calculation with chunking for large datasets
@njit
def calculate_chunk_distances(chunk, data2_vars):
    distances = np.sqrt(np.sum((chunk[:, np.newaxis, :] - data2_vars[np.newaxis, :, :])**2, axis=2)).flatten()
    return np.sum(distances)

def calculate_distance(data1, data2, var_lst, chunk_size=1000):
    data1_vars = np.column_stack([data1[var] for var in var_lst])
    data2_vars = np.column_stack([data2[var] for var in var_lst])

    if len(data1) * len(data2) > 1e7:  # Arbitrary threshold for switching methods
        #print("циклы")
        n1 = data1_vars.shape[0]
        #n2 = data2_vars.shape[0]

        #distances = []
        dists = 0.0
        for i in range(0, n1, chunk_size):
            chunk = data1_vars[i:i + chunk_size]
            #dists = np.sqrt(np.sum((chunk[:, np.newaxis, :] - data2_vars[np.newaxis, :, :])**2, axis=2)).flatten()
            dists += calculate_chunk_distances(chunk, data2_vars)
            #distances.append(dists)

        #return np.concatenate(distances, axis=0)
        return dists
    else:
        #print("cdist")
        #data1_vars = np.column_stack([data1[var] for var in var_lst])
        #data2_vars = np.column_stack([data2[var] for var in var_lst])
        distances = cdist(data1_vars, data2_vars, 'euclidean').flatten()
        return np.sum(distances)

# Implementation of the permutation test for each variable
def calculate_permuted_T(data, mc_data, var_lst):
    permuted_data, permuted_mc_data = permute_data(data, mc_data)
    permuted_distances_data = calculate_distance(permuted_data, permuted_data, var_lst)
    permuted_distances_mc_data = calculate_distance(permuted_data, permuted_mc_data, var_lst)
    permuted_T_value = calculate_T(permuted_distances_data, permuted_distances_mc_data, len(permuted_data), len(permuted_mc_data))
    return permuted_T_value

def dissimilarity_method(data, mc_data, var_lst, n_permutations=25):
    if len(data) == 0 or len(mc_data) == 0:
        raise ValueError("Data and MC data must not be empty")

    nd = len(data)
    nmc = len(mc_data)

    # Pre-calculate distances between original data and MC data
    distances_data = calculate_distance(data, data, var_lst)
    distances_mc_data = calculate_distance(data, mc_data, var_lst)

    observed_T = calculate_T(distances_data, distances_mc_data, nd, nmc)


    # Parallel processing of permutation tests
    try:
        permuted_T_values = Parallel(n_jobs=-1, backend="loky")(
            delayed(calculate_permuted_T)(data, mc_data, var_lst) for _ in range(n_permutations)
        )
        permuted_T_values = np.array(permuted_T_values)
    except Exception as e:
        print(f"Error in joblib Parallel: {e}")
        traceback.print_exc()
        raise e

    # Calculate p-value
    p_value = np.mean(permuted_T_values < observed_T)

    return observed_T, p_value

# Function to calculate local density using structured array
def calculate_local_density(data, k=5):
    kdtree = KDTree(np.vstack([data[var] for var in data.dtype.names]).T)
    densities = []
    for point in data:
        distances, _ = kdtree.query([np.array([point[var] for var in data.dtype.names])], k=k+1)
        densities.append(np.mean(distances[0][1:]))  # Exclude the point itself
    return np.array(densities)

# Function to calculate permuted U-value
def calculate_permuted_U_LD(data, mc_data, k):
    permuted_data, permuted_mc_data = permute_data(data, mc_data)
    permuted_density_data = calculate_local_density(permuted_data, k)
    permuted_density_mc_data = calculate_local_density(permuted_mc_data, k)
    U_stat, _ = mannwhitneyu(permuted_density_data, permuted_density_mc_data, alternative='two-sided')
    return U_stat

# Main function to execute Local-Density method
def local_density_method(data, mc_data, k=5, n_permutations=25):
    if len(data) == 0 or len(mc_data) == 0:
        raise ValueError("Data and MC data must not be empty")

    # Calculate local densities for original data and MC data
    density_data = calculate_local_density(data, k)
    density_mc_data = calculate_local_density(mc_data, k)

    # Perform Mann-Whitney U test
    observed_U, p_value = mannwhitneyu(density_data, density_mc_data, alternative='two-sided')

    # Parallel processing of permutation tests
    try:
        permuted_U_values = Parallel(n_jobs=-1, backend="loky")(
            delayed(calculate_permuted_U_LD)(data, mc_data, k) for _ in range(n_permutations)
        )
        permuted_U_values = np.array(permuted_U_values)
    except Exception as e:
        print(f"Error in joblib Parallel: {e}")
        traceback.print_exc()
        raise e

    # Calculate permutation p-value
    permutation_p_value = np.mean(permuted_U_values < observed_U)

    return observed_U, permutation_p_value

# Function to calculate kernel density
def calculate_kernel_density(data, bw_method='scott'):
    data_numeric = np.vstack([data[var] for var in data.dtype.names]).T
    kde = gaussian_kde(data_numeric.T, bw_method=bw_method)
    densities = kde(data_numeric.T)
    return densities

# Function to calculate permuted U-value
def calculate_permuted_U_KB(data, mc_data, bw_method):
    permuted_data, permuted_mc_data = permute_data(data, mc_data)
    permuted_density_data = calculate_kernel_density(permuted_data, bw_method)
    permuted_density_mc_data = calculate_kernel_density(permuted_mc_data, bw_method)
    U_stat, _ = mannwhitneyu(permuted_density_data, permuted_density_mc_data, alternative='two-sided')
    return U_stat

# Main function to execute Kernel-Based method
def kernel_based_method(data, mc_data, bw_method='scott', n_permutations=25):
    if len(data) == 0 or len(mc_data) == 0:
        raise ValueError("Data and MC data must not be empty")

    # Calculate kernel densities for original data and MC data
    density_data = calculate_kernel_density(data, bw_method)
    density_mc_data = calculate_kernel_density(mc_data, bw_method)

    # Perform Mann-Whitney U test
    observed_U, p_value = mannwhitneyu(density_data, density_mc_data, alternative='two-sided')

    # Parallel processing of permutation tests
    try:
        permuted_U_values = Parallel(n_jobs=-1, backend="loky")(
            delayed(calculate_permuted_U_KB)(data, mc_data, bw_method) for _ in range(n_permutations)
        )
        permuted_U_values = np.array(permuted_U_values)
    except Exception as e:
        print(f"Error in joblib Parallel: {e}")
        traceback.print_exc()
        raise e

    # Calculate permutation p-value
    permutation_p_value = np.mean(permuted_U_values < observed_U)

    return observed_U, permutation_p_value

def calculate_T_MS(sum_within, sum_between, n_within, n_between):
    return (1 / (n_within**2)) * sum_within - (1 / (n_within * n_between)) * sum_between

def permute_and_split(data, mc_data):
    combined_data = np.concatenate([data, mc_data])
    np.random.shuffle(combined_data)
    half_point = len(combined_data) // 2
    return combined_data[:half_point], combined_data[half_point:]

def calculate_distances_MS(data1, data2):
    data1_numeric = np.vstack([data1[var] for var in data1.dtype.names]).T
    data2_numeric = np.vstack([data2[var] for var in data2.dtype.names]).T
    return cdist(data1_numeric, data2_numeric, 'euclidean')

def sum_distances(distances):
    return np.sum(distances)

def calculate_permuted_T_mixed(data, mc_data):
    permuted_data, permuted_mc_data = permute_and_split(data, mc_data)
    within_distances = calculate_distances_MS(permuted_data, permuted_data)
    between_distances = calculate_distances_MS(permuted_data, permuted_mc_data)
    sum_within = sum_distances(within_distances)
    sum_between = sum_distances(between_distances)
    permuted_T_value = calculate_T_MS(sum_within, sum_between, len(permuted_data), len(permuted_mc_data))
    return permuted_T_value

def mixed_samples_method(data, mc_data, n_permutations=25):
    if len(data) == 0 or len(mc_data) == 0:
        raise ValueError("Data and MC data must not be empty")

    nd = len(data)
    nmc = len(mc_data)

    observed_within_distances = calculate_distances_MS(data, data)
    observed_between_distances = calculate_distances_MS(data, mc_data)

    sum_within = sum_distances(observed_within_distances)
    sum_between = sum_distances(observed_between_distances)

    observed_T = calculate_T_MS(sum_within, sum_between, nd, nmc)

    try:
        permuted_T_values = Parallel(n_jobs=-1, backend="loky")(
            delayed(calculate_permuted_T_mixed)(data, mc_data) for _ in range(n_permutations)
        )
        permuted_T_values = np.array(permuted_T_values)
    except Exception as e:
        print(f"Error in joblib Parallel: {e}")
        traceback.print_exc()
        raise e

    p_value = np.mean(permuted_T_values < observed_T)

    return observed_T, p_value


def good_fits(data, data_mc=[], var_lst=[], method='PPD', k = 5, bw_method='scott'):

    if not isinstance(var_lst, list) or not all(isinstance(var, str) for var in var_lst):
        raise TypeError("Var list must be a list of strings")
    if method not in ['kNN', 'PPD', 'LD', 'KB', 'MS']:
        raise ValueError("Unknown method. Use 'kNN', 'PPD', 'LD', 'KB' or 'MS'")
    
    ds = ds2numpy(data, var_lst)
    ds_mc = ds2numpy(data_mc, var_lst)
    
    if method == 'kNN':
        return distance_to_nearest_neighbor(data)
    elif method == 'PPD':
        T_value, p_value = dissimilarity_method(ds, ds_mc, var_lst)
        return T_value, p_value
    elif method == 'LD':
        U_value_LD, p_value = local_density_method(ds, ds_mc, k)
        return U_value_LD, p_value
    elif method == 'KB':
        U_value_KB, p_value = kernel_based_method(ds, ds_mc, bw_method)
        return U_value_KB, p_value
    elif method == 'MS':
        T_value, p_value = mixed_samples_method(ds, ds_mc)
        return T_value, p_value
    else:
        raise ValueError("Unknown method")
