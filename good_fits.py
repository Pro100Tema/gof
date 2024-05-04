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
    #term1 = np.sum(psi_distances_data)
    #term2 = np.sum(psi_distances_mc_data)
    return (1 / (nd**2)) * sum_nd - (1 / (nd * nmc)) * sum_mc

# Combination and random selection of original data and MC data
def permute_data(data, mc_data):
    combined_data = np.concatenate([data, mc_data])
    np.random.shuffle(combined_data)
    permuted_data = combined_data[:len(data)]
    permuted_mc_data = combined_data[len(data):]
    return permuted_data, permuted_mc_data

# old method using cdist
#def calculate_distance(data1, data2, var_lst):
#    data1_vars = np.column_stack([data1[var] for var in var_lst])
#    data2_vars = np.column_stack([data2[var] for var in var_lst])
#
#    distances = cdist(data1_vars, data2_vars, 'euclidean').flatten()
#    return distances

# Calculate distance between each pair of points
@jit(nopython=True)
def calculate_block_distance(data_block):
    n = data_block.shape[0]
    total_distance = 0.0

    # Iterate over all pairs of points
    for i in range(n):
        for j in range(i + 1, n):
            # Calculate squared Euclidean distance between points
            # and add square to total distance
            sum_sq = np.sum((data_block[i] - data_block[j]) ** 2)
            total_distance += np.sqrt(sum_sq)
    return total_distance

# Calculate total distance within a dataset
def calculate_distance_nd(data, var_lst, block_size=1000):
    data_array = np.column_stack([data[var] for var in var_lst])
    total_distance = 0
    num_blocks = (len(data) + block_size - 1) // block_size

    # Iterate over all data blocks
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, len(data))
        data_block = data_array[start:end]
        total_distance += calculate_block_distance(data_block)

    return total_distance

# Calculate distance between each pair of points from data and mc_data
@jit(nopython=True)
def calculate_block_distances(data_block, mc_data_block):
    n = data_block.shape[0]
    m = mc_data_block.shape[0]
    total_distance = 0.0
    # Iterate over all pairs of points from data and mc_data
    for i in range(n):
        for j in range(m):
            # Calculate squared Euclidean distance between points
            # and add square to total distance
            sum_sq = np.sum((data_block[i] - mc_data_block[j]) ** 2)
            total_distance += np.sqrt(sum_sq)
    return total_distance

# Calculate total distance between data and mc_data
def calculate_distance_nmc(data, mc_data, var_lst, block_size=1000):
    data_array = np.column_stack([data[var] for var in var_lst])
    mc_data_array = np.column_stack([mc_data[var] for var in var_lst])

    total_distance = 0
    num_blocks_data = (len(data) + block_size - 1) // block_size
    num_blocks_mc_data = (len(mc_data) + block_size - 1) // block_size
    
    # Iterate over all data blocks
    for i in range(num_blocks_data):
        start_i = i * block_size
        end_i = min((i + 1) * block_size, len(data))
        data_block = data_array[start_i:end_i]

        for j in range(num_blocks_mc_data):
            start_j = j * block_size
            end_j = min((j + 1) * block_size, len(mc_data))
            mc_data_block = mc_data_array[start_j:end_j]

            total_distance += calculate_block_distances(data_block, mc_data_block)

    return total_distance


# Implementation of the permutation test for each variable
# Repeated n times to obtain multiple T-value instances,
# to get the p_value, the condition T < T_perm must be satisfied
def calculate_permuted_T(data, mc_data, var_lst):
    permuted_data, permuted_mc_data = permute_data(data, mc_data)
    permuted_distances_data = calculate_distance_nd(permuted_data, var_lst)
    permuted_distances_mc_data = calculate_distance_nmc(permuted_data, permuted_mc_data, var_lst)
    permuted_T_value = calculate_T(permuted_distances_data, permuted_distances_mc_data, len(permuted_data), len(permuted_mc_data))
    return permuted_T_value


def dissimilarity_method(data, mc_data, var_lst, n_permutations=25):
    nd = len(data)
    nmc = len(mc_data)

    # Pre-calculate distances between original data and MC data
    distances_data = calculate_distance_nd(data, var_lst)
    distances_mc_data = calculate_distance_nmc(data, mc_data, var_lst)

    # Pre-calculate psi function values for all distances
    #psi_distances_data = psi(distances_data)
    #psi_distances_mc_data = psi(distances_mc_data)

    observed_T = calculate_T(distances_data, distances_mc_data, nd, nmc)


    #PREVIOUS VERSION
    #permuted_T_values = np.zeros(n_permutations)
    #for i in range(n_permutations):
        #permuted_data, permuted_mc_data = permute_data(data, mc_data)
        #permuted_distances_data = calculate_distance(permuted_data, permuted_data, var_lst)
        #permuted_distances_mc_data = calculate_distance(permuted_data, permuted_mc_data, var_lst)
        #permuted_T_values[i] = calculate_T(permuted_distances_data, permuted_distances_mc_data, nd, nmc)
        #permuted_T_values[i] = calculate_permuted_T(data, mc_data, var_lst, sigma)


    # add parallel process to speed up permutation test work 
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = [executor.submit(calculate_permuted_T, data, mc_data, var_lst) for _ in range(10)]
        permuted_T_values = np.array([task.result() for task in concurrent.futures.as_completed(tasks)])

    #with concurrent.futures.ThreadPoolExecutor() as executor:
    #    tasks = [executor.submit(calculate_permuted_T, data, mc_data, var_lst) for _ in range(10)]
    #    permuted_T_values = np.array([task.result() for task in concurrent.futures.as_completed(tasks)])


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
