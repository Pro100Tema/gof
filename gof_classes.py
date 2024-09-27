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
from ROOT import RooAbsPdf, RooDataSet
from scipy.spatial import cKDTree

class DataToNumpy:
    def __init__(self, data, var_lst):
        self.data = data
        self.var_lst = var_lst
    
    def transform(self):
        if isinstance(self.data, np.ndarray):
            return self.data
        elif isinstance(self.data, RooDataSet):
            return ds2numpy(self.data, self.var_lst)
        elif isinstance(self.data, RooAbsPdf):
            return self._transform_pdf()
        else:
            raise ValueError(f"Unsupported data type: {type(self.data)}")

    def _transform_pdf(self):
        if isinstance(self.data, RooAbsPdf):
            data_pdf = self.data.generate(RooArgSet(*self.var_lst))
            return ds2numpy(data_pdf, self.var_lst)
        else:
            raise ValueError("Data is not a valid PDF for transformation.")

class GOFMethods:
    def __init__(self, data, data_mc, var_lst):
        self.data = data
        self.data_mc = data_mc
        self.var_lst = var_lst

    def distance_to_nearest_neighbor(self, data):
        tree = cKDTree(self.data) # creating k-tree
        distances, _ = tree.query(self.data, k=2) # calculating the distance to the nearest neighbor
        return np.mean(distances[:, 1])
    
    @staticmethod
    def calculate_T(sum_nd, sum_mc, nd, nmc):
        return (1 / (nd**2)) * sum_nd - (1 / (nd * nmc)) * sum_mc

    @staticmethod
    def calculate_chunk_distances(chunk, data2_vars):
        distances = np.sqrt(np.sum((chunk[:, np.newaxis, :] - data2_vars[np.newaxis, :, :])**2, axis=2)).flatten()
        return np.sum(distances)

    @staticmethod
    def calculate_distance_ppd(data1, data2, var_lst, chunk_size=1000):
        data1_vars = np.column_stack([data1[var] for var in var_lst])
        data2_vars = np.column_stack([data2[var] for var in var_lst])

        if len(data1) * len(data2) > 1e7:
            dists = 0.0
            for i in range(0, len(data1_vars), chunk_size):
                chunk = data1_vars[i:i + chunk_size]
                dists += GOFMethods.calculate_chunk_distances(chunk, data2_vars)
            return dists
        else:
            distances = cdist(data1_vars, data2_vars, 'euclidean').flatten()
            return np.sum(distances)

    def calculate_permuted_T_ppd(self):
        permuted_data, permuted_mc_data = self.permute_and_split('PPD')
        permuted_distances_data = self.calculate_distance_ppd(permuted_data, permuted_data, self.var_lst)
        permuted_distances_mc_data = self.calculate_distance_ppd(permuted_data, permuted_mc_data, self.var_lst)
        permuted_T_value = self.calculate_T(permuted_distances_data, permuted_distances_mc_data, len(permuted_data), len(permuted_mc_data))
        return permuted_T_value

    def permute_and_split(self, method):
        combined_data = np.concatenate([self.data, self.data_mc])
        np.random.shuffle(combined_data)

        if method == 'MS':
            half_point = len(combined_data) // 2
            return combined_data[:half_point], combined_data[half_point:]
        else:
            return combined_data[:len(self.data)], combined_data[len(self.data):]

    def calculate_local_density(self, data, k=5):
        kdtree = KDTree(np.vstack([data[var] for var in data.dtype.names]).T)
        densities = []
        for point in data:
            distances, _ = kdtree.query([np.array([point[var] for var in data.dtype.names])], k=k+1)
            densities.append(np.mean(distances[0][1:]))
        return np.array(densities)

    def calculate_kernel_density(self, data, bw_method='scott'):
        data_numeric = np.vstack([data[var] for var in data.dtype.names]).T
        kde = gaussian_kde(data_numeric.T, bw_method=bw_method)
        return kde(data_numeric.T)

    def calculate_permuted_U(self, method, k=5, bw_method='scott'):
        permuted_data, permuted_mc_data = self.permute_and_split('LD')
        if method == 'LD':
            permuted_density_data = self.calculate_local_density(permuted_data, k)
            permuted_density_mc_data = self.calculate_local_density(permuted_mc_data, k)
        elif method == 'KB':
            permuted_density_data = self.calculate_kernel_density(permuted_data, bw_method)
            permuted_density_mc_data = self.calculate_kernel_density(permuted_mc_data, bw_method)
        U_stat, _ = mannwhitneyu(permuted_density_data, permuted_density_mc_data, alternative='two-sided')
        return U_stat


    def calculate_T_MS(self, sum_within, sum_between, n_within, n_between):
        return (1 / (n_within**2)) * sum_within - (1 / (n_within * n_between)) * sum_between

    def calculate_distances_MS(self, data1, data2):
        data1_numeric = np.vstack([data1[var] for var in data1.dtype.names]).T
        data2_numeric = np.vstack([data2[var] for var in data2.dtype.names]).T
        return cdist(data1_numeric, data2_numeric, 'euclidean')

    def sum_distances(self, distances):
        return np.sum(distances)

    def calculate_permuted_T_mixed(self, data, mc_data):

        permuted_data, permuted_mc_data = self.permute_and_split('MS')
        within_distances = self.calculate_distances_MS(permuted_data, permuted_data)
        between_distances = self.calculate_distances_MS(permuted_data, permuted_mc_data)
        sum_within = self.sum_distances(within_distances)
        sum_between = self.sum_distances(between_distances)
        permuted_T_value = self.calculate_T_MS(sum_within, sum_between, len(permuted_data), len(permuted_mc_data))
        return permuted_T_value


    def choose_gof_method(self, method, k=5, bw_method='scott', n_permutations=25):
        if len(self.data) == 0 or len(self.data_mc) == 0:
            raise ValueError("Data and MC data must not be empty")

        try:
            if method == 'PPD':
                distances_data = self.calculate_distance_ppd(self.data, self.data, self.var_lst)
                distances_mc_data = self.calculate_distance_ppd(self.data, self.data_mc, self.var_lst)
                observed_T = self.calculate_T(distances_data, distances_mc_data, len(self.data), len(self.data_mc))
                calculate_permuted = self.calculate_permuted_T_ppd
                observed_value = observed_T

            elif method in ['LD', 'KB']:
                if method == 'LD':
                    density_data = self.calculate_local_density(self.data, k)
                    density_mc_data = self.calculate_local_density(self.data_mc, k)
                elif method == 'KB':
                    density_data = self.calculate_kernel_density(self.data, bw_method)
                    density_mc_data = self.calculate_kernel_density(self.data_mc, bw_method)
                observed_U, _ = mannwhitneyu(density_data, density_mc_data, alternative='two-sided')
                calculate_permuted = lambda: self.calculate_permuted_U(method, k, bw_method)
                observed_value = observed_U

            elif method == 'MS':
                observed_within_distances = self.calculate_distances_MS(self.data, self.data)
                observed_between_distances = self.calculate_distances_MS(self.data, self.data_mc)
                sum_within = self.sum_distances(observed_within_distances)
                sum_between = self.sum_distances(observed_between_distances)
                observed_T = self.calculate_T_MS(sum_within, sum_between, len(self.data), len(self.data_mc))
                calculate_permuted = lambda: self.calculate_permuted_T_mixed(self.data, self.data_mc)
                observed_value = observed_T

            else:
                raise ValueError("Invalid method specified")

            try:
                permuted_values = Parallel(n_jobs=-1, backend="loky")(
                    delayed(calculate_permuted)() for _ in range(n_permutations)
                )
            except Exception as e_joblib:
                print(f"Error in joblib Parallel: {e_joblib}")
                print("Switching to to sequential processing...")
                permuted_values = [calculate_permuted() for _ in range(n_permutations)]

            p_value = np.mean(np.array(permuted_values) < observed_value)
            return observed_value, p_value

        except Exception as e:
            print(f"Error in choose_gof_method: {e}")
            traceback.print_exc()
            raise e


def good_fits(data, data_mc=[], var_lst=[], method='PPD', k=5, bw_method='scott'):

    if method == 'kNN':
        gof2 = GOFMethods(data, data_mc, var_lst)
        return gof2.distance_to_nearest_neighbor(data)

    transformer_data = DataToNumpy(data, var_lst)
    data_numpy = transformer_data.transform()

    transformer_mc = DataToNumpy(data_mc, var_lst)
    mc_numpy = transformer_mc.transform()

    gof = GOFMethods(data_numpy, mc_numpy, var_lst)

    if method in ['PPD', 'LD', 'KB', 'MS']:
        return gof.choose_gof_method(method, k, bw_method, n_permutations=25)
    
    raise ValueError("Unknown method")




#########################################################################################
#########################################################################################
#################################### New Test Results ###################################

# PPD: len(rd) = 10^2, len(mc) = 10^4, p-value = 0.1, time = 10.3 sec, max cpu = 16.4 MB
# PPD: len(rd) = 10^4, len(mc) = 10^4, p-value = 0.6, time = 22.9 sec, max cpu = 250.3 MB

# kNN: len(rd) = 10^2, p-value = 0.0, time = 0.03 sec, max cpu = 0.04 MB
# kNN: len(rd) = 10^4, p-value = 0.0, time = 1.2 sec, max cpu = 2.0 MB
# kNN: len(rd) = 10^5, p-value = 0.0, time = 30.5 sec, max cpu = 20.0 MB

# LD: len(rd) = 10^2, len(mc) = 10^4, p-value = 0.36, time = 2.1 sec, max cpu = 1.1 MB
# LD: len(rd) = 10^4, len(mc) = 10^4, p-value = 0.88, time = 4.1 sec, max cpu = 1.9 MB
# LD: len(rd) = 10^4, len(mc) = 10^5, p-value = 0.0, time = 23.4 sec, max cpu = 10.6 MB

# KB: len(rd) = 10^2, len(mc) = 10^4, p-value = 0.0, time = 3.8 sec, max cpu = 1.3 MB
# KB: len(rd) = 10^4, len(mc) = 10^4, p-value = 0.64, time = 7.4 sec, max cpu = 2.4 MB
# KB: len(rd) = 10^4, len(mc) = 10^5, p-value = 1.0, time = 374.7 sec, max cpu = 13.3 MB

# MS: len(rd) = 10^2, len(mc) = 10^4, p-value = 0.0, time = 0.8 sec, max cpu = 8.7 MB
# MS: len(rd) = 10^4, len(mc) = 10^4, p-value = 0.6, time = 13.6 sec, max cpu = 1601.1 MB

#########################################################################################
#########################################################################################
