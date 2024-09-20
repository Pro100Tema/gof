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

    def _transform_pdf(self, n_events=1000):
        if isinstance(self.data, RooAbsPdf):
            data_pdf = self.data.generate(RooArgSet(self.var_lst), n_events)
            return ds2numpy(data_pdf, self.var_lst)
        else:
            raise ValueError("Data is not a valid PDF for transformation.")


class GOFMethods:
    def __init__(self, data, data_mc, var_lst):
        self.data = data
        self.data_mc = data_mc
        self.var_lst = var_lst

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
        permuted_data, permuted_mc_data = self.permute_data()
        permuted_distances_data = self.calculate_distance_ppd(permuted_data, permuted_data, self.var_lst)
        permuted_distances_mc_data = self.calculate_distance_ppd(permuted_data, permuted_mc_data, self.var_lst)
        permuted_T_value = self.calculate_T(permuted_distances_data, permuted_distances_mc_data, len(permuted_data), len(permuted_mc_data))
        return permuted_T_value

    def permute_data(self):
        combined_data = np.concatenate([self.data, self.data_mc])
        np.random.shuffle(combined_data)
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
        if method == 'LD':
            permuted_data, permuted_mc_data = self.permute_data()
            permuted_density_data = self.calculate_local_density(permuted_data, k)
            permuted_density_mc_data = self.calculate_local_density(permuted_mc_data, k)
        elif method == 'KB':
            permuted_data, permuted_mc_data = self.permute_data()
            permuted_density_data = self.calculate_kernel_density(permuted_data, bw_method)
            permuted_density_mc_data = self.calculate_kernel_density(permuted_mc_data, bw_method)
        U_stat, _ = mannwhitneyu(permuted_density_data, permuted_density_mc_data, alternative='two-sided')
        return U_stat

    def calculate_permuted_T_ms(self):
        permuted_data, permuted_mc_data = self.permute_data()
        within_distances = self.calculate_distance_ppd(permuted_data, permuted_data, self.var_lst)
        between_distances = self.calculate_distance_ppd(permuted_data, permuted_mc_data, self.var_lst)
        return self.calculate_T(within_distances, between_distances, len(permuted_data), len(permuted_mc_data))

    def choose_gof_method(self, method, k=5, bw_method='scott', n_permutations=25):
        if len(self.data) == 0 or len(self.data_mc) == 0:
            raise ValueError("Data and MC data must not be empty")

        try:
            # В зависимости от метода выбираем наблюдаемое значение
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
                within_distances = self.calculate_distance_ppd(self.data, self.data, self.var_lst)
                between_distances = self.calculate_distance_ppd(self.data, self.data_mc, self.var_lst)
                observed_T = self.calculate_T(within_distances, between_distances, len(self.data), len(self.data_mc))
                calculate_permuted = self.calculate_permuted_T_ms
                observed_value = observed_T

            else:
                raise ValueError("Invalid method specified")

            try:
                permuted_values = Parallel(n_jobs=-1, backend="loky")(
                    delayed(calculate_permuted)() for _ in range(n_permutations)
                )
            except Exception as e:
                print(f"Error in joblib Parallel: {e}")
                print("Switching to Ostap WorkManager")

                permuted_values = self.work_manager_parallel(calculate_permuted, n_permutations)

            p_value = np.mean(np.array(permuted_values) < observed_value)
            return observed_value, p_value

        except Exception as e:
            print(f"Error in choose_gof_method: {e}")
            traceback.print_exc()
            raise e

    def work_manager_parallel(self, calculate_permuted, n_permutations):

        manager = WorkManager(ncpus=-1, silent=True)
        tasks = [(i,) for i in range(n_permutations)]

        def task_function(_):
            return calculate_permuted()

        # Используем ProgressBar для удобства отслеживания прогресса
        with ProgressBar(min_value=0, max_value=n_permutations) as progress_bar:
            results = manager.process(task_function, tasks, progress=progress_bar)

        return results


def good_fits(data, data_mc=[], var_lst=[], method='PPD', k=5, bw_method='scott'):
    transformer_data = DataToNumpy(data, var_lst)
    data_numpy = transformer_data.transform()

    transformer_mc = DataToNumpy(data_mc, var_lst)
    mc_numpy = transformer_mc.transform()

    gof = GOFMethods(data_numpy, mc_numpy, var_lst)
    
    if method == 'kNN':
        return gof.distance_to_nearest_neighbor(transformer.data_np)

    if method in ['PPD', 'LD', 'KB', 'MS']:
        return gof.choose_gof_method(method, k, bw_method, n_permutations=25)
    
    raise ValueError("Unknown method")
