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

    tree = cKDTree(data) # создание k-мерного дерева 
    distances, _ = tree.query(data, k=2) # расчет расстояния до ближайшего соседа

    return distances[:, 1]


def dissimilarity_method(data, mc_data, var_lst, sigma=0.01, n_permutations=100):
    nd = len(data)
    nmc = len(mc_data)

    # Определение функции psi (из статьи выбор как e^(-x^2) / (2*sigma^2))
    # Значение sigma выбирается из статьи, sigma = 0.01
    def psi(x):
        return np.exp(-x**2 / (2 * sigma**2))

    # Вычисление значения T-value по формуле из статьи
    def calculate_T(data, mc_data, var):
        term1 = np.sum(psi(np.abs(data[var][:, None] - data[var])), axis=(0, 1))
        term2 = np.sum(psi(np.abs(data[var][:, None] - mc_data[var])), axis=(0, 1))
        return (1 / (nd**2)) * term1 - (1 / (nd * nmc)) * term2

    #numpy массивы для вычисленных значений T и p-value
    observed_T_values = np.zeros(len(var_lst))
    p_values = np.zeros(len(var_lst))

    # Реализация перестановочного теста (Permutation test) для каждой переменной
    #повторяется n раз, для получения нескольких значений T-value,
    #для получения значения p_value должно выполняться условие T < T_perm
    for idx, var in enumerate(var_lst):
        observed_T_values_var = np.zeros(n_permutations)
        for i in range(n_permutations):
            # комбинация и случайный выбор исходных данных и данных МК
            combined_data = np.concatenate([data, mc_data])
            np.random.shuffle(combined_data)

            permuted_data = combined_data[:nd]
            permuted_mc_data = combined_data[nd:]

            permuted_T = calculate_T(permuted_data, permuted_mc_data, var)
            observed_T_values_var[i] = permuted_T
        
        #вычисление T-value
        observed_T = calculate_T(data, mc_data, var)
        # выбор p-value
        p_values[idx] = np.mean(observed_T_values_var >= observed_T)
        observed_T_values[idx] = observed_T

    return observed_T_values, p_values

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
