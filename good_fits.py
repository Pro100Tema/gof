# input: 
# data - initial dataset 
# data_mc - generated MC dataset
# var_lst - name of variables in data
#
# T_value, p_value = good_fits(dataset, dataset_mc, ['x', 'y'])
# output:
# T_value and p_value calculated using point-to-point-dissimilarity method

from ROOT import * 
import numpy as np
import os
from   ostap.fitting.ds2numpy  import ds2numpy

#point-to-point-dissimilarity method
def dissimilarity_method(data, mc_data, var_lst, sigma=0.01, n_permutations=10):
    nd = len(data)
    nmc = len(mc_data)
    x_val = var_lst[0]

    # определение функции psi (из статьи выбор как e^(-x^2) / (2*sigma^2))
    # значение sigma выбирается из статьи, sigma = 0.01
    def psi(x):
        return np.exp(-x**2 / (2 * sigma**2))

    # вычисление значения T-value по формуле из статьи
    def calculate_T(data, mc_data):
        term1 = np.sum(psi(np.abs(data[x_val][:, None] - data[x_val])), axis=(0, 1))
        term2 = np.sum(psi(np.abs(data[x_val][:, None] - mc_data[x_val])), axis=(0, 1))
        return (1 / (nd**2)) * term1 - (1 / (nd * nmc)) * term2

    observed_T = calculate_T(data, mc_data)

    # реализация перестановочного теста (Permutation test)
    permuted_T_values = np.zeros(n_permutations)

    #повторяется n раз, для получения нескольких значений T-value,
    #для получения значения p_value должно выполняться условие T < T_perm
    for i in range(n_permutations):
        # комбинация и случайный выбор исходных данных и данных МК
        combined_data = np.concatenate([data, mc_data])
        np.random.shuffle(combined_data)

        permuted_data = combined_data[:nd]
        permuted_mc_data = combined_data[nd:]

        #вычисление T-value
        permuted_T_values[i] = calculate_T(permuted_data, permuted_mc_data)

    # выбор p-value
    p_value = np.mean(permuted_T_values >= observed_T)

    return observed_T, p_value

def good_fits(data, data_mc, var_lst):

    ds = ds2numpy(data, var_lst)
    ds_mc = ds2numpy(data_mc, var_lst)

    T_value, p_value = dissimilarity_method(ds, ds_mc, var_lst)

    return T_value, p_value
