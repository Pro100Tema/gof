from ROOT import * 
import numpy as np
import os
from   ostap.fitting.ds2numpy  import ds2numpy

def dissimilarity_method(data, mc_data, sigma_bar=0.01, n_permutations=10):
    nd = len(data)
    nmc = len(mc_data)

    # Define the weighting function
    def psi(x):
        return np.exp(-x**2 / (2 * sigma_bar**2))

    # Calculate the dissimilarity statistic T
    def calculate_T(data, mc_data):
        term1 = np.sum([psi(np.linalg.norm(data[i]['x'] - data[j]['x'])) for i in range(nd) for j in range(i + 1, nd)])
        term2 = np.sum([psi(np.linalg.norm(data[i]['x'] - mc_data[j]['x'])) for i in range(nd) for j in range(nmc)])
        term3 = np.sum([psi(np.linalg.norm(mc_data[i]['x'] - data[j]['x'])) for i in range(nmc) for j in range(nd)])
        return (1 / (nd**2)) * term1 - (1 / (nd * nmc)) * (term2 + term3)

    observed_T = calculate_T(data, mc_data)

    # Permutation test
    permuted_T_values = []
    for _ in range(n_permutations):
        combined_data = np.concatenate([data, mc_data])
        np.random.shuffle(combined_data)

        permuted_data = combined_data[:nd]
        permuted_mc_data = combined_data[nd:]

        permuted_T = calculate_T(permuted_data, permuted_mc_data)
        permuted_T_values.append(permuted_T)

    # Calculate p-value
    p_value = np.mean(permuted_T_values >= observed_T)

    return observed_T, p_value


def create_mc_data(data):
    x = RooRealVar("x", "x", 0, 100)
    y = RooRealVar("y", "y", 0, 100)

    # Задаем параметры функции Гаусса
    mean = 0.0
    sigma = 1.0

    # Задаем количество точек в изначальном датасете
    num_points = len(data) * 100

    varset = RooArgSet(x, y)

    dataset = ROOT.RooDataSet("dataset_mc", "dataset_mc", varset)

    # Генерируем случайные данные для x
    x_val = np.random.normal(mean, sigma, num_points)

    # Генерируем соответствующие значения функции Гаусса
    y_val = np.exp(-((x_val - mean) / sigma)**2 / 2) / (sigma * np.sqrt(2 * np.pi))

    # Нормализуем значения функции, чтобы их сумма была равна 1
    y_val /= np.sum(y_val)

    for i in range(num_points):
        x.setVal(x_val[i])
        y.setVal(y_val[i])
        dataset.add(varset)

    dataset.SaveAs('dataset_mc.root')
    return dataset


def good_fits(data, data_mc, var_lst):
    ds = ds2numpy(data, var_lst)
    ds_mc = ds2numpy(data_mc, var_lst)

    observed_statistic, p_value = dissimilarity_method(ds, ds_mc)

    print("Observed Dissimilarity Statistic:", observed_statistic)
    print("Permutation Test P-value:", p_value)

f = ROOT.TFile('dataset.root', 'read')
dataset = f['dataset']
f.close()

f = ROOT.TFile('dataset_mc.root', 'read')
dataset_mc = f['dataset_mc']
f.close()

ws = good_fits(dataset, dataset_mc, ['x', 'y'])
