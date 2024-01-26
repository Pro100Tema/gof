from ROOT import *
import random as rd
import numpy as np 


def create_mc_data(data):
    x = RooRealVar("x", "x", -100, 100)
    y = RooRealVar("y", "y", -100, 100)

    # Задаем параметры функции Гаусса
    mean = 0.0
    sigma = 5.0

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


def create_mc_data(data):
    x = RooRealVar("x", "x", -100, 100)
    y = RooRealVar("y", "y", -100, 100)

    # Задаем параметры функции Гаусса
    mean = 0.0
    sigma = 5.0

    # Задаем количество точек в изначальном датасете
    num_points = 100

    varset = RooArgSet(x, y)

    dataset = ROOT.RooDataSet("dataset", "dataset", varset)

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

    dataset.SaveAs('dataset.root')
    return dataset


#for i in dataset:
#    print(i.x, i.y)
