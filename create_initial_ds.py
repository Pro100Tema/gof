from ROOT import *
import random as rd
import numpy as np 

x = RooRealVar("x", "x", 0, 100)
y = RooRealVar("y", "y", 0, 100)

# Задаем параметры функции Гаусса
mean = 0.0
sigma = 1.0

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

for i in dataset:
    print(i.x, i.y)
