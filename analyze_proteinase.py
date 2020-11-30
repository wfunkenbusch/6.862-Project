#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch.nn import Linear, ReLU, MSELoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d
from torch.optim import Adam

from protein_cnn import *
from supp_fun import *

# Loading Data
x_data = np.load('data_proteinase.npy')

# Loading Model
model = Net()
optimizer = Adam(model.parameters(), lr = 1e-5)
loss = MSELoss()
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
loss = checkpoint['loss']

model.eval()

x1, x2 = train_test_split(x_data, test_size = 0.75, shuffle = False)
x2, x3 = train_test_split(x2, test_size = 0.67, shuffle = False)
x3, x4 = train_test_split(x3, test_size = 0.5, shuffle = False)
x1 = torch.from_numpy(one_hot_encode(x1, 48)).float()
x2 = torch.from_numpy(one_hot_encode(x2, 48)).float()
x3 = torch.from_numpy(one_hot_encode(x3, 48)).float()
x4 = torch.from_numpy(one_hot_encode(x4, 48)).float()

del x_data

y1 = model(x1).detach().numpy()
y2 = model(x2).detach().numpy()
y3 = model(x3).detach().numpy()
y4 = model(x4).detach().numpy()

y = np.concatenate((y1, y2, y3, y4), axis = 0)
del y1, y2, y3, y4

best_y = np.argsort(y[:, 0])[:10]

for i in range(len(best_y)):
    print('x Index: ', best_y[i])
    print('    Ki: ', 10**y[best_y[i]][0])
