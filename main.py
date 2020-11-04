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
# Ligand/Proteins pairs will be converted into [N_samples, 1, max_sequence_size, number_of_encodings]

def main():
    n_encodings = 48
    max_sequence_length = 1100
    iterations = 50
    lr = 2e-6

    x_data = np.load('x_data.npy')
    y_data = np.load('y_data.npy')

    x_train = one_hot_encode(x_data, n_encodings)
    y_train = np.log(y_data)

    # Get data and split into training and validation
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1)
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).float()

    # Model Initialization
    model = Net()
    optimizer = Adam(model.parameters(), lr = lr)
    loss = MSELoss()

    if torch.cuda.is_available():
        model = model.cuda()
        loss = loss.cuda()

    # Model Training
    train_losses, val_losses = train(model, optimizer, loss, x_train, y_train, x_val, y_val, iterations)

    # Plotting losses as function of epoch
    plt.figure()
    plt.plot(range(1, iterations + 1), train_losses, label = 'training loss')
    plt.plot(range(1, iterations + 1), val_losses, label = 'validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.savefig('Loss.pdf')

if __name__ == '__main__':
    main()