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
    # Input parameters
    n_encodings = 48
    max_sequence_length = 1100
    epochs = 45
    lr = 1e-5
    test_size = 0.1
    train_losses = []
    val_losses = []

    # Model Initialization
    model = Net()
    optimizer = Adam(model.parameters(), lr = lr)
    loss = MSELoss()

    ############## If Training New Model
'''
    # Load data
    x_data = np.load('x_data.npy')
    y_data = np.load('y_data.npy')

    # Convert to proper format
    x_train = one_hot_encode(x_data, n_encodings)
    y_train = np.log(y_data)

    del x_data
    del y_data

    # Get data and split into training and validation
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = test_size)
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).float()
'''

    ############## If Loading from Previous Run
    checkpoint = torch.load('model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epochs']
    loss = checkpoint['loss']
    x_train = checkpoint['x_train']
    y_train = checkpoint['y_train']
    x_val = checkpoint['x_val']
    y_val = checkpoint['y_val']
    train_losses = checkpoint['train_losses']
    val_losses = checkpoint['val_losses']

    # GPU Support
    if torch.cuda.is_available():
        model = model.cuda()
        loss = loss.cuda()

    # Model Training
    train_losses, val_losses = train(model, optimizer, loss, x_train, y_train, x_val, y_val, epochs, train_losses, val_losses)

    # Saving Results
    torch.save({
            'epochs': epochs,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'x_train': x_train,
            'y_train': y_train,
            'x_val': x_val,
            'y_val': y_val,
            'train_losses': train_losses,
            'val_losses': val_losses
            }, 'model.pt')

    # Plotting losses as function of epoch
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label = 'training loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label = 'validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.savefig('Loss.pdf')

    # Plotting predicted vs actual values post-training
    model.eval()
    y_pred = model(x_val)
    plt.figure()
    plt.plot(range(-2, 12), range(-2, 12))
    plt.scatter(y_val.detach().numpy(), y_pred.detach().numpy(), c = 'k')
    plt.grid()
    plt.xlabel('Actual log(K_i)')
    plt.ylabel('Predicted log(K_i)')
    plt.savefig('Accuracy.pdf')

if __name__ == '__main__':
    main()