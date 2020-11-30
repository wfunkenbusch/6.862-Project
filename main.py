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
    epoch = 0
    epochs = 0
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
    x_train = x_data
    y_train = y_data

    indices = []
    for i in range(y_train.size):
        if y_train[i] >= 1e4 or y_train[i] == 1e3:
            indices.append(i)

    x_train = np.delete(x_train, indices, axis = 0)
    y_train = np.delete(y_train, indices)
    y_train = y_train.reshape((y_train.shape[0], 1))

    del x_data
    del y_data

    # Get data and split into training and validation
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = test_size)
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

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # GPU Support
    if torch.cuda.is_available():
        model = model.cuda()
        loss = loss.cuda()

    # Model Training
    train_losses, val_losses = train(model, optimizer, loss, x_train, y_train, x_val, y_val, epoch, epochs, train_losses, val_losses)

    # Saving Result
    torch.save({
            'epochs': epoch + epochs,
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
    plt.rc('axes', axisbelow = True)
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'k', label = 'training loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, 'r')
    plt.scatter(range(1, len(val_losses) + 1), val_losses, s = 3, c = 'r', label = 'validation loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.savefig('Loss.pdf')

    # Plotting predicted vs actual values post-training
    model.eval()

    x_val1, x_val2, y_val1, y_val2 = train_test_split(x_val, y_val, test_size = 0.5)
    x_val1 = torch.from_numpy(one_hot_encode(x_val1, 48)).float()
    x_val2 = torch.from_numpy(one_hot_encode(x_val2, 48)).float()
    y_val1 = torch.from_numpy(np.log10(y_val1)).float()
    y_val2 = torch.from_numpy(np.log10(y_val2)).float()
    y_pred1 = model(x_val1)
    y_pred2 = model(x_val2)

    val_loss = (loss(y_pred1, y_val1).item() + loss(y_pred2, y_val2).item())/2
    print('Validation Loss: ', val_loss)

    plt.figure()
    plt.plot(range(-4, 5), range(-4, 5), 'k')
    plt.plot(range(-4, 5), range(-3, 6), 'r-')
    plt.plot(range(-4, 5), range(-5, 4), 'r-')
    plt.scatter(y_val1.detach().numpy(), y_pred1.detach().numpy(), s = 1,  c = 'royalblue')
    plt.scatter(y_val2.detach().numpy(), y_pred2.detach().numpy(), s = 1, c = 'royalblue')
    plt.grid()
    plt.xlabel('Actual $log(K_i)$')
    plt.ylabel('Predicted $log(K_i)$')
    plt.savefig('Accuracy.pdf')

if __name__ == '__main__':
    main()
