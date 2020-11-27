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
    epochs = 5000
    lr = 1e-5
    test_size = 0.1
    train_losses = []
    val_losses = []

    # Model Initialization
    model = Net()
    optimizer = Adam(model.parameters(), lr = lr)
    loss = MSELoss()

    ############## If Training New Model
    # Load data
    x_data = np.load('x_data.npy')
    y_data = np.load('y_data.npy')

    # Convert to proper format
    x_train = x_data
    y_train = y_data

    del x_data
    del y_data

    # Get data and split into training and validation
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = test_size)

    ############## If Loading from Previous Run
    '''
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
    '''

    # GPU Support
    if torch.cuda.is_available():
        model = model.cuda()
        loss = loss.cuda()

    # Model Training
    model.train()
    train_losses, val_losses = train(model, optimizer, loss, x_train, y_train, x_val, y_val, epoch, epochs, train_losses, val_losses)

    # Saving Results
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
    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, label = 'training loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label = 'validation loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid()
    plt.legend()
    plt.savefig('Loss.pdf')

    # Plotting predicted vs actual values post-training
    model.eval()

    x_val1, x_val2, y_val1, y_val2 = train_test_split(x_val, y_val, test_size = 0.5)
    y_pred1 = model(x_val1)
    y_pred2 = model(x_val2)

    plt.figure()
    plt.plot(range(-2, 6), range(-2, 6))
    plt.scatter(y_val1.detach().numpy(), y_pred1.detach().numpy(), c = 'k')
    plt.scatter(y_val2.detach().numpy(), y_pred2.detach().numpy(), c = 'k')
    plt.grid()
    plt.xlabel('Actual logK_i')
    plt.ylabel('Predicted logK_i')
    plt.savefig('Accuracy.pdf')

if __name__ == '__main__':
    main()
