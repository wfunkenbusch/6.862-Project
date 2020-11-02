import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch.nn import Linear, ReLU, MSELoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout
from torch.optim import Adam

from protein_cnn import *
# Ligand/Proteins pairs will be converted into [N_samples, 1, max_sequence_size, number_of_encodings]

def main():
    n_encodings = 64
    max_sequence_length = 64
    iterations = 100

    n_samples = 100
    x_train = np.random.rand(n_samples, 1, max_sequence_length, n_encodings)
    y_train = np.zeros((n_samples, 1))
    for i in range(n_samples):
        x_train_i = x_train[i, :, :, :].reshape(max_sequence_length, n_encodings)
        y_train[i] = np.sum(x_train_i[0:10, :])/np.sum(x_train_i[10:, :])

    # Get data and split into training and validation
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.1)
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).float()

    # Model Initialization
    model = Net(n_encodings, max_sequence_length)
    optimizer = Adam(model.parameters(), lr = 0.1)
    loss = MSELoss()

    # Model Training
    train_losses, val_losses = train(model, optimizer, loss, x_train, y_train, x_val, y_val, iterations)

    plt.figure()
    plt.plot(range(1, iterations + 1), train_losses, label = 'training loss')
    plt.plot(range(1, iterations + 1), val_losses, label = 'validation loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.savefig('Loss.pdf')

if __name__ == '__main__':
    main()