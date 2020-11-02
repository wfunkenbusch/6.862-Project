import numpy as np
import matplotlib
%matplotlib inline
from sklearn.model_selection import train_test_split

import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout
from torch.optim import Adam
# Ligand/Proteins pairs will be converted into [N_samples, 2, max_sequence_size, number_of_encodings]

def main():
    # defining the model
    model = Net()
    # defining the optimizer
    optimizer = Adam(model.parameters(), lr = 0.1)
    # defining the loss function
    criterion = CrossEntropyLoss()

if __name__ == '__main__':
    main()