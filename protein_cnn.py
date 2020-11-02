import numpy as np
import matplotlib
%matplotlib inline
from sklearn.model_selection import train_test_split

import torch
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout
from torch.optim import Adam

class Net(Module):   
    def __init__(self, n_encodings, max_sequence_length):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(1, 16, kernel_size = (3, n_encodings), stride = 1),
            BatchNorm2d(16),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = 2, stride = 2),
            # 2nd 2D convolution layer
            Conv2d(16, 16, kernel_size = (3, n_encodings), stride = 1, padding = 1),
            BatchNorm2d(16),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = 2, stride = 2),
        )

        self.linear_layers = Sequential(
            Linear(16 * n_encodings * max_sequence_length, 1)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
