import numpy as np
import math
import matplotlib
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout
from torch.optim import Adam

### Need to figure out: padding
#                       how many layers/what layers
#                       linear layers output
class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_layers = Sequential(
            # [N_data, 1, sequence_length, n_encodings]
            Conv2d(1, 128, kernel_size = (7, 48), stride = 1, padding = (3, 0)), #2D convolutional layer
            BatchNorm2d(128),
            ReLU(inplace = True),
            # [N_data, 128, sequence_length, 1]
            MaxPool2d(kernel_size = (2, 1), stride = 2),
            # [N_data, 128, sequence_length/2, 1]

            Conv2d(128, 128, kernel_size = (3, 1), stride = 1, padding = 1),
            BatchNorm2d(128),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = (2, 1), stride = 2),
            # [N_data, 128, sequence_length/4, 1]

            Conv2d(128, 128, kernel_size = (3, 1), stride = 1, padding = 1),
            BatchNorm2d(128),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = (2, 1), stride = 2))
            # [N_data, 128, sequence_length/8, 1]

        self.linear_layers = Sequential(
            Linear(int(128 * 138), 1024),
            Linear(1024, 1)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

def train(model, optimizer, loss, x_train, y_train, x_val, y_val, iterations):
    model.train()
    train_losses = []
    val_losses = []

    # Getting the data
    x_train, y_train = Variable(x_train), Variable(y_train)
    x_val, y_val = Variable(x_val), Variable(y_val)

    for iteration in range(iterations):
        optimizer.zero_grad()
        
        # Current model predictions
        output_train = model(x_train)
        output_val = model(x_val)

        # Current model losses
        train_loss = loss(output_train, y_train)
        val_loss = loss(output_val, y_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Updating model
        train_loss.backward()
        optimizer.step()
        tr_loss = train_loss.item()
        val_loss = val_loss.item()
        # printing the validation loss
        print('Epoch : ', iteration + 1, '\n    Train Loss:', tr_loss, '\n    Valid Loss:', val_loss)

    return train_losses, val_losses
    