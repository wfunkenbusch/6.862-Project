import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout, Conv1d, MaxPool1d, BatchNorm1d
from torch.optim import Adam

from supp_fun import one_hot_encode

class Net(Module):   
    def __init__(self):
        super(Net, self).__init__()

        self.cnn_initial = Sequential(
            # [N_data, 1, sequence_length, n_encodings]
            Conv2d(1, 256, kernel_size = (7, 48), stride = 1, padding = (3, 0)), #2D convolutional layer
            BatchNorm2d(256),
            ReLU(inplace = True),
            # [N_data, 256, sequence_length, 1]
            )

        self.cnn_final = Sequential(
            Conv1d(256, 256, kernel_size = 7, stride = 1, padding = 3),
            BatchNorm1d(256),
            ReLU(inplace = True),
            MaxPool1d(kernel_size = 3, stride = 3),
            # [N_data, 256, sequence_length/3]

            Conv1d(256, 256, kernel_size = 7, stride = 1, padding = 3),
            BatchNorm1d(256),
            ReLU(inplace = True),
            MaxPool1d(kernel_size = 3, stride = 3),
            # [N_data, 256, sequence_length/9]

            Conv1d(256, 256, kernel_size = 7, stride = 1, padding = 3),
            BatchNorm1d(256),
            ReLU(inplace = True),
            MaxPool1d(kernel_size = 3, stride = 3),
            # [N_data, 256, sequence_length/27]

            Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm1d(256),
            ReLU(inplace = True),

            Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm1d(256),
            ReLU(inplace = True),

            Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm1d(256),
            ReLU(inplace = True),

            Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm1d(256),
            ReLU(inplace = True),

            Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm1d(256),
            ReLU(inplace = True),

            Conv1d(256, 256, kernel_size = 3, stride = 1, padding = 1),
            BatchNorm1d(256),
            ReLU(inplace = True),
            MaxPool1d(kernel_size = 3, stride = 3)
            )

        self.linear_layers = Sequential(
            Linear(int(256 * 13), 2048),
            Dropout(0.5),
            Linear(2048, 2048),
            Dropout(0.5),
            Linear(2048, 1)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = self.cnn_initial(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]))
        x = self.cnn_final(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

def train(model, optimizer, loss, x_train, y_train, x_val, y_val, epoch, epochs, train_losses, val_losses):
    model.train()

    # Getting Data
    x_train, y_train = Variable(x_train), Variable(y_train)
    x_val, y_val = Variable(x_val), Variable(y_val)

    # GPU Support
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    for i in range(epochs):
        # Initialize Gradient
        optimizer.zero_grad()
        
        # Only train on part of data at each step
        x_train_cut, _, y_train_cut, _ = train_test_split(x_train, y_train, test_size = x_train.shape[0] - 500)
        x_val_cut, _, y_val_cut, _ = train_test_split(x_val, y_val, test_size = x_val.shape[0] - 500)

        # Current Model Predictions
        output_train = model(x_train_cut)
        output_val = model(x_val_cut)

        # Current Model Losses
        train_loss = loss(output_train, y_train_cut)
        val_loss = loss(output_val, y_val_cut)

        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        # Back Propagation
        train_loss.backward()
        optimizer.step()

        # Emptying Cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Displaying Losses
        tr_loss = train_loss.item()
        val_loss = val_loss.item()
        print('Epoch : ', epoch + i + 1, '\n    Train Loss:', tr_loss, '\n    Valid Loss:', val_loss)

        if i % 100 == 0:
            torch.save({'epochs': epoch + epochs,
                        'model_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'x_train': x_train,
                        'y_train': y_train,
                        'x_val': x_val,
                        'y_val': y_val,
                        'train_losses': train_losses,
                        'val_losses': val_losses
                        }, 'model.pt')

    return train_losses, val_losses
    
