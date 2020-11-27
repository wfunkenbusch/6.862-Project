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

        # CNN for SMILES representation
        self.cnn_SMILES = Sequential(
            # [N_data, 1, sequence_length, n_encodings]
            # Standard 2D CNN layer
            Conv2d(1, 256, kernel_size = (7, 28), stride = 1, padding = (3, 0)),
            BatchNorm2d(256),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = (3, 1), stride = (3, 1)),
            Permute(),
            # [N_data, 256, sequence_length/3, 1]

            Conv2d(1, 256, kernel_size = (7, 256), stride = 1, padding = (3, 0)),
            BatchNorm2d(256),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = (3, 1), stride = (3, 1)),
            Permute(),
            # [N_data, 256, sequence_length/9, 1]

            Conv2d(1, 256, kernel_size = (7, 256), stride = 1, padding = (3, 0)),
            BatchNorm2d(256),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = (3, 1), stride = (3, 1)),
            Permute(),
            # [N_data, 256, sequence_length/27, 1]

            Conv2d(1, 256, kernel_size = (3, 256), stride = 1, padding = (1, 0)),
            BatchNorm2d(256),
            ReLU(inplace = True),
            Permute(),

            Conv2d(1, 256, kernel_size = (3, 256), stride = 1, padding = (1, 0)),
            BatchNorm2d(256),
            ReLU(inplace = True),
            Permute(),

            Conv2d(1, 256, kernel_size = (3, 256), stride = 1, padding = (1, 0)),
            BatchNorm2d(256),
            ReLU(inplace = True),
            Permute(),

            Conv2d(1, 256, kernel_size = (3, 256), stride = 1, padding = (1, 0)),
            BatchNorm2d(256),
            ReLU(inplace = True),
            Permute(),

            Conv2d(1, 256, kernel_size = (3, 256), stride = 1, padding = (1, 0)),
            BatchNorm2d(256),
            ReLU(inplace = True),
            Permute(),

            Conv2d(1, 256, kernel_size = (3, 256), stride = 1, padding = (1, 0)),
            BatchNorm2d(256),
            ReLU(inplace = True),
            Permute(),
            MaxPool2d(kernel_size = (3, 1), stride = (3, 1))
            # [N_data, 256, sequence_length/51, 1]
        )

        # CNN for protein sequence
        self.cnn_protein = Sequential(
            # [N_data, 1, sequence_length, n_encodings]
            Conv2d(1, 256, kernel_size = (7, 20), stride = 1, padding = (3, 0)), #2D convolutional layer
            BatchNorm2d(256),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = (3, 1), stride = (3, 1)),
            Permute(),
            # [N_data, 256, sequence_length/3, 1]

            Conv2d(1, 256, kernel_size = (7, 256), stride = 1, padding = (3, 0)),
            BatchNorm2d(256),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = (3, 1), stride = (3, 1)),
            Permute(),
            # [N_data, 256, sequence_length/9, 1]

            Conv2d(1, 256, kernel_size = (7, 256), stride = 1, padding = (3, 0)),
            BatchNorm2d(256),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = (3, 1), stride = (3, 1)),
            Permute(),
            # [N_data, 256, sequence_length/27, 1]

            Conv2d(1, 256, kernel_size = (3, 256), stride = 1, padding = (1, 0)),
            BatchNorm2d(256),
            ReLU(inplace = True),
            Permute(),

            Conv2d(1, 256, kernel_size = (3, 256), stride = 1, padding = (1, 0)),
            BatchNorm2d(256),
            ReLU(inplace = True),
            Permute(),

            Conv2d(1, 256, kernel_size = (3, 256), stride = 1, padding = (1, 0)),
            BatchNorm2d(256),
            ReLU(inplace = True),
            Permute(),

            Conv2d(1, 256, kernel_size = (3, 256), stride = 1, padding = (1, 0)),
            BatchNorm2d(256),
            ReLU(inplace = True),
            Permute(),

            Conv2d(1, 256, kernel_size = (3, 256), stride = 1, padding = (1, 0)),
            BatchNorm2d(256),
            ReLU(inplace = True),
            Permute(),

            Conv2d(1, 256, kernel_size = (3, 256), stride = 1, padding = (1, 0)),
            BatchNorm2d(256),
            ReLU(inplace = True),
            Permute(),
            MaxPool2d(kernel_size = (3, 1), stride = (3, 1))
            # [N_data, 256, sequence_length/51, 1]
        )

        # Combined fully connected layers
        self.linear_layers = Sequential(
            Linear(int(256 * 13), 2048),
            Dropout(0.5),
            Linear(2048, 2048),
            Dropout(0.5),
            Linear(2048, 2048),
            Dropout(0.5),
            Linear(2048, 1)
        )

    # Defining the forward pass    
    def forward(self, x):
        # Splitting data into SMILES and protein sections
        x1 = x[:, :, 0:100, 0:28]
        x2 = x[:, :, 100:, 28:]

        # CNN's for each representation
        x1 = self.cnn_SMILES(x1)
        x2 = self.cnn_protein(x2)

        # Combining into single representation
        x = torch.cat((x1, x2), dim = 2)

        # Flattening and running fully connected layers
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

class Permute(torch.nn.Module):
    # Module for permuting tensor to appropriate dimensions within Sequential()
    def forward(self, x):
        return x.permute(0, 3, 2, 1)

def train(model, optimizer, loss, x_train0, y_train0, x_val0, y_val0, epoch, epochs, train_losses, val_losses):
    # Enter training mode
    model.train()

    # Converting data to correct format for PyTorch
    n_encodings = 48
    x_train = one_hot_encode(x_train0, n_encodings)
    x_val = one_hot_encode(x_val0, n_encodings)
    y_train = np.log10(y_train0)
    y_val = np.log10(y_val0)

    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).float()

    # Getting Data
    x_train, y_train = Variable(x_train), Variable(y_train)
    x_val, y_val = Variable(x_val), Variable(y_val)

    # Checking for GPU Support
    if torch.cuda.is_available():
        x_train = x_train.cuda()
        y_train = y_train.cuda()
        x_val = x_val.cuda()
        y_val = y_val.cuda()

    for i in range(epochs):
        # Initialize Gradient
        optimizer.zero_grad()
        
        # Only train on part of data at each step (500 random points)
        x_train_cut, _, y_train_cut, _ = train_test_split(x_train, y_train, test_size = x_train.shape[0] - 500)
        x_val_cut, _, y_val_cut, _ = train_test_split(x_val, y_val, test_size = x_val.shape[0] - 500)

        # Current Model Predictions
        output_train = model(x_train_cut)
        output_val = model(x_val_cut)

        # Current Model Losses
        train_loss = loss(output_train, y_train_cut)
        val_loss = loss(output_val, y_val_cut)

        # Storing Model Losses
        train_losses.append(train_loss.item())
        val_losses.append(val_loss.item())

        # Back Propagation
        train_loss.backward()
        optimizer.step()

        # Emptying Cache (GPU)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Displaying Losses
        tr_loss = train_loss.item()
        val_loss = val_loss.item()
        print('Epoch : ', epoch + i + 1, '\n    Train Loss:', tr_loss, '\n    Valid Loss:', val_loss)

        # File saves every 100 iterations
        if i % 100 == 0:
            torch.save({'epochs': epoch + epochs,
                        'model_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'x_train': x_train0,
                        'y_train': y_train0,
                        'x_val': x_val0,
                        'y_val': y_val0,
                        'train_losses': train_losses,
                        'val_losses': val_losses
                        }, 'model.pt')

    return train_losses, val_losses
    
