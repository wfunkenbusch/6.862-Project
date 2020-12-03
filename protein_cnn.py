import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold

import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, BatchNorm2d, Dropout, Conv1d, MaxPool1d, BatchNorm1d
from torch.optim import Adam

from supp_fun import one_hot_encode

class Net(Module):   
    def __init__(self, n_SMILES_filters = 128, n_protein_filters = 128):
        super(Net, self).__init__()

        self.cnn_SMILES = Sequential(
            # [N_data, 1, sequence_length, n_SMILES_encodings]
            Conv2d(1, n_SMILES_filters, kernel_size = (11, 28), stride = 1, padding = (5, 0)),
            BatchNorm2d(n_SMILES_filters),
            ReLU(inplace = True),
            Permute(),
            # [N_data, 1, sequence_length, 128]

            Conv2d(1, n_SMILES_filters, kernel_size = (7, n_SMILES_filters), stride = 1, padding = (3, 0)),
            BatchNorm2d(n_SMILES_filters),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = (3, 1), stride = (3, 1)),
            Permute()
            # [N_data, 1, sequence_length/3, 128]
        )

        self.cnn_Protein = Sequential(
            # [N_data, 1, sequence_length, n_Protein_encodings]
            Conv2d(1, n_protein_filters, kernel_size = (11, 20), stride = 1, padding = (5, 0)),
            BatchNorm2d(n_protein_filters),
            ReLU(inplace = True),
            Permute(),
            # [N_data, 1, sequence_length, 128]

            Conv2d(1, n_protein_filters, kernel_size = (7, n_protein_filters), stride = 1, padding = (3, 0)),
            BatchNorm2d(n_protein_filters),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = (3, 1), stride = (3, 1)),
            Permute()
            # [N_data, 1, sequence_length/3, 128]
        )

        self.cnn_layers = Sequential(
            # Standard 2D Convolutional Layer
            # [N_data, 1, sequence_length, 256]
            Conv2d(1, n_SMILES_filters + n_protein_filters, kernel_size = (7, n_SMILES_filters + n_protein_filters), stride = 1, padding = (3, 0)),
            BatchNorm2d(n_SMILES_filters + n_protein_filters),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = (3, 1), stride = (3, 1)),
            Permute(),
            # [N_data, 1, sequence_length/9, 256]

            Conv2d(1, n_SMILES_filters + n_protein_filters, kernel_size = (7, n_SMILES_filters + n_protein_filters), stride = 1, padding = (3, 0)),
            BatchNorm2d(n_SMILES_filters + n_protein_filters),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = (3, 1), stride = (3, 1)),
            Permute(),
            # [N_data, 1, sequence_length/27, 256]

            Conv2d(1, n_SMILES_filters + n_protein_filters, kernel_size = (3, n_SMILES_filters + n_protein_filters), stride = 1, padding = (1, 0)),
            BatchNorm2d(n_SMILES_filters + n_protein_filters),
            ReLU(inplace = True),
            Permute(),

            Conv2d(1, n_SMILES_filters + n_protein_filters, kernel_size = (3, n_SMILES_filters + n_protein_filters), stride = 1, padding = (1, 0)),
            BatchNorm2d(n_SMILES_filters + n_protein_filters),
            ReLU(inplace = True),
            Permute(),

            Conv2d(1, n_SMILES_filters + n_protein_filters, kernel_size = (3, n_SMILES_filters + n_protein_filters), stride = 1, padding = (1, 0)),
            BatchNorm2d(n_SMILES_filters + n_protein_filters),
            ReLU(inplace = True),
            Permute(),

            Conv2d(1, n_SMILES_filters + n_protein_filters, kernel_size = (3, n_SMILES_filters + n_protein_filters), stride = 1, padding = (1, 0)),
            BatchNorm2d(n_SMILES_filters + n_protein_filters),
            ReLU(inplace = True),
            Permute(),

            Conv2d(1, n_SMILES_filters + n_protein_filters, kernel_size = (3, n_SMILES_filters + n_protein_filters), stride = 1, padding = (1, 0)),
            BatchNorm2d(n_SMILES_filters + n_protein_filters),
            ReLU(inplace = True),
            Permute(),

            Conv2d(1, n_SMILES_filters + n_protein_filters, kernel_size = (3, n_SMILES_filters + n_protein_filters), stride = 1, padding = (1, 0)),
            BatchNorm2d(n_SMILES_filters + n_protein_filters),
            ReLU(inplace = True),
            MaxPool2d(kernel_size = (3, 1), stride = (3, 1))
            # [N_data, 256, sequence_length/51, 1]
        )

        # Combined, Fully-Connected Layers
        self.linear_layers = Sequential(
            Linear(int((n_SMILES_filters + n_protein_filters) * 13), 2048),
            Dropout(0.5),
            Linear(2048, 2048),
            Dropout(0.5),
            Linear(2048, 2048),
            Dropout(0.5),
            Linear(2048, 1)
        )

    # Defining the forward pass    
    def forward(self, x):
        x_SMILES = x[:, :, :, 0:28]
        x_Protein = x[:, :, :, 28:]
        x_SMILES = self.cnn_SMILES(x_SMILES)
        x_Protein = self.cnn_Protein(x_Protein)
        x = torch.cat((x_SMILES, x_Protein), dim = 3)
        x = self.cnn_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

class Permute(torch.nn.Module):
    # Class for converting filters into features within Sequential
    def forward(self, x):
        return x.permute(0, 3, 2, 1)

def train_model(model, optimizer, loss, x_train0, y_train0, x_val0, y_val0, epoch, epochs, suppress = False):
    # Enable Training Mode
    model.train()

    # Featurization of Data
    x_train = one_hot_encode(x_train0, 48)
    x_val = one_hot_encode(x_val0, 48)
    y_train = np.log10(y_train0)
    y_val = np.log10(y_val0)

    # Converting to PyTorch Format
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).float()
    x_val = torch.from_numpy(x_val).float()
    y_val = torch.from_numpy(y_val).float()

    x_train, y_train = Variable(x_train), Variable(y_train)
    x_val, y_val = Variable(x_val), Variable(y_val)

    train_losses = []
    val_losses = []

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
        if not suppress:
            print('Iteration : ', epoch + i + 1, '\n    Train Loss:', tr_loss, '\n    Valid Loss:', val_loss)

        '''
        # Saving Model
        if i % 100 == 0:
            torch.save({'epochs': epoch + epochs,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'x_train': x_train0,
                        'y_train': y_train0,
                        'x_val': x_val0,
                        'y_val': y_val0,
                        'train_losses': train_losses,
                        'val_losses': val_losses
                        }, 'model.pt')
        '''

    return train_losses, val_losses
    
def cross_val_model(lr, loss, x_train, y_train, params):
    kfold = KFold(5, shuffle = True)

    best_params = [0, 0]
    train_loss, val_loss = (100, 100)

    for n_SMILES_filters in params:
        for n_protein_filters in params:
            train_losses_n = 0
            val_losses_n = 0
            for train, test in kfold.split(y_train):
                x_train_split = x_train[train, :]
                y_train_split = y_train[train]
                x_val_split = x_train[test, :]
                y_val_split = y_train[test]

                model = Net(n_SMILES_filters, n_protein_filters)
                optimizer = Adam(model.parameters(), lr = lr)

                train_losses_split, val_losses_split = train_model(model, optimizer, loss, x_train_split, y_train_split, x_val_split, y_val_split, 0, 600, suppress = True)
                train_losses_n += train_losses_split[-1]
                val_losses_n += val_losses_split[-1]

            train_losses_n = train_losses_n/5
            val_losses_n = val_losses_n/5

            if val_losses_n < val_loss:
                val_loss = val_losses_n
                train_loss = train_losses_n
                best_params = [n_SMILES_filters, n_protein_filters]

            print('SMILES Filters: {}'.format(n_SMILES_filters))
            print('Protein Filters: {}'.format(n_protein_filters))
            print('    Training Loss: {}'.format(train_losses_n))
            print('    Validation Loss: {}'.format(val_losses_n))

    print('Best SMILES Filters: {}'.format(best_params[0]))
    print('Best Protein Filters: {}'.format(best_params[1]))
    print('    Best Training Loss: {}'.format(train_loss))
    print('    Best Validation Loss: {}'.format(val_loss))

    return best_params, train_loss, val_loss

            

            


