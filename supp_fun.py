#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def one_hot_encode(x_data, n_encodings):
    # Turns 2D x data encoding location of each one-hot into 4D array for input into CNN. Array will be of size shape, which should be (N_data, 1, sequence_length, n_encodings)

    # Inputs:
    #     x_data - (N_data, sequence_length) array of one hot encodes. For example, if x_data[3, 5] = 7, then output[3, 1, 5, 7] = 1 and output[3, 1, 5, != 7] = 0
    #     n_encodings - number of one hot encodings
    output = np.zeros((x_data.shape[0], 1, x_data.shape[1], n_encodings))

    for i in range(x_data.shape[0]):
        for j in range(x_data.shape[1]):
            output[i, 0, j, int(x_data[i, j])] = 1

    return output