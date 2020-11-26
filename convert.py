#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import sparse

data = pd.read_excel('curated_df.xlsx')

SMILES = data['SMILES']
protein = data['Protein']
Ki = data['Ki']

SMILES_string_length = 100
protein_string_length = 1000
SMILES_n_encodings = 28
protein_n_encodings = 20
N_data = len(SMILES)
Ki = np.array(Ki).reshape((N_data, 1))

# Encoding each atom/bond/chirality/charge as a value which corresponds to its one-hot encoded representation
encoded_data = np.zeros((N_data, SMILES_string_length + protein_string_length))
for i in range(N_data):
    SMILES_i = SMILES[i]
    if len(SMILES_i) != SMILES_string_length:
        print('SMILES length is {}, not {}, at {}'.format(len(SMILES_i), SMILES_string_length, i))
    for j in range(SMILES_string_length):
        char = SMILES_i[j]

        if char == '0':
            pass
        elif char == 'C':
            encoded_data[i, j] = 1
        elif char == 'c':
            encoded_data[i, j] = 2
        elif char == 'S':
            encoded_data[i, j] = 3
        elif char == 's':
            encoded_data[i, j] = 4
        elif char == 'N':
            encoded_data[i, j] = 5
        elif char == 'n':
            encoded_data[i, j] = 6
        elif char == 'P':
            encoded_data[i, j] = 7
        elif char == 'O':
            encoded_data[i, j] = 8
        elif char == 'o':
            encoded_data[i, j] = 9
        elif char == '*': # Silicon
            encoded_data[i, j] = 10
        elif char == 'F':
            encoded_data[i, j] = 11
        elif char == '!': # Chlorine
            encoded_data[i, j] = 12
        elif char == 'B':
            encoded_data[i, j] = 13
        elif char == 'I':
            encoded_data[i, j] = 14
        elif char == 'H':
            encoded_data[i, j] = 15
        elif char == '=':
            encoded_data[i, j] = 16
        elif char == '#':
            encoded_data[i, j] = 17
        elif char == '/':
            encoded_data[i, j] = 18
        elif char == '\\':
            encoded_data[i, j] = 19
        elif char == '@':
            encoded_data[i, j] = 20
        elif char == '.':
            encoded_data[i, j] = 21
        elif char == 'X':
            encoded_data[i, j] = 22
        elif char == '(':
            encoded_data[i, j] = 23
        elif char == ')':
            encoded_data[i, j] = 24
        elif char == '[':
            encoded_data[i, j] = 25
        elif char == ']':
            encoded_data[i, j] = 26
        elif char == '+':
            encoded_data[i, j] = 27
        elif char == '-':
            encoded_data[i, j] = 28
        else:
            print(char)
            print(i) 

# Same as above but for AA sequence
for i in range(N_data):
    protein_i = protein[i]
    if len(protein_i) != protein_string_length:
        print('AA length is {}, not {}, at {}'.format(len(protein_i), protein_string_length, i))
    for j in range(protein_string_length):
        char = protein_i[j]

        if char == '0':
            pass
        elif char == 'A':
            encoded_data[i, SMILES_string_length + j] = 1
        elif char == 'C':
            encoded_data[i, SMILES_string_length + j] = 2
        elif char == 'D':
            encoded_data[i, SMILES_string_length + j] = 3
        elif char == 'E':
            encoded_data[i, SMILES_string_length + j] = 4
        elif char == 'F':
            encoded_data[i, SMILES_string_length + j] = 5
        elif char == 'G':
            encoded_data[i, SMILES_string_length + j] = 6
        elif char == 'H':
            encoded_data[i, SMILES_string_length + j] = 7
        elif char == 'I':
            encoded_data[i, SMILES_string_length + j] = 8
        elif char == 'K':
            encoded_data[i, SMILES_string_length + j] = 9
        elif char == 'L':
            encoded_data[i, SMILES_string_length + j] = 10
        elif char == 'M':
            encoded_data[i, SMILES_string_length + j] = 11
        elif char == 'N':
            encoded_data[i, SMILES_string_length + j] = 12
        elif char == 'P':
            encoded_data[i, SMILES_string_length + j] = 13
        elif char == 'Q':
            encoded_data[i, SMILES_string_length + j] = 14
        elif char == 'R':
            encoded_data[i, SMILES_string_length + j] = 15
        elif char == 'S':
            encoded_data[i, SMILES_string_length + j] = 16
        elif char == 'T':
            encoded_data[i, SMILES_string_length + j] = 17
        elif char == 'V':
            encoded_data[i, SMILES_string_length + j] = 18
        elif char == 'W':
            encoded_data[i, SMILES_string_length + j] = 19
        elif char == 'Y':
            encoded_data[i, SMILES_string_length + j] = 20
        else:
            print(char)

np.save('x_data.npy', encoded_data)
np.save('y_data.npy', Ki)
