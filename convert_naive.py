#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from scipy import sparse

data = pd.read_excel('curated_df_11_3.xlsx')

SMILES = data['SMILES']
protein = data['Protein']
Ki = data['Ki']

SMILES_string_length = 100
protein_string_length = 1000
SMILES_n_encodings = 28
protein_n_encodings = 20
N_data = len(SMILES)

# Encoding each atom/bond/chirality/charge as a value which corresponds to its one-hot encoded representation
SMILES_encoded = [''] * N_data
for i in range(N_data):
    SMILES_i = SMILES[i]
    if len(SMILES_i) != SMILES_string_length:
        print('SMILES length is {}, not {}, at {}'.format(len(SMILES_i), SMILES_string_length, i))
    for j in range(SMILES_string_length):
        char = SMILES_i[j]

        if char == '0':
            SMILES_encoded[i] += '0 '
        elif char == 'C':
            SMILES_encoded[i] += '1 '
        elif char == 'c':
            SMILES_encoded[i] += '2 '
        elif char == 'S':
            SMILES_encoded[i] += '3 '
        elif char == 's':
            SMILES_encoded[i] += '4 '
        elif char == 'N':
            SMILES_encoded[i] += '5 '
        elif char == 'n':
            SMILES_encoded[i] += '6 '
        elif char == 'P':
            SMILES_encoded[i] += '7 '
        elif char == 'O':
            SMILES_encoded[i] += '8 '
        elif char == 'o':
            SMILES_encoded[i] += '9 '
        elif char == 'SILICON':
            SMILES_encoded[i] += '10 '
        elif char == 'F':
            SMILES_encoded[i] += '11 '
        elif char == 'K':
            SMILES_encoded[i] += '12 '
        elif char == 'B':
            SMILES_encoded[i] += '13 '
        elif char == 'I':
            SMILES_encoded[i] += '14 '
        elif char == 'H':
            SMILES_encoded[i] += '15 '
        elif char == '=':
            SMILES_encoded[i] += '16 '
        elif char == '#':
            SMILES_encoded[i] += '17 '
        elif char == '/':
            SMILES_encoded[i] += '18 '
        elif char == '\\':
            SMILES_encoded[i] += '19 '
        elif char == '@':
            SMILES_encoded[i] += '20 '
        elif char == '.':
            SMILES_encoded[i] += '21 '
        elif char == 'X':
            SMILES_encoded[i] += '22 '
        elif char == '(':
            SMILES_encoded[i] += '23 '
        elif char == ')':
            SMILES_encoded[i] += '24 '
        elif char == '[':
            SMILES_encoded[i] += '25 '
        elif char == ']':
            SMILES_encoded[i] += '26 '
        elif char == '+':
            SMILES_encoded[i] += '27 '
        elif char == '-':
            SMILES_encoded[i] += '28 '
        else:
            print(char)  

# Same as above but for AA sequence
protein_encoded = [''] * N_data
for i in range(N_data):
    protein_i = protein[i]
    if len(protein_i) != protein_string_length:
        print('AA length is {}, not {}, at {}'.format(len(protein_i), protein_string_length, i))
    for j in range(protein_string_length):
        char = protein_i[j]

        if char == 'A':
            protein_encoded[i] += '1 '
        elif char == 'C':
            protein_encoded[i] += '2 '
        elif char == 'D':
            protein_encoded[i] += '3 '
        elif char == 'E':
            protein_encoded[i] += '4 '
        elif char == 'F':
            protein_encoded[i] += '5 '
        elif char == 'G':
            protein_encoded[i] += '6 '
        elif char == 'H':
            protein_encoded[i] += '7 '
        elif char == 'I':
            protein_encoded[i] += '8 '
        elif char == 'K':
            protein_encoded[i] += '9 '
        elif char == 'L':
            protein_encoded[i] += '10 '
        elif char == 'M':
            protein_encoded[i] += '11 '
        elif char == 'N':
            protein_encoded[i] += '12 '
        elif char == 'P':
            protein_encoded[i] += '13 '
        elif char == 'Q':
            protein_encoded[i] += '14 '
        elif char == 'R':
            protein_encoded[i] += '15 '
        elif char == 'S':
            protein_encoded[i] += '16 '
        elif char == 'T':
            protein_encoded[i] += '17 '
        elif char == 'V':
            protein_encoded[i] += '18 '
        elif char == 'W':
            protein_encoded[i] += '19 '
        elif char == 'Y':
            protein_encoded[i] += '20 '
        else:
            print(char)

# Converting these strings into format used by NN
#input_array = sparse.coo_matrix(([],([],[])), shape = (N_data, 1, SMILES_string_length + protein_string_length, SMILES_n_encodings + protein_n_encodings))
input_array = np.zeros((N_data, 1, SMILES_string_length + protein_string_length, SMILES_n_encodings + protein_n_encodings))
for i in range(len(SMILES_encoded)):
    split_SMILES = SMILES_encoded[i].split()
    
    if len(split_SMILES) != SMILES_string_length:
        print('New SMILES length is {}, not {}, at {}'.format(len(split_SMILES), SMILES_string_length, i))
    for j in range(len(split_SMILES)):
        input_array[i, 0, j, int(split_SMILES[j]) - 1] = 1

for i in range(len(protein_encoded)):
    split_protein = protein_encoded[i].split()

    if len(split_protein) != protein_string_length:
        print('New protein length is {}, not {}, at {}'.format(len(split_protein), protein_string_length, i))
    for j in range(len(split_protein)):
        input_array[i, 0, SMILES_string_length + j, SMILES_n_encodings + int(split_protein[j]) - 1] = 1

np.save('data.npy', input_array)

data = np.load('data.npy')

print((data == input_array).all())