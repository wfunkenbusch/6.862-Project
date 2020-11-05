# COVID

Authors: William Funkenbusch and Zachary Strasser
Date: 11/1/20

Purpose: This project creates a neural network based on the Ki from protein ligand pairs found in the BindingDB database

Data table: "BindingDB_PDSPKi3.tsv" - tsv file with Ki of approx 27k protein-ligand pairs from BindingDB database

Step 1:
"Applied_machine_learning_Datacuration.ipynb" - Open this file and run it. It will curate the protein DB file (formatting SMILES, AA chains) so that a neural network can be run on the information. It will take in "BindingDB_PDSPKi3.tsv" and output the excel file "curated_df.xlsx"

Step 2:
python convert.py - this will take the data in curated_df.xlsx and convert it into numpy files "x_data.npy" and "y_data.npy".

Step 3:
python -u main.py - this will train the model, printing the training and validation error at each epoch as it is run. At the end, plots will be generated showing the training and validation errors over the epochs, and a plot of predicted vs. actual log(Ki) values. These will be saved as "Loss.pdf" and "Accuracy.pdf," respectively.

Appendix:
"Loss.pdf" - figure of loss output from neural network
"curated_df.xlsx" - excel file with curated data to be used by neural network
