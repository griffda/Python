"""
Created on Wed Nov 16 13:58:57 2022

@author: tomgriffiths

In this script we will attepmt to create a BN using the library pyBBN 
for f = ma linear model.

In this script we will attempy to perform cross validation of the BN model: 
- train several models on different subsets of training data
- evaluate them on the complementary subset of testing data. 
- Use cross validation to detect overfitting i.e., failing to generalise a pattern.
- k-fold validation. 

Root Mean Squared Error:
- our output values i.e., y_pred are probability distributions and not hard values. 
- try using posterior probabilities before and after applying evidence. 

"""
import pandas as pd
from pybbn.graph.factory import Factory
from pybbn.pptc.inferencecontroller import InferenceController
from pybbn.graph.jointree import EvidenceBuilder
import numpy as np
import matplotlib.pyplot as plt # for drawing graphs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle



##Steps 2a (store inoput values), and 2b (store response values) into csv columns. 
##This loads csv into a dataframe to be manipulated. 
df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/outputv3.csv',
                 index_col=False,
                 usecols=['mass', 'force','acceleration'],
                 encoding=('utf-8')
                 )

def binning_data(df, n_folds):

    # Select the columns of interest
    x_cols = ['mass', 'force']
    y_cols = ['acceleration']
    x_df = df[x_cols]
    y_df = df[y_cols]

    labels = [1,2,3,4]
    number_of_bins = 4

    # Define empty dictionaries
    bin_edges_dict = {}
    prior_dict_xytrn = {}
    prior_dict_xytst = {}
    bin_edges_dict_test = {}
 
    ### Apply equidistant binning to the input variables
    for col in x_df.columns:
        col_bins = col + '_bins'
        x_df.loc[:, col_bins], bin_edges = pd.cut(x_df.loc[:, col], number_of_bins, labels=labels, retbins=True)
        bin_edges_dict[col_bins] = bin_edges
        prior = x_df.loc[:, col_bins].value_counts(normalize=True).sort_index()
        prior_dict_xytrn[col + '_priors'] = prior.to_dict()


    ### Apply percentile binning to the output variable
    for col in y_df.columns:
        col_bins = col + '_bins'
        y_df.loc[:, col_bins], bin_edges = pd.qcut(y_df.loc[:, col], number_of_bins, labels=labels, retbins=True)
        # y_df.loc[:, col_bins], bin_edges = pd.qcut(y_df.loc[:, col].values, number_of_bins, labels=labels, retbins=True)
        # y_df.loc[:, col_bins] = pd.qcut(y_df.loc[:, col].values, number_of_bins, labels=labels, retbins=True)[0].copy()
        # bin_edges = pd.qcut(y_df.loc[:, col].values, number_of_bins, labels=labels, retbins=True)[1]
        bin_edges_dict[col_bins] = bin_edges
        prior = y_df[col_bins].value_counts(normalize=True).sort_index()
        prior_dict_xytrn[col + '_priors'] = prior.to_dict()


    # Split the data into training and testing sets using k-fold validation
    kfold = KFold(n_splits=n_folds)
    dfs_binned = []
    dfs_test_xy = []
    dfs_test_x = []
    for train_idx, test_idx in kfold.split(x_df):
        x_train, x_test = x_df.iloc[train_idx], x_df.iloc[test_idx]
        y_train, y_test = y_df.iloc[train_idx], y_df.iloc[test_idx]

        # Combine the binned data into a single DataFrame for each fold
        df_binned = pd.concat([x_train.drop(x_cols, axis=1), y_train.drop(y_cols, axis=1)], axis=1)
        df_test_xy = pd.concat([x_test.drop(x_cols, axis=1), y_test.drop(y_cols, axis=1)], axis=1)
        df_test_x = x_test.drop(x_cols, axis=1)

        dfs_binned.append(df_binned)
        dfs_test_xy.append(df_test_xy)
        dfs_test_x.append(df_test_x)

    return dfs_binned, dfs_test_xy, dfs_test_x, bin_edges_dict, prior_dict_xytrn

dfs_test_xy = binning_data(df, 5)

print(dfs_test_xy)