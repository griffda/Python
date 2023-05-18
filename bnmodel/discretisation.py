import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


def binning_data(data, test_size=0.2, x_cols=None, y_cols=None):
    """
    Discretise the input and output data.
    Corresponds to steps 2a and 2b in Zac's thesis.

    Parameters
    ----------
    data : pandas dataframe with all the data
    test_size : float
    x_cols : list of str input variables
    y_cols : list of str output variables. Last column of the csv file is used by default.
    TODO: change y_cols to output so it can be a single string (only one possible output)
    
    """
    # Check if x_cols and y_cols are provided, otherwise use default column names
    if x_cols is None and y_cols is None:
        x_cols = data.columns[:-1].tolist()  # Select all columns except the last one as x_cols
        y_cols = [data.columns[-1]]  # Select the last column as y_cols

    if x_cols is None and y_cols is not None:
        aux = data.drop(y_cols, axis=1)
        x_cols = aux.columns[:]

    x = data[x_cols]
    y = data[y_cols]

    labels = [1,2,3,4,5]
    number_of_bins = 5

    # Define empty dictionaries
    bin_edges = {}
    prior_xytrn = {}
    
    # Apply equidistant binning to the input variables
    for col in x.columns:
        x.loc[:, col], bin_edge_aux = pd.cut(x.loc[:, col], number_of_bins, labels=labels, retbins=True)
        bin_edges[col] = bin_edge_aux
        prior = x.loc[:, col].value_counts(normalize=True).sort_index()
        prior_xytrn[col] = np.array(prior)

    # Apply percentile binning to the output variable
    for col in y.columns:
        y.loc[:, col], bin_edge_aux = pd.qcut(y.loc[:, col], number_of_bins, labels=labels, retbins=True)
        bin_edges[col] = bin_edge_aux
        prior = y[col].value_counts(normalize=True).sort_index()
        prior_xytrn[col] = np.array(prior)

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)
    # print(y_test.head())
    
    # Combine the binned data into a single DataFrame for each set
    train_binned = pd.concat([x_train, y_train], axis=1)
    test_binned = pd.concat([x_test, y_test], axis=1)
    

    ###Pybbn only reads data types as strings, so this line converts the data in the csv from int64 to string
    train_binned = train_binned.astype(str)

    return train_binned, test_binned, bin_edges, prior_xytrn