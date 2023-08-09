import pandas as pd
import numpy as np


def binning_data(data, nbins: int = 5, x_cols=None, y_cols=None):
    """
    Discretise the input and output data.
    Corresponds to steps 2a and 2b in Zac's thesis.

    Parameters
    ----------
    data : pandas dataframe with all the data
    test_size : float
    nbis : int number of bins
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

    labels = np.arange(1, nbins+1)

    # Define empty dictionaries
    bin_edges = {}
    prior_xytrn = {}
    
    # Apply equidistant binning to the input variables
    for col in x.columns:
        x.loc[:, col], bin_edge_aux = pd.cut(x.loc[:, col], nbins, labels=labels, retbins=True)
        bin_edges[col] = bin_edge_aux
        prior = x.loc[:, col].value_counts(normalize=True).sort_index()
        prior_xytrn[col] = np.array(prior)

    # Apply percentile binning to the output variable
    for col in y.columns:
        y.loc[:, col], bin_edge_aux = pd.qcut(y.loc[:, col], nbins, labels=labels, retbins=True)
        bin_edges[col] = bin_edge_aux
        prior = y[col].value_counts(normalize=True).sort_index()
        prior_xytrn[col] = np.array(prior)
    
    return x, y, bin_edges, prior_xytrn