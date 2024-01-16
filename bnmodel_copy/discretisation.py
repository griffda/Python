import pandas as pd
import numpy as np


def binning_data(data, nbins = None, x_cols=None, y_cols=None):
    """
    Discretise the input and output data.
    Corresponds to steps 2a and 2b in Zac's thesis.

    Parameters
    ----------
    data : pandas dataframe with all the data
    test_size : float
    nbins : int number of bins
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

    # Define default values for input_nbins and output_nbins
    input_nbins = 5
    output_nbins = 5

    # Check if nbins is a list with a dictionary inside
    if isinstance(nbins, list) and len(nbins) > 0:
        nbins_dict = nbins[0]  # Assuming you're interested in the first dictionary in the list
        input_nbins = nbins_dict.get('inputs', 5)
        output_nbins = nbins_dict.get('output', 5)
        print('Using nbins from the list: ', input_nbins, output_nbins)


    # Now you can convert them to integers and use them
    labels_input = list(np.arange(1, int(input_nbins) + 1))
    labels_output = list(np.arange(1, int(output_nbins) + 1))


    # Define empty dictionaries
    bin_edges = {}
    prior_xytrn = {}
    
    # Apply equidistant binning to the input variables
    for col in x.columns:
        # x.loc[:, col], bin_edge_aux = pd.cut(x.loc[:, col], bins=int(input_nbins), labels=labels_input, retbins=True)
        x.loc[:, col], bin_edge_aux = pd.qcut(x.loc[:, col], q=int(input_nbins), labels=labels_input, retbins=True)

        bin_edges[col] = bin_edge_aux
        prior = x.loc[:, col].value_counts(normalize=True).sort_index()
        prior_xytrn[col] = np.array(prior)

    # Apply percentile binning to the output variable
    
    for col in y.columns:
        # Add a small random noise to the data
        # y.loc[:, col] = y.loc[:, col] + np.random.uniform(-0.0001, 0.0001, size=len(y))
        y.loc[:, col], bin_edge_aux = pd.qcut(y.loc[:, col], q=int(output_nbins), labels=labels_output, retbins=True)
        bin_edges[col] = bin_edge_aux
        prior = y[col].value_counts(normalize=True).sort_index()
        prior_xytrn[col] = np.array(prior)
    
    return x, y, bin_edges, prior_xytrn




