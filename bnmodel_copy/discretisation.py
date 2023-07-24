import pandas as pd
import numpy as np

class DataBinner:
    def __init__(self, nbins=5, x_cols=None, y_cols=None):
        self.nbins = nbins
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.bin_edges = {}
        self.prior_xytrn = {}

    def binning_data(self, data):
        """
        Discretise the input and output data.
        Corresponds to steps 2a and 2b in Zac's thesis.

        Parameters
        ----------
        data : pandas dataframe with all the data
        """
        # Check if x_cols and y_cols are provided, otherwise use default column names
        if self.x_cols is None and self.y_cols is None:
            self.x_cols = data.columns[:-1].tolist()  # Select all columns except the last one as x_cols
            self.y_cols = [data.columns[-1]]  # Select the last column as y_cols

        if self.x_cols is None and self.y_cols is not None:
            aux = data.drop(self.y_cols, axis=1)
            self.x_cols = aux.columns[:]

        x = data[self.x_cols]
        y = data[self.y_cols]

        labels = np.arange(1, self.nbins + 1)

        # Apply equidistant binning to the input variables
        for col in x.columns:
            x.loc[:, col], bin_edge_aux = pd.cut(x.loc[:, col], self.nbins, labels=labels, retbins=True)
            self.bin_edges[col] = bin_edge_aux
            prior = x.loc[:, col].value_counts(normalize=True).sort_index()
            self.prior_xytrn[col] = np.array(prior)

        # Apply percentile binning to the output variable
        for col in y.columns:
            y.loc[:, col], bin_edge_aux = pd.qcut(y.loc[:, col], self.nbins, labels=labels, retbins=True)
            self.bin_edges[col] = bin_edge_aux
            prior = y[col].value_counts(normalize=True).sort_index()
            self.prior_xytrn[col] = np.array(prior)

        return x, y, self.bin_edges, self.prior_xytrn
