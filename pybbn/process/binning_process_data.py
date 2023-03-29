"""
Created on Wed March 1 17:47 2023

@author: tomgriffiths

In this script we will attempt to create a BN using the data collected from PROCESS SA. 

BBNs structure is updated: 
- Input nodes: 'etath', 'etaiso', 'triang', 'coreradius', 'bt', 'bscfmax', 
- Output nodes: 'coe', 'rmajor', 'capcost'
- Larger data set of 877 rows.
- Data will use preset bin widths and will test how changing this affects the outputs.
- priorPDs will be plotted.
- posterior probabilities will be calculated.
- evidence can be supplied to inform on posterior probabilities.
- evidence, posteriors, and priorPDs will all be super-imposed on each other for same plot. 

- Each step followed in the script will follow flow diagram from Zack Quereb Conti thesis on page 71.  
- Steps: have all been carried out in other scripts. 
    1c (run analytical model i.e., PROCESS) 

"""
import pandas as pd
from pybbn.graph.factory import Factory
from pybbn.pptc.inferencecontroller import InferenceController
from pybbn.graph.jointree import EvidenceBuilder
import numpy as np
import matplotlib.pyplot as plt # for drawing graphs
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt


###Steps 2a (store inoput values), and 2b (store response values) into csv columns. 
###This loads csv into a dataframe to be manipulated. 
df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/PROCESS-/griff_work/HPC/monte_carlo/mc_output_test_uniform/hdf08032023.csv',
                 index_col=False,
                 usecols=['etath', 'etaiso', 'triang', 'coreradius', 'bt', 'bscfmax', 'coe', 'rmajor', 'capcost'],
                 encoding=('utf-8')
                 )

def binning_data(df):
    # Select the columns of interest
    x_cols = ['etath', 'etaiso', 'triang', 'coreradius', 'bt', 'bscfmax']
    y_cols = ['coe']
    x_df = df[x_cols]
    y_df = df[y_cols]

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.5)

    # Define the bin labels and the number of bins
    # labels = [1, 2]
    labels = [1,2,3,4]
    number_of_bins = 4

    # Define empty dictionaries
    bin_edges_dict = {}
    prior_dict_xytrn = {}
    prior_dict_xytst = {}
    bin_edges_dict_test = {}

    prior_dict_xytrnLS = {}

    ### Training set
    ### Apply equidistant binning to the input variables
    for col in x_train.columns:
        col_bins = col + '_bins'
        x_train[col_bins], bin_edges = pd.cut(x_train[col], number_of_bins, labels=labels, retbins=True)
        bin_edges_dict[col_bins] = bin_edges
        prior = x_train[col_bins].value_counts(normalize=True).sort_index()
        prior_dict_xytrn[col + '_train_priors'] = prior.to_dict()

    ### Apply percentile binning to the output variable
    for col in y_train.columns:
        col_bins = col + '_bins'
        y_train[col_bins], bin_edges = pd.qcut(y_train[col], number_of_bins, labels=labels, retbins=True)
        bin_edges_dict[col_bins] = bin_edges
        prior = y_train[col_bins].value_counts(normalize=True).sort_index()
        prior_dict_xytrn[col + '_train_priors'] = prior.to_dict()


    ### Testing set
    ###Equidistant binning for the inputs 

    for col in x_test.columns:
        col_bins_xtst = col + '_bins'
        x_test[col_bins_xtst], bin_edges = pd.cut(x_test[col], number_of_bins, labels=labels, retbins=True)
        bin_edges_dict_test[col_bins_xtst]=bin_edges
        prior = x_test[col_bins_xtst].value_counts(normalize=True).sort_index()
        prior_dict_xytst[col + '_test_priors'] = prior.to_dict()

    # Apply percentile binning to the output variable
    for col in y_test.columns:
        col_bins_ytst = col + '_bins'
        y_test[col_bins_ytst], bin_edges = pd.qcut(y_test[col], number_of_bins, labels=labels, retbins=True)
        bin_edges_dict[col_bins_ytst] = bin_edges
        prior = y_test[col_bins_ytst].value_counts(normalize=True).sort_index()
        prior_dict_xytst[col + '_test_priors'] = prior.to_dict()

    ### Combine the binned data into a single DataFrame
    df_binned = pd.concat([x_train.drop(x_cols, axis=1), y_train.drop(y_cols, axis=1)], axis=1)
    df_test_xy = pd.concat([x_test.drop(x_cols, axis=1), y_test.drop(y_cols, axis=1)], axis=1)
    df_test_x = x_test.drop(x_cols, axis=1)


    return df_binned, df_test_xy, df_test_x


df_test_x, df_binned, df_test_xy = binning_data(df)
print(df_binned.head())

###Step 5
###This is telling us how the network is structured between parent nodes and posteriors. 
###Using this method, we avoid having to build our own conditional probability tables using maximum likelihood estimation. 
# def create_structure(df):
#     columns = df.columns.tolist()
#     structure = {}
#     for column in columns:
#         if column != 'coe_bins':
#             structure[column] = []
#     structure['coe_bins'] = columns[:-1]
#     return structure

# structure = create_structure(df_binned)
# print(structure)

structure = {
    'etath_bins':[],
    'etaiso_bins': [],
    'triang_bins': [],
    'coreradius_bins': [],
    'bt_bins': [],
    'bscfmax_bins': [],
    'coe_bins': ['etath_bins', 'etaiso_bins', 'triang_bins', 'coreradius_bins', 'bt_bins', 'bscfmax_bins']
}

###Step 5
###Here we are calling the Factory function from pybbn and applying the above structure and data frame from csv as arguments. 
bbn = Factory.from_data(structure, df_binned)
###Step 5 
###this line performs inference on the data 
join_tree = InferenceController.apply(bbn)

# ###Write a function for probability dists:
# def prob_dists(structure, data):
#     bbn = Factory.from_data(structure, data)
#     join_tree = InferenceController.apply(bbn)
#     return join_tree

# join_tree = prob_dists(structure, df_binned) ###Use the function:
for node, posteriors in join_tree.get_posteriors().items(): ### this is a list of dictionaries 
    p_no_ev = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
    print(f'{node} : {p_no_ev}')

##Write a function for adding evidence:
def evidence(nod, bin_index, val):
    ev = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name(nod)) \
    .with_evidence(bin_index, val) \
    .build()
    return ev 

###inserting observation evidence
###This is saying that if there is evidence submitted in the arguments for the function to fill evidenceVars with that evidence
###e.g., evidence = {'deflection':[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'weight':[1.0, 0.0, 0.0, 0.0, 0.0, 0.0] }
ev_dict = {}
dataDict = {}


for col in df_test_x: ###col is the column header i.e., nod in the function.
    ev_dict = {'nod':col, 'bin_index':1, 'val': 1.0}
    # ev = evidence(col, '2', 1.0) ###we want to apply hard evidence (100% prob) to the first bin, BUT THIS ISN'T SELECTING THE FIRST BIN, MENTION TO ZACK. 
    ev = evidence(**ev_dict)
    join_tree.set_observation(ev)
#     # print(ev_dict)

for node, posteriors in join_tree.get_posteriors().items(): ### this is a list of dictionaries 
    p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
    print(f'{node} : {p}')

# def create_figure(n_rows, n_cols):
#     """
#     Creates a new matplotlib figure with the specified number of rows and columns.

#     Args:
#         n_rows (int): The number of rows in the figure.
#         n_cols (int): The number of columns in the figure.

#     Returns:
#         matplotlib.figure.Figure: The newly created figure.
#     """
#     return plt.figure(figsize=((200 * n_cols) / 96, (200 * n_rows) / 96), dpi=96, facecolor='white')

# def add_histogram_subplot(fig, row, col, var_name, bin_edges, posteriors):
#     """
#     Adds a new histogram subplot to the specified figure.

#     Args:
#         fig (matplotlib.figure.Figure): The figure to add the histogram subplot to.
#         row (int): The row index of the subplot.
#         col (int): The column index of the subplot.
#         var_name (str): The name of the variable being plotted.
#         bin_edges (numpy.ndarray): The edges of the histogram bins.
#         posteriors (dict): A dictionary of posterior probabilities for each value in the bin.
#     """
#     ax = fig.add_subplot(row, col, col * row - (row - 1) + col) # calculate the position of the subplot
#     ax.set_facecolor("whitesmoke")
    
#     bin_widths = np.diff(bin_edges)
#     x_ticks = bin_edges[:-1] + bin_widths / 2

#     ax.bar(x_ticks, list(posteriors.values()), width=bin_widths, color='blue', alpha=0.2, linewidth=0.2)
    
#     plt.xlim(bin_edges[0], bin_edges[-1])
#     plt.xticks(bin_edges, rotation='vertical')
#     plt.ylim(0, 1)
    
#     ax.grid(color='0.2', linestyle=':', linewidth=0.1, dash_capstyle='round')
#     ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
#     ax.set_title(var_name, fontweight="bold", size=6)
#     ax.set_ylabel('Probabilities', fontsize=7)
#     ax.set_xlabel('Ranges', fontsize=7)

# def plot_posterior_probabilities(structure, bin_edges_dict, join_tree):
#     """
#     Plots the posterior probabilities for a given Bayesian network.

#     Args:
#         structure (dict): A dictionary representing the structure of the Bayesian network.
#         bin_edges_dict (dict): A dictionary of bin edges for each variable in the network.
#         join_tree (pgmpy.inference.JunctionTree): A junction tree representing the Bayesian network.

#     Returns:
#         matplotlib.figure.Figure: The created figure object.
#     """
#     fig = create_figure(1, len(structure.keys()))

#     for i, var_name in enumerate(structure.keys()):
#         bin_edges = bin_edges_dict[var_name]
#         posteriors = join_tree.get_posteriors()[var_name]

#         add_histogram_subplot(fig, 1, len(structure.keys()), var_name, bin_edges, posteriors)

#     fig.suptitle('Posterior Probabilities', fontsize=8)
#     fig.tight_layout()
#     fig.subplots_adjust(top=0.85)

#     return fig
