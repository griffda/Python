"""
Created on Wed Nov 16 13:58:57 2022

@author: tomgriffiths

In this script we will attepmt to create a BN using the library pyBBN
for f = ma linear model.

In this script we will attempy to perform cross validation of the BN model:
- train several models on different subsets of training data
- evaluate them on the complementary subset of testing data.

This re-runs the model every time the run button is used and therefore creates diffeent output
probability distributions.

"""
import pandas as pd
from pybbn.graph.factory import Factory
from pybbn.pptc.inferencecontroller import InferenceController
from pybbn.graph.jointree import EvidenceBuilder
import numpy as np
import matplotlib.pyplot as plt # for drawing graphs
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import pickle

###Steps 2a (store inoput values), and 2b (store response values) into csv columns.
###This loads csv into a dataframe to be manipulated.
df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/outputv3.csv',
                 index_col=False,
                 usecols=['mass', 'force','acceleration'],
                 encoding=('utf-8')
                 )

def binning_data(df, test_size=0.2):
    # Select the columns of interest
    x_cols = ['mass', 'force']
    y_cols = ['acceleration']
    x_df = df[x_cols]
    y_df = df[y_cols]

    labels = [1,2,3,4,5,6]
    number_of_bins = 6

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
        bin_edges_dict[col_bins] = bin_edges
        prior = y_df[col_bins].value_counts(normalize=True).sort_index()
        prior_dict_xytrn[col + '_priors'] = prior.to_dict()

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=test_size, random_state=42)

    # Combine the binned data into a single DataFrame for each set
    df_train_binned = pd.concat([x_train.drop(x_cols, axis=1), y_train.drop(y_cols, axis=1)], axis=1)
    df_test_binned = pd.concat([x_test.drop(x_cols, axis=1), y_test.drop(y_cols, axis=1)], axis=1)
    df_test_x = x_test.drop(x_cols, axis=1)
    # print("df_train_binned", df_train_binned)
    # print("df_test_binned", df_test_binned)

    return df_train_binned, df_test_binned, df_test_x, bin_edges_dict, prior_dict_xytrn

df_train_binned, df_test_binned, df_test_x, bin_edges_dict, prior_dict_xytrn = binning_data(df, 0.2)

###Step 5
###This is telling us how the network is structured between parent nodes and posteriors.
###Using this method, we avoid having to build our own conditional probability tables using maximum likelihood estimation.
structure = {
    'mass_bins':[],
    'force_bins': [],
    'acceleration_bins': ['force_bins', 'mass_bins']
}

##Write a function for probability dists:
def prob_dists(structure, data):
    bbn = Factory.from_data(structure, data)
    join_tree = InferenceController.apply(bbn)
    return join_tree

# join_tree = prob_dists(structure, df_binned_xy) ###Use the function:
join_tree = prob_dists(structure, df_train_binned) ###Use the function:

##Write a function for adding evidence:
def evidence(nod, bin_index, val):
    ev = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name(nod)) \
    .with_evidence(bin_index, val) \
    .build()
    return ev

ev_dict = {}
dataDict = {}

def generate_obs_dict(test_df):
    # choose a random row from the test_df
    row = test_df.sample()
    print("Selected row index:", row.index[0])

    # generate an obs_dict from the chosen row
    obs_dict = {}
    for col in test_df.columns:
        bin_index = str(row[col].values[0])
        obs_dict[col] = {'bin_index': bin_index, 'val': 1.0}

    print("Observation dictionary:", obs_dict)
    return obs_dict

def generate_multiple_obs_dicts(test_df, num_samples):
    obs_dicts = []
    for i in range(num_samples):
        obs_dict = generate_obs_dict(test_df)
        obs_dicts.append(obs_dict)
    print("Observation dictionaries:", obs_dicts)
    return obs_dicts

obs_dicts = generate_multiple_obs_dicts(df_test_binned, 2)

def set_multiple_observations(df, obs_dicts, default_bin=5):
    all_ev_list = []
    for obs_dict in obs_dicts:
        ev_list = []
        for col in df.columns:
            if col in obs_dict:
                bin_index = obs_dict[col]['bin_index']
                val = obs_dict[col]['val']
            else:
                bin_index = str(default_bin)
                val = 1.0
            ev_dict = {'nod':col, 'bin_index':bin_index, 'val': val}
            ev_list.append(ev_dict)
        all_ev_list.append(ev_list)
    print("All evidence lists:", all_ev_list)
    return all_ev_list


# set_observations(df_test_x, obs_dict, default_bin=5)

all_ev_list = set_multiple_observations(df_test_x, obs_dicts)
print(all_ev_list)

def get_obs_and_pred_posteriors(join_tree, target_variable):
    obs_posteriors = {}

    for node, posteriors in join_tree.get_posteriors().items():
        obs = node[:-5]  # remove the "_bins" suffix from the node name to get the observation name
        obs_posteriors[obs] = [posteriors[val] for val in sorted(posteriors)]  # sort the posteriors by value and add them to the dictionary

    print("Observation posteriors:", obs_posteriors)

    predictedTargetPosteriors = []

    for node, posteriors in join_tree.get_posteriors().items():
        obs = node[:-5]  # remove the "_bins" suffix from the node name to get the observation name
        if obs == target_variable:  # check if the observation corresponds to the specified target variable
            predictedTargetPosteriors.append([posteriors[val] for val in sorted(posteriors)])  # sort the posteriors by value and add them to the list

    print("Predicted target posteriors:", predictedTargetPosteriors)

    return obs_posteriors, predictedTargetPosteriors

def get_posteriors(all_ev_list, join_tree):
    obs_posteriors_dict = {}
    predicted_posteriors_list = []

    for ev_list in all_ev_list:
        for ev_dict in ev_list:
            print(ev_dict)
            ev = evidence(ev_dict['nod'], int(ev_dict['bin_index']), ev_dict['val'])
            join_tree.set_observation(ev)
            obs_posteriors, predictedTargetPosteriors = get_obs_and_pred_posteriors(join_tree, "acceleration")

            # Add observation posteriors to dictionary
            for node_id, posterior in obs_posteriors.items():
                if node_id not in obs_posteriors_dict:
                    obs_posteriors_dict[node_id] = []
                obs_posteriors_dict[node_id].append(posterior)

            # Add predicted target posteriors to list
            predicted_posteriors_list.append(predictedTargetPosteriors)
    
    print("Observation posteriors:", obs_posteriors_dict)
    print("Predicted target posteriors:", predicted_posteriors_list)

    return obs_posteriors_dict, predicted_posteriors_list

get_posteriors(all_ev_list, join_tree)

# # ##This is for the figure parameters.
# n_rows = 1
# n_cols = len(structure.keys()) ##the length of the BN i.e., five nodes

# def create_figure(n_rows, n_cols):
#     """
#     Create a figure with specified number of rows and columns.
#     """
#     fig = plt.figure(figsize=((200 * n_cols) / 96, (200 * n_rows) / 96), dpi=96, facecolor='white')
#     fig.suptitle('Posterior Probabilities', fontsize=8)
#     return fig

# def create_subplot(fig, n_rows, n_cols, count, varName, bin_edges_dict, priors_dict, obs_posteriors):
#     """
#     Create a subplot with specified parameters.
#     """
#     ax = fig.add_subplot(n_rows, n_cols, count+1)
#     ax.set_facecolor("whitesmoke")

#     index = bin_edges_dict[varName]
#     edge = np.zeros((len(bin_edges_dict.items()), len(index[:])))
#     binwidths = np.zeros((len(bin_edges_dict.items()), len(index[:-1])))
#     xticksv = np.zeros((len(bin_edges_dict.items()), len(index[:-1])))

#     for i in range(len(index)):
#         edge[count, i] = index[i]

#     for i in range(len(index)-1):
#         binwidths[count, i] = (index[i+1] - index[i])
#         xticksv[count,i]  = ((index[i+1] - index[i]) / 2.) + index[i]

#     dataDict = {}

#     for node, posteriors in obs_posteriors.items():
#         varName2 = varName[:-5]
#         if varName2 == node:
#             dataDict[node] = posteriors
#             if node == 'acceleration':
#                 ax.bar(xticksv[count], dataDict[node], align='center', width=binwidths[count], color='red', alpha=0.2, linewidth=0.2)
#             elif node == 'mass' or node == 'force':
#                 ax.bar(xticksv[count], dataDict[node], align='center', width=binwidths[count], color='green', alpha=0.2, linewidth=0.2)

#     priorPDs_dict = {}

#     for var2, idx in priors_dict.items():
#         varName3 = var2[:-12]
#         if varName3 == node:
#             priorPDs_dict[var2] = list(idx.values())
#             ax.bar(xticksv[count], priorPDs_dict[var2], align='center', width=binwidths[count], color='black', alpha=0.2, linewidth=0.2)


#     plt.xlim(min(edge[count]), max(edge[count]))
#     plt.xticks([np.round(e, 2) for e in edge[count]], rotation='vertical')
#     plt.ylim(0, 1)

#     ax.grid(color='0.2', linestyle=':', linewidth=0.1, dash_capstyle='round')
#     ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
#     ax.set_title(varName2, fontweight="bold", size=6)
#     ax.set_ylabel('Probabilities', fontsize=7)
#     ax.set_xlabel('Ranges', fontsize=7)
#     return fig

# def create_all_subplots(fig, n_rows, n_cols, bin_edges_dict, priors_dict, obs_posteriors):
#     """
#     Create all subplots for the given figure.
#     """
#     count = 0
#     for varName in bin_edges_dict:
#         fig = create_subplot(fig, n_rows, n_cols, count, varName, bin_edges_dict, priors_dict, obs_posteriors)
#         count += 1
#     return fig

# def plot_posterior_probabilities(n_rows, n_cols, bin_edges_dict, priors_dict, obs_posteriors, plot):
#     """
#     Plot posterior probabilities for each variable in bin_edges_dict.
#     """
#     fig = create_figure(n_rows, n_cols)
#     fig = create_all_subplots(fig, n_rows, n_cols, bin_edges_dict, priors_dict, obs_posteriors)
#     fig.tight_layout()
#     fig.subplots_adjust(top=0.85)
#     if plot == True:
#         plt.show()

# plot_posterior_probabilities(n_rows, n_cols, bin_edges_dict, prior_dict_xytst, obs_posteriors, plot=True)

# def run_plot_posterior_probabilities(n_runs, n_rows, n_cols, bin_edges_dict, prior_dict, join_tree):
#     posteriors_list = []
#     for i in range(n_runs):
#         obs_posteriors = get_obs_and_pred_posteriors(join_tree, target_variable='acceleration')
#         posterior = [obs_posteriors['acceleration'][i] for i in range(len(obs_posteriors['acceleration']))]
#         posteriors_list.append(posterior)
#         plot_posterior_probabilities(n_rows, n_cols, bin_edges_dict, prior_dict, obs_posteriors, plot=True)
#     return posteriors_list



# posteriors = run_plot_posterior_probabilities(2, n_rows, n_cols, bin_edges_dict, prior_dict_xytst, join_tree)


# all_posteriors = run_plot_posterior_probabilities2(5, n_rows, n_cols, bin_edges_dict, prior_dict_xytst)