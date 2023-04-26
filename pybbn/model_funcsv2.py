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
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle



###Steps 2a (store inoput values), and 2b (store response values) into csv columns.
###This loads csv into a dataframe to be manipulated.
df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/outputv3.csv',
                 index_col=False,
                 usecols=['mass', 'force','acceleration'],
                 encoding=('utf-8')
                 )

###This is getting the input data from the model
x_df = df.iloc[:,[0,1]]
# print(x_df.head())

###This is getting the output data from the model i.e., the target.
y_df = df.iloc[:,[2]]
# print(y_df.head())

###Step 3 is to split data into training and testing sets.
###Here the data is split 50/50
###Target (i.e., y_data are the outputs from the model)
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.5)
# x_train, x_test = train_test_split(df, test_size=0.5)

###Bin labels:
labels = [1,2,3,4,5,6]
# labels = [1,2,3]

number_of_bins = 6

###Empty dicts to fill
bin_edges_dict = {}
prior_dict_xytrn = {}
prior_dict_xytrnLS = {}
# prior_dict_ytrn = {}


###Step 4a
###Replace altl 'df' with 'x_train' to ensure sampling from training data
###Equidistant binning for the inputs
for name in x_train:
    name_bins_xtrn = name + '_bins'
    x_train[name_bins_xtrn], bin_edges = pd.cut(x_train[name], number_of_bins, labels=labels, retbins=True)
    bin_edges_dict[name_bins_xtrn]=bin_edges
    # print(bin_edges_dict.items())

###This is storing the priorPDs so we can plot them
    name_priors = name + '_train_priors'
    prior = x_train[name_bins_xtrn].value_counts(normalize=True).sort_index()
    priorPDs = prior.to_dict()
    prior_dict_xytrn[name_priors] = priorPDs
    # print(prior_dict_xytrn.items())


###Percentile binning for the outputs
###Step 4b
for name in y_train:
    name_bins_ytrn = name + '_bins'
    y_train[name_bins_ytrn], bin_edges = pd.qcut(y_train[name], number_of_bins, labels=labels, retbins=True)
    # y_train[name_bins], bin_edges = pd.cut(y_train[name], 4, labels=labels, retbins=True)
    bin_edges_dict[name_bins_ytrn]=bin_edges
        # for i in range(len(y_train[name]-1)):
    with open('bin_edges_dict_train.pkl', 'wb') as f: pickle.dump(bin_edges_dict, f)


    # print(bin_edges_dict.items())

###This is storing the priorPDs so we can plot them
    name_priors = name + '_train_priors'
    prior = y_train[name_bins_ytrn].value_counts(normalize=True).sort_index()
    priorPDs = prior.to_dict()
    prior_dict_xytrn[name_priors] = priorPDs
    # print(prior_dict_xytrn.items())

###Must join x_train and y_train and pass to the BN:
# df_binned = x_train.drop(['m', 'theta','v0', 'vf', 'KE'], axis=1)
df_binned = x_train.drop(['force', 'mass'], axis=1)
df_binned_y = y_train.drop(['acceleration'], axis=1)
df_binned_xy = pd.concat([df_binned, df_binned_y], axis=1)
# print(df_binned_xy.head())
# print(df_binned_xy.eq(0).any())

###Pybbn only reads data types as strings, so this line converts the data in the csv from int64 to string
df_binned = df_binned.applymap(str)
df_binned_y = df_binned_y.applymap(str)
df_binned_xy = df_binned_xy.applymap(str)

###Step 5
###This is telling us how the network is structured between parent nodes and posteriors.
###Using this method, we avoid having to build our own conditional probability tables using maximum likelihood estimation.
structure = {
    'mass_bins':[],
    'force_bins': [],
    'acceleration_bins': ['force_bins', 'mass_bins']
}

##Step 5
##Here we are calling the Factory function from pybbn and applying the above structure and data frame from csv as arguments.
# bbn = Factory.from_data(structure, df_binned_xy)
# ##Step 5
# ##this line performs inference on the data
# join_tree = InferenceController.apply(bbn)

##Write a function for probability dists:
def prob_dists(structure, data):
    bbn = Factory.from_data(structure, data)
    join_tree = InferenceController.apply(bbn)
    return join_tree

join_tree = prob_dists(structure, df_binned_xy) ###Use the function:
# for node, posteriors in join_tree.get_posteriors().items(): ### this is a list of dictionaries
#     p_no_ev = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
#     print(f'{node} : {p_no_ev}')
#     with open('xy_train_priors.pkl', 'wb') as f: pickle.dump(join_tree.get_posteriors(), f)

##Write a function for adding evidence:
def evidence(nod, bin_index, val):
    ev = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name(nod)) \
    .with_evidence(bin_index, val) \
    .build()
    return ev


"""
<<THESE ARE MARGINALS FROM ABOVE TRAINED NET AT STEP 5: LEARN BAYESIAN NET>>
force_bins : 1=0.10800, 2=0.40400, 3=0.40000, 4=0.08800
acceleration_bins : 1=0.17103, 2=0.23270, 3=0.25005, 4=0.34623
mass_bins : 1=0.07200, 2=0.35600, 3=0.41600, 4=0.15600
"""

###TESTING:
###Step 4a
###Replace all 'df' with 'x_test' to ensure sampling from testing data
###Equidistant binning for the inputs

prior_dict_ytst = {}
prior_dict_xtst = {}
prior_dict_xytst = {}
bin_edges_dict_xtest = {}
bin_edges_dict_ytest = {}

testingData_x = {}

for name in x_test:
    name_bins_xtst = name + '_bins'
    x_test[name_bins_xtst], bin_edges = pd.cut(x_test[name], number_of_bins, labels=labels, retbins=True)
    bin_edges_dict_xtest[name_bins_xtst]=bin_edges


###This is storing the priorPDs so we can plot them
    name_priors = name + '_test_priors'
    prior = x_test[name_bins_xtst].value_counts(normalize=True).sort_index()
    priorPDs = prior.to_dict()
    prior_dict_xytst[name_priors] = priorPDs
    testingData_x[name] = list(priorPDs.values())##this line converts testing data into format required for generate errors function.
    # print(prior_dict_ytst.items()) ###this contains same data that is outputted
    with open('x_test_bins.pkl', 'wb') as f: pickle.dump(bin_edges_dict_xtest, f)
    with open('y_testing_probs2.pkl', 'wb') as f: pickle.dump(testingData_x[name], f)


testingData_y = {}

###Percentile binning for the outputs
###Step 4b
for name in y_test:
    name_bins_ytst = name + '_bins'
    y_test[name_bins_ytst], bin_edges = pd.qcut(y_test[name], number_of_bins, labels=labels, retbins=True)
    bin_edges_dict_ytest[name_bins_ytst]=bin_edges
    print(bin_edges_dict_ytest.items())
    # print(bin_edges_dict_test[name_bins_ytst])
    with open('y_test_bins.pkl', 'wb') as f: pickle.dump(bin_edges_dict_ytest, f)

###This is storing the priorPDs so we can plot them
    name_priors = name + '_test_priors'
    prior = y_test[name_bins_ytst].value_counts(normalize=True).sort_index()
    priorPDs = prior.to_dict()
    prior_dict_xytst[name_priors] = priorPDs
    prior_dict_ytst[name_priors] = priorPDs
    testingData_y[name] = list(priorPDs.values()) ##this line converts testing data into format required for generate errors function.
    # print(testingData_y)
    with open('y_testing_probs.pkl', 'wb') as f: pickle.dump(prior_dict_ytst[name_priors], f)
    with open('y_testing_probs2.pkl', 'wb') as f: pickle.dump(testingData_y[name], f)
print(prior_dict_xytst)

###Must join x_test and y_test and pass to the BN:
###We are testing for outputs and therefore do not want to use any of the training data:
df_test_x = x_test.drop(['force', 'mass'], axis=1)
df_test_y = y_test.drop(['acceleration'], axis=1)
df_test_xy = pd.concat([df_test_x, df_test_y], axis=1)

###Pybbn only reads data types as strings, so this line converts the data in the csv from int64 to string
df_test_x = df_test_x.applymap(str)
df_test_y = df_test_y.applymap(str)
df_test_xy = df_test_xy.applymap(str)
# print(df_test_x.head())

###inserting observation evidence
###This is saying that if there is evidence submitted in the arguments for the function to fill evidenceVars with that evidence
###e.g., evidence = {'deflection':[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'weight':[1.0, 0.0, 0.0, 0.0, 0.0, 0.0] }
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

obs_dicts = generate_multiple_obs_dicts(df_test_xy, 2)


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


# for ev_list in all_ev_list:
#     for ev_dict in ev_list:
#         ev = evidence(ev_dict['nod'], ev_dict['bin_index'], ev_dict['val'])
#         join_tree.set_observation(ev)
#         obs_posteriors, predictedTargetPosteriors = get_obs_and_pred_posteriors(join_tree, "acceleration")

# obs_posteriors_dict = {}
# predicted_posteriors_list = []

# for ev_list in all_ev_list:
#     for ev_dict in ev_list:
#         ev = evidence(ev_dict['nod'], ev_dict['bin_index'], ev_dict['val'])
#         join_tree.set_observation(ev)
#         obs_posteriors, predictedTargetPosteriors = get_obs_and_pred_posteriors(join_tree, "acceleration")
        
#         # Add observation posteriors to dictionary
#         for node_id, posterior in obs_posteriors.items():
#             if node_id not in obs_posteriors_dict:
#                 obs_posteriors_dict[node_id] = []
#             obs_posteriors_dict[node_id].append(posterior)
        
#         # Add predicted target posteriors to list
#         predicted_posteriors_list.append(predictedTargetPosteriors)

# print(obs_posteriors_dict)
# print(predicted_posteriors_list)

def get_posteriors(all_ev_list, join_tree):
    obs_posteriors_dict = {}
    predicted_posteriors_list = []

    for ev_list in all_ev_list:
        for ev_dict in ev_list:
            ev = evidence(ev_dict['nod'], ev_dict['bin_index'], ev_dict['val'])
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