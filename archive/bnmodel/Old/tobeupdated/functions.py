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
from sklearn.metrics import mean_squared_error
from math import sqrt



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
labels = [1,2,3,4]
# labels = [1,2,3]

number_of_bins = 4

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

    name_priorsLS = name + '_train_priorsLS'
    alpha = 1 # smoothing factor
    priorLS = x_train[name_bins_xtrn].value_counts().sort_index()
    print(priorLS)
    priorPDsLS = (priorLS + alpha) / (len(x_train) + alpha*number_of_bins)
    prior_dict_xytrnLS[name_priorsLS] = priorPDsLS.to_dict()
    extreme_values_mask = (priorLS == 0) | (priorLS == 1)
    priorLS[extreme_values_mask] = priorPDsLS[extreme_values_mask]
    print(prior_dict_xytrnLS.items())


    

      
###Percentile binning for the outputs
###Step 4b
for name in y_train:
    name_bins_ytrn = name + '_bins'
    y_train[name_bins_ytrn], bin_edges = pd.qcut(y_train[name], number_of_bins, labels=labels, retbins=True)
    # y_train[name_bins], bin_edges = pd.cut(y_train[name], 4, labels=labels, retbins=True)
    bin_edges_dict[name_bins_ytrn]=bin_edges
        # for i in range(len(y_train[name]-1)):
    # print(bin_edges_dict.items())   

###This is storing the priorPDs so we can plot them
    name_priors = name + '_train_priors'
    prior = y_train[name_bins_ytrn].value_counts(normalize=True).sort_index()
    priorPDs = prior.to_dict()
    prior_dict_xytrn[name_priors] = priorPDs
    # print(prior_dict_xytrn.items())

    # alpha = 1 # smoothing factor
    # prior = y_train[name_bins_ytrn].value_counts().sort_index()
    # priorPDs = (prior + alpha) / (len(y_train) + alpha*number_of_bins)
    # prior_dict_xytrn[name_priors] = priorPDs.to_dict()

###Must join x_train and y_train and pass to the BN:
# df_binned = x_train.drop(['m', 'theta','v0', 'vf', 'KE'], axis=1)
df_binned = x_train.drop(['force', 'mass'], axis=1)
df_binned_y = y_train.drop(['acceleration'], axis=1)
df_binned_xy = pd.concat([df_binned, df_binned_y], axis=1)
print(df_binned_xy.head())
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

###Step 5
###Here we are calling the Factory function from pybbn and applying the above structure and data frame from csv as arguments. 
# bbn = Factory.from_data(structure, df_binned_xy)
###Step 5 
###this line performs inference on the data 
# join_tree = InferenceController.apply(bbn)

###Write a function for probability dists:
# def prob_dists(structure, data):
#     bbn = Factory.from_data(structure, data)
#     join_tree = InferenceController.apply(bbn)
#     return join_tree

# join_tree = prob_dists(structure, df_binned_xy) ###Use the function:
# for node, posteriors in join_tree.get_posteriors().items(): ### this is a list of dictionaries 
#     p_no_ev = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
#     print(f'{node} : {p_no_ev}')

# ##Write a function for adding evidence:
# def evidence(nod, bin_index, val):
#     ev = EvidenceBuilder() \
#     .with_node(join_tree.get_bbn_node_by_name(nod)) \
#     .with_evidence(bin_index, val) \
#     .build()
#     return ev 


# """
# <<THESE ARE MARGINALS FROM ABOVE TRAINED NET AT STEP 5: LEARN BAYESIAN NET>>
# force_bins : 1=0.10800, 2=0.40400, 3=0.40000, 4=0.08800
# acceleration_bins : 1=0.17103, 2=0.23270, 3=0.25005, 4=0.34623
# mass_bins : 1=0.07200, 2=0.35600, 3=0.41600, 4=0.15600
# """

# ###TESTING: 
# ###Step 4a 
# ###Replace all 'df' with 'x_test' to ensure sampling from testing data
# ###Equidistant binning for the inputs 

# prior_dict_ytst = {}
# bin_edges_dict_test = {}

# for name in x_test:
#     name_bins_xtst = name + '_bins'
#     x_test[name_bins_xtst], bin_edges = pd.cut(x_test[name], number_of_bins, labels=labels, retbins=True)
#     bin_edges_dict_test[name_bins_xtst]=bin_edges
    
    
# ###This is storing the priorPDs so we can plot them
#     name_priors = name + '_test_priors'
#     prior = x_test[name_bins_xtst].value_counts(normalize=True).sort_index()
#     priorPDs = prior.to_dict()
#     prior_dict_ytst[name_priors] = priorPDs
#     # print(prior_dict_ytst.items()) ###this contains same data that is outputted 


# ###Percentile binning for the outputs
# ###Step 4b
# for name in y_test:
#     name_bins_ytst = name + '_bins'
#     y_test[name_bins_ytst], bin_edges = pd.qcut(y_test[name], number_of_bins, labels=labels, retbins=True)
#     bin_edges_dict_test[name_bins_ytst]=bin_edges
#     # print(bin_edges_dict.items())
#     # print(bin_edges_dict_test[name_bins_ytst])

# ###This is storing the priorPDs so we can plot them
#     name_priors = name + '_test_priors'
#     prior = y_test[name_bins_ytst].value_counts(normalize=True).sort_index()
#     priorPDs = prior.to_dict()
#     prior_dict_ytst[name_priors] = priorPDs
#     # print(prior_dict_ytst.items())

# ###Must join x_test and y_test and pass to the BN:
# ###We are testing for outputs and therefore do not want to use any of the training data: 
# df_test_x = x_test.drop(['force', 'mass'], axis=1)
# df_test_y = y_test.drop(['acceleration'], axis=1)
# df_test_xy = pd.concat([df_test_x, df_test_y], axis=1)

# ###Pybbn only reads data types as strings, so this line converts the data in the csv from int64 to string 
# df_test_x = df_test_x.applymap(str)
# df_test_y = df_test_y.applymap(str)
# df_test_xy = df_test_xy.applymap(str)
# # print(df_test_x.head())

# ###inserting observation evidence
# ###This is saying that if there is evidence submitted in the arguments for the function to fill evidenceVars with that evidence
# ###e.g., evidence = {'deflection':[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'weight':[1.0, 0.0, 0.0, 0.0, 0.0, 0.0] }
# ev_dict = {}
# dataDict = {}
# # for col in df_test_x: ###col is the column header i.e., nod in the function.
# #     ev_dict = {'nod':col, 'bin_index':'2', 'val': 1.0}
# #     # ev = evidence(col, '2', 1.0) ###we want to apply hard evidence (100% prob) to the first bin, BUT THIS ISN'T SELECTING THE FIRST BIN, MENTION TO ZACK. 
# #     ev = evidence(**ev_dict)
# #     join_tree.set_observation(ev)
# # #     # print(ev_dict)

# for node, posteriors in join_tree.get_posteriors().items(): ### this is a list of dictionaries 
#     p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
#     print(f'{node} : {p}')
#     dataDict[node] = list(posteriors.values())
#     # print(dataDict[node])

# """
# CROSS VALIDATION METHODS:
# - RMSE USING MEAN OF PREDICTED BIN:
# """

# """MY VERSIONS BELOW"""

# # def expectedValue(dict, probabilities):
# #     expectedV = 0.0
# #     for varName, bin_ranges in dict: 
# #         v_max = bin_ranges[0]
# #         v_min = bin_ranges[1]
# #         mean_bin_value = ((v_max - v_min) / 2) + v_min
# #         expectedV += mean_bin_value * probabilities[varName]
# #     return expectedV

# # expectedV = expectedValue(bin_edges_dict.items(), dataDict[node][0])
# # print(expectedV)

# # def errors(pred_posteriors, testing_data, index, target):
# #     posteriorPDmean = []
# #     for posterior in pred_posteriors:
# #         posteriorPDmean.append(expectedValue(index[target], posterior))
# #     mse = mean_squared_error(testing_data[target], posteriorPDmean)
# #     rmse = sqrt(mse)
# #     return mse

# # mse = errors(dataDict[node], bin_edges_dict_test, bin_edges_dict, varName)

# expectedV = 0.0
# count = 0

# # for varName, index in bin_edges_dict_test.items():  
# #     if varName == 'acceleration_bins':
# #         mean_bin_value = np.zeros((len(bin_edges_dict.items()), len(index[:-1])))
# #         print(varName)
# #         print(index)
# #         print(dataDict[node])
# #         v_max = index[i][0]
        
# #         v_min = index[i][1]

# #         for i in range(len(index)-1): ###This for loop find the edges, binwidths and midpoints (xticksv) for each of the bins in the dict      
# #             mean_bin_value[count,i] = ((v_max[i+1] - v_min[i]) / 2.) + v_min[i]


# #         # mean_bin_value = ((v_max - v_min) / 2) + v_min ###i.e., the middle of the bin!
    
# #         expectedV += mean_bin_value * dataDict[node][0] ###this 
# #         print(v_max)
# #         print(v_min)
# #         print(mean_bin_value)
# #         print(expectedV) 

# #         posteriorPDmean = []

# #         for posterior in dataDict[node]:
# #             print(posterior)
# #             posteriorPDmean.append(expectedV)
# #             print(posteriorPDmean)
# #             mse = mean_squared_error(bin_edges_dict_test[varName], posteriorPDmean)
# #             print(mse)
# #             rmse = sqrt(mse)
# #             print()


# ###index = this is the list of bin edges from the bin edges.dict 
# ###pred_posteriors = this is the output probability distribution
# ###test and see if probabilities = pred_posteriors
# ###probabilities = datadict[node] for plotting
# ###target = acceleration_bins i.e., varName? 

# """ZACK'S VERSIONS BELOW"""

# # def expectedValue(bin_ranges, probabilities):
# #     expectedV = 0.0
# #     for index, bin_range in enumerate(bin_ranges):

# #         v_max = bin_range[0]
# #         v_min = bin_range[1]

# #         mean_bin_value = ((v_max - v_min / 2) + v_min)

# #         expectedV += mean_bin_value * probabilities[index]

# #     return expectedV    

# # def errors(pred_posteriors, testing_data, bin_ranges, target):
# #     posteriorPDmean = []
# #     for posterior in pred_posteriors:
# #         posteriorPDmean.append(expectedValue(bin_ranges[target], posterior))
# #     mse = mean_squared_error(testing_data[target], posteriorPDmean)
# #     rmse = sqrt(mse)
# #     return rmse

# ########################### LAPLACE SMOOTHING TO AVOID ZERO DIVISION ERROR WHEN WE HAVE EMPTY BINS #############################
# ### #Laplace smoothing should add 1 to the zero value in the probability calculation. 
# # for vertex in bn.V:
# #     #print 'vertex ', vertex
# #     # print bn.V[vertex]
# #     numBins = bn.Vdata[vertex]['numoutcomes']

# #     if not (bn.Vdata[vertex]["parents"]):  # has no parents
# #     #    for i in range(len(bn.Vdata[vertex]['cprob'])):
# #     #        bn.Vdata[vertex]['cprob'][i][0] += 1  # numerator (count)
# #     #        bn.Vdata[vertex]['cprob'][i][1] += numBins  # denomenator (total count)

# #         for counts in bn.Vdata[vertex]['cprob']:
# #             counts[0] += 1  # numerator (count) ###this is adding 1 to our zero value of probability 
# #             counts[1] += numBins  # denomenator (total count) ###including k-value (number of classes)


# #     else:

# #         countdict = bn.Vdata[vertex]['cprob']

# #         for key in countdict.keys():
# #             for counts in countdict[key]:
# #                 counts[0]+=1 
# #                 counts[1]+=numBins ###

#             #print '5 ------'

# ##This is for the figure parameters. 
# n_rows = 1
# n_cols = len(structure.keys()) ##the length of the BN i.e., five nodes

# ###instantiate a figure as a placaholder for each distribution (axes)
# fig = plt.figure(figsize=((200 * n_cols) / 96, (200 * n_rows) / 96), dpi=96, facecolor='white')
# fig.suptitle('Posterior Probabilities', fontsize=8) # title

# ###Instantiate a counter
# i = 0
# count = 0

# ###This is to plot the priorPDs that we stored above.
# priorPDs_dict = {}



# ###This creates a variable that corresponds to key (varname) and another variable which corresponds to the value (index)
# for varName, index in bin_edges_dict.items(): 
#     ax = fig.add_subplot(n_rows, n_cols, count+1) ###subplot with three arguments taken from above, including count
#     ax.set_facecolor("whitesmoke") ###sets the background colour of subplot
#     # print(index)

#     edge = np.zeros((len(bin_edges_dict.items()), len(index[:])))
#     binwidths = np.zeros((len(bin_edges_dict.items()), len(index[:-1])))
#     xticksv = np.zeros((len(bin_edges_dict.items()), len(index[:-1])))

#     ev2 = list(ev_dict.values()) 

#     for i in range(len(index)): ###This for loop find the edges, binwidths and midpoints (xticksv) for each of the bins in the dict      
#         edge[count, i] = index[i]       
                
#     for i in range(len(index)-1): ###This for loop find the edges, binwidths and midpoints (xticksv) for each of the bins in the dict      
#         binwidths[count, i] = (index[i+1] - index[i])
#         xticksv[count,i]  = ((index[i+1] - index[i]) / 2.) + index[i]

#     ###This line plots the bars using xticks on x axis, probabilities on the y and binwidths as bar widths. 
#     ###It counts through them for every loop within the outer for loop 
#     ###posteriotrs.values() is a dict and therefore not ordered, so need to a way to make it ordered for future use. 
#     ###This line plots the priorPDs that we stored in the forloop above. 
    
#     dataDict = {}

#     ###Loop goes through a dictionary which contains a key and a value
#     ##The first variable i.e., node will correspond to the key and the second i.e., posteriors, will correspond to the value.      
#     for node, posteriors in join_tree.get_posteriors().items(): ### this is a list of dictionaries 
#         p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
#         # print(posteriors) #So Dict(str,List[float]) and Dict(str,Dict(str,float))
#         # print(node) ##Can you make the second dict into the same data types as the first
#         if varName == node:
#             dataDict[node] = list(posteriors.values())
#             # print(dataDict[node])
#             if varName == 'acceleration_bins':
#                 ax.bar(xticksv[count], dataDict[node], align='center', width=binwidths[count], color='red', alpha=0.2, linewidth=0.2)         
#             elif varName == 'mass_bins' or 'force_bins':
#                 ax.bar(xticksv[count], dataDict[node], align='center', width=binwidths[count], color='green', alpha=0.2, linewidth=0.2)

#     # for var2, idx in prior_dict_xytrn.items():
#     #     print(var2)
#     #     print(idx)
#     #     priorPDs_dict[var2] = list(idx.values())
#     #     print(priorPDs_dict[var2])
#     #     ax.bar(xticksv[count], priorPDs_dict[var2], align='center', width=binwidths[count], color='black', alpha=0.2, linewidth=0.2)  
    

#     ###These lines plot the limits of each axis. 
#     plt.xlim(min(edge[count]), max(edge[count]))
#     plt.xticks([np.round(e, 2) for e in edge[count]], rotation='vertical')
#     plt.ylim(0, 1) 

#     ###These lines set labels and formatting style for the plots. 
#     ax.grid(color='0.2', linestyle=':', linewidth=0.1, dash_capstyle='round')
#     ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
#     ax.set_title(varName, fontweight="bold", size=6)
#     ax.set_ylabel('Probabilities', fontsize=7)  # Y label
#     ax.set_xlabel('Ranges', fontsize=7)  # X label
#     i+=1
#     count+=1

    
# fig.tight_layout()  # Improves appearance a bit.
# fig.subplots_adjust(top=0.85)  # white spacing between plots and title   
# # plt.show()
