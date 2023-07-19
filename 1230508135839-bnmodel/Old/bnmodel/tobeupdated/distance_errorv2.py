"""
Created on Wed Nov 16 13:58:57 2022

@author: tomgriffiths

In this script we will attepmt to create a BN using the library pyBBN 
for f = ma linear model.

In this script we will attempy to perform cross validation of the BN model: 
- train several models on different subsets of training data
- evaluate them on the complementary subset of testing data. 

This takes dictionaries of output data and does validation so that model does not have
to be re-run every time. 

"""


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # for drawing graphs
from sklearn.metrics import mean_squared_error
import sklearn
import math
import operator


posteriorDict_training = {}
priorDict_training = {}
posteriorDict_testing = {}
y_testing_probsDict = {}

###Results of STEP 5 Learn Bayes Net:
###These are the training set probabilities for the inputs and outputs with NO evidence applied. 
with open('xy_train_priors.pkl', 'rb') as f:
    xy_train_priors = pickle.load(f)
    # print(xy_train_priors.items())
for node, posteriors in xy_train_priors.items(): ### this is a list of dictionaries 
    p_no_ev = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
    # print(f'{node} : {p_no_ev}')
    if node == 'acceleration_bins':
        priorDict_training[node] = (list(posteriors.values()))

###Step 6: Validate Bayes Net requires: 
###input testing set, which through inference gives predicted output set
with open('x_test_bins.pkl', 'rb') as f:
    x_test_bins = pickle.load(f)
    # print(x_test_bins)

###output testing set
###These are the testing set probability distribution for the output (y target) (output testing set)
with open('y_testing_probs.pkl', 'rb') as f:
    y_testing_probs = pickle.load(f)
    # print(y_testing_probs)
    y_testing_probs = list(y_testing_probs.values())
    # print(y_testing_probs)

with open('y_testing_probs2.pkl', 'rb') as f:
    testingData_y = pickle.load(f)
    # print(testingData_y)

###These are the testing set probabilities for inputs and outputs with evidence applied. 
####Predicted output set
with open('posteriors_evidence.pkl', 'rb') as f:
    posteriors_evidence = pickle.load(f)

for node, posteriors in posteriors_evidence.items(): ### this is a list of dictionaries 
    p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
    print(f'{node} : {p}')
    posteriorDict_testing[node] = list(posteriors.values())

expectedV = 0.0
count = 0
i = 0 

###These are the bins for the training set inputs and outputs:
with open('bin_edges_dict_train.pkl', 'rb') as f:
    bin_edges_train = pickle.load(f)

###These are the bins for the testing set outputs:
with open('y_test_bins.pkl', 'rb') as f:
    y_test_bins = pickle.load(f)


###here is how the different arguments could be taken: 
###this is the same as I used anyway. 
###correct_bin_locations is a list of integers indicating the bin locations of the actual values. 
###i.e., when evidence is placed on testing data, this reveals the bin locations of where the model should place the prediction
###e.g., bin 0 will be where correct bin is for testing data. 
correct_bin_locations = [0, 1, 2, 3, 4]

###predicted_bin_probabilities is a list of lists
###where each inner list contains the predicted probabilities for each bin for a single test data point.
###i.e., every time a distribution is computed from setting evidence to bins
predicted_bin_probabilities = [[0.1, 0.2, 0.3, 0.2, 0.2],
                               [0.2, 0.3, 0.1, 0.2, 0.2],
                               [0.3, 0.2, 0.1, 0.1, 0.3],
                               [0.2, 0.2, 0.2, 0.2, 0.2],
                               [0.1, 0.1, 0.4, 0.2, 0.2]]


###MY DATA
Predicted_target_posteriors = [[0.0, 0.0, 0.0, 0.0, 0.04, 0.94], 
                              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 
                              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 
                              [0.17, 0.17, 0.17, 0.17, 0.17, 0.17]]

Observation_posteriors = {'acceleration': [[0.03, 0.03, 0.03, 0.03, 0.03, 0.86], [0.17, 0.17, 0.17, 0.17, 0.17, 0.17], [0.37, 0.33, 0.16, 0.02, 0.11, 0.02], [0.02, 0.08, 0.11, 0.11, 0.29, 0.41]], 
                         'force': [[0.07, 0.22, 0.36, 0.25, 0.08, 0.02], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]], 
                         'mass': [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.04, 0.18, 0.42, 0.3, 0.06], [0.0, 0.04, 0.18, 0.42, 0.3, 0.06]]}


Observation_dictionary = {'mass_bins': {'bin_index': '2', 'val': 1.0}, 
                          'force_bins': {'bin_index': '5', 'val': 1.0}, 
                          'acceleration_bins': {'bin_index': '10', 'actual_value': 2.24}}

Observation_dictionaries = [{'mass_bins': {'bin_index': '2', 'val': 1.0}, 
                             'force_bins': {'bin_index': '4', 'val': 1.0}, 
                             'acceleration_bins': {'bin_index': '6', 'actual_value': 2.11}}, 
                            {'mass_bins': {'bin_index': '1', 'val': 1.0}, 
                             'force_bins': {'bin_index': '5', 'val': 1.0}, 
                             'acceleration_bins': {'bin_index': '6', 'actual_value': 2.26}}, 
                            {'mass_bins': {'bin_index': '3', 'val': 1.0}, 
                            'force_bins': {'bin_index': '3', 'val': 1.0}, 
                            'acceleration_bins': {'bin_index': '4', 'actual_value': 2.01}}, 
                            {'mass_bins': {'bin_index': '4', 'val': 1.0}, 
                            'force_bins': {'bin_index': '4', 'val': 1.0}, 
                            'acceleration_bins': {'bin_index': '4', 'actual_value': 2.0}}]

def extract_data_from_dict_list(dict_list, target_dict):
    bin_indices = []
    actual_values = []
    for d in dict_list:
        for k, v in d.items():
            if k == target_dict:
                bin_indices.append(int(v['bin_index']))
                if 'actual_value' in v:
                    actual_values.append(v['actual_value'])
                else:
                    actual_values.append(None)
    print('bin_indices:', bin_indices)
    print('actual_values:', actual_values)  
    return bin_indices, actual_values

target_dict = 'acceleration_bins'
bin_indices, actual_values = extract_data_from_dict_list(Observation_dictionaries, target_dict)

bin_indices = [6, 6, 4, 4]
actual_values = [2.11, 2.26, 2.01, 2.0]
                              
###This argument is a list of actual output values from the testing dataset 
###which are used to compute the distance error between the predicted bin and the actual bin. 
###e.g, the first value should corrspond to the first bin in correct bin locations data. 


###bin_ranges is a list of lists, where each inner list contains the upper and lower bounds of each bin.
bins_dict = y_test_bins

###this function does a good job of extracting bin ranges from the test dictionary. 
def extract_bin_ranges(variable_name, bin_dict): # specify the variable name for which to extract bin ranges
    bins = bin_dict.get(f"{variable_name}_bins") # get the bin values for the specified variable
    bin_ranges = []
    for i in range(len(bins) - 1):
        bin_ranges.append([round(bins[i],3), round(bins[i+1],3)]) # calculate the bin ranges
    print(variable_name, bin_ranges)
    return bin_ranges

# acceleration_bin_ranges = extract_bin_ranges('acceleration', bins_dict)
bin_ranges = extract_bin_ranges('acceleration', bins_dict)

def distribution_distance_error(correct_bin_locations, predicted_bin_probabilities, actual_values, bin_ranges, plot=False):

    distance_errors = []
    norm_distance_errors = []
    output_bin_means = []

    for i in range(0, len(bin_ranges)):

        max_bound = bin_ranges[i][1]
        min_bound = bin_ranges[i][0]

        output_bin_means.append(((max_bound - min_bound) * 0.5) + min_bound)

    for i in range(len(correct_bin_locations)):
        probabilities = predicted_bin_probabilities[i]
        index, value = max(enumerate(probabilities), key=operator.itemgetter(1))  # finds bin with max probability and returns it's value and index
        actual_bin = correct_bin_locations[i]  # bin containing actual value

        # distance between bin means
        # distance_error = abs(output_bin_means[predicted_bin] - output_bin_means[actual_bin])
        # OR
        # distance between actual value and bin mean
        distance_error = abs(output_bin_means[index] - actual_values[i])

        norm_distance_error = (distance_error - bin_ranges[0][0]) / (
        bin_ranges[len(bin_ranges) - 1][1] - bin_ranges[0][0])

        distance_errors.append(round(distance_error,3))
        norm_distance_errors.append(round(norm_distance_error*100,3)) # remove 100 to normalise

        # print('distance_error:', round(distance_error,3))
        # print('max def value:', bin_ranges[len(bin_ranges) - 1][1])
        # print('min def value:', bin_ranges[0][0])
        # print('normalised distance error:', round(norm_distance_error,3))

    if plot == True:
        plt.hist(norm_distance_errors, bins=15)
        plt.xlim(-1, 1)
        plt.show()

    return distance_errors, norm_distance_errors, output_bin_means

distance_errors, norm_distance_errors, output_bin_means = distribution_distance_error(bin_indices, Predicted_target_posteriors, actual_values, bin_ranges, plot=False)

###implementing zacks code via generateErrors function to calculate rmse
###MY VERSION
testingData = {'acceleration': [2.11, 2.26, 2.01, 2.0]}

###MY VERSION
###because it reads first value of bins as 0 this is giving out of range error. 
# binnedTestingData = {'acceleration': [6, 6, 4, 4]}
binnedTestingData = {'acceleration': [5, 5, 3, 3]}

binRanges = {'acceleration': [(1.779, 1.922), (1.922, 1.966), (1.966, 2.003), (2.003, 2.04),(2.04, 2.09),(2.09, 2.256)]}

# bin_ranges = extract_bin_ranges('acceleration', bins_dict)
###MY VERSION
target = 'acceleration'

posteriorPDmeans = []

###function for expected value
def expectedValue(binRanges, probabilities):

    expectedV = 0.0

    for index, binrange in enumerate(binRanges):

        v_max = binrange[0]
        v_min = binrange[1]

        meanBinvalue = ((v_max - v_min) / 2) + v_min

        expectedV += meanBinvalue * probabilities[index]
        posteriorPDmeans.append(expectedV)

    return expectedV

def generateErrors (predictedTargetPosteriors, testingData, binnedTestingData, binRanges, target):

    posteriorPDmeans = []

    for posterior in predictedTargetPosteriors:

        posteriorPDmeans.append(expectedValue((binRanges[target]), posterior))

    mse = mean_squared_error(testingData[target], posteriorPDmeans)
    # mse = mean_squared_error(unbinnedTargetActual[targetList[0]], posteriorPDmeans)
    rmse = math.sqrt(mse)

    #print 'binnedTestingData[target] ', binnedTestingData[target]
    #print 'predictedTargetPosteiors ', predictedTargetPosteriors

    loglossfunction = sklearn.metrics.log_loss(binnedTestingData[target], predictedTargetPosteriors,normalize=True, labels=range(0, len(binRanges[target])))
    norm_distance_errors = distribution_distance_error(binnedTestingData[target], predictedTargetPosteriors,testingData[target], binRanges[target], False)

    correct_bin_probabilities = []
    for p in range(len(testingData[target])):
        # print(p)
        correct_bin_probabilities.append(predictedTargetPosteriors[p][binnedTestingData[target][p]])
    print(correct_bin_probabilities)

    return float(rmse),float(loglossfunction),norm_distance_errors,correct_bin_probabilities

rmse, loglossfunction, norm_distance_errors, correct_bin_probabilities = generateErrors(Predicted_target_posteriors, testingData, binnedTestingData, binRanges, target)

print(distance_errors)
print(norm_distance_errors)
