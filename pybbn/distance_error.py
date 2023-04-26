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
        prior_mean = np.mean(priorDict_training[node])
        # print(prior_dict[node])
        # print(prior_mean)



###Step 6: Validate Bayes Net requires: 
###input testing set, which through inference gives predicted output set
with open('x_test_bins.pkl', 'rb') as f:
    x_test_bins = pickle.load(f)
    # print(x_test_bins)

###output testing set
###These are the testing set probability distribution for the output (y target) (output testing set)
with open('y_testing_probs.pkl', 'rb') as f:
    y_testing_probs = pickle.load(f)
    print(y_testing_probs)
    y_testing_probs = list(y_testing_probs.values())
    print(y_testing_probs)

# for node, posteriors in y_testing_probs.items(): ### this is a list of dictionaries 
#     print(node)
#     print(posteriors)
    # y_testing_probsDict[node] = list(posteriors)
    # print(y_testing_probs[node])
    # p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
    # print(f'{node} : {p}')
    # if node == 'acceleration_bins':
    #     y_testing_probsDict[node] = list(posteriors.values())
    #     print(posteriorDict_training[node])


###These are the testing set probabilities for inputs and outputs with evidence applied. 
####Predicted output set
with open('posteriors_evidence.pkl', 'rb') as f:
    posteriors_evidence = pickle.load(f)

print(posteriors_evidence)


for node, posteriors in posteriors_evidence.items(): ### this is a list of dictionaries 
    p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
    print(f'{node} : {p}')
    posteriorDict_testing[node] = list(posteriors.values())

    # print(posteriorDict_testing['acceleration_bins'])
    # print(posteriorDict_testing[node])
    

expectedV = 0.0
count = 0
i = 0 

###These are the bins for the training set inputs and outputs:
with open('bin_edges_dict_train.pkl', 'rb') as f:
    bin_edges_train = pickle.load(f)
for varName, index in bin_edges_train.items():
    # print(bin_edges_dict_test.items())
    if varName == 'acceleration_bins':
        # print(posteriorDict_testing['acceleration_bins'])
        mean_bin_value = np.zeros((len(bin_edges_train['acceleration_bins'][:-6]), len(index[:-1])))
        mean2 = np.mean(index)
        # print(mean2)
        for i in range(len(index)-1): ###This for loop find the edges, binwidths and midpoints (xticksv) for each of the bins in the dict      
            # print(index)
            mean_bin_value[count,i] = ((index[i+1] - index[i]) / 2.) + index[i]
        expectedV += mean_bin_value * priorDict_training['acceleration_bins'] ###this 
        # print(varName)
        # print(index)
        # print(mean_bin_value)
        # print(expectedV)


###These are the bins for the testing set outputs:
with open('y_test_bins.pkl', 'rb') as f:
    y_test_bins = pickle.load(f)
    # y_test_binsL = list(y_test_bins.values())
# for varName, index in y_test_bins.items():  
#     # print(y_test_bins.items())
#     if varName == 'acceleration_bins':        
#         mean_bin_value = np.zeros((len(y_test_bins['acceleration_bins'][:-6]), len(index[:-1])))
#         # print(mean_bin_value)
#         # mean_bin_value_predicted = ((index[6] - index[5]) / 2) + index[5] ###just for distribution of predictions which contain values 
#         # output_bin_means.append(mean_bin_value_predicted)
#         for i in range(len(index)-1): ###This for loop find the edges, binwidths and midpoints (xticksv) for each of the bins in the dict      
#                 # print(index)
#             mean_bin_value = ((index[i+1] - index[i]) / 2.) + index[i]
#             # expectedV += mean_bin_value * posteriorDict_testing['acceleration_bins']
#             output_bin_means.append(mean_bin_value)
#         print('output bin means2', output_bin_means)
#         # print('y test bins', y_test_bins)
       
# output_bin_means = [] ###output bin means
# distance_errors = []
# norm_distance_errors = []


# for i in range(0, len(bin_ranges)):

#     max_bound = bin_ranges[i][1]
#     min_bound = bin_ranges[i][0]

#     output_bin_means.append(((max_bound - min_bound) * 0.5) + min_bound)

# print('output bin means1', output_bin_means)   

# correct_bin_locations = [0,1,2,3,4,5]
# predicted_bin_probabilities = posteriorDict_testing['acceleration_bins']
# # bin_ranges = index


# y_test_binsL = list(y_test_bins['acceleration_bins'])


# for i in range(len(correct_bin_locations)):
#     probabilities = predicted_bin_probabilities
#     idx, value = max(enumerate(probabilities), key=operator.itemgetter(1))  # finds bin with max probability and returns it's value and index
#     # print(idx)
#     # print(value)
#     actual_bin = correct_bin_locations[i]  # bin containing actual value

#     # distance between bin means
#     # distance_error = abs(output_bin_means[predicted_bin] - output_bin_means[actual_bin])
#     # OR
#     # distance between actual value and bin mean
#     distance_error = abs(output_bin_means[idx] - actual_values[i])
#     # print(distance_error)

#     norm_distance_error = (distance_error - bin_ranges[0][0]) / (bin_ranges[len(bin_ranges) - 1][1] - bin_ranges[0][0])
#     # print(norm_distance_error)
#     distance_errors.append(distance_error)
#     norm_distance_errors.append(norm_distance_error*100) # remove 100 to normalise

#     print('distance_error:', distance_error)
#     print('max def value:', bin_ranges[len(bin_ranges) - 1][1])
#     print('min def value:', bin_ranges[0][0])
#     print('normalised distance error:', norm_distance_error)

#     plt.hist(norm_distance_errors, bins=15)
#     plt.xlim(-1, 1)
#     # plt.show()       

# expectedV = 0.0
# probabilities = [0.0, 0.0, 0.0, 0.0, 0.15, 0.85]
# probabilities = [0.0, 0.015, 0.035, 0.5, 0.10, 0.80]
# predictedTargetPosteriors = probabilities

# for index, binrange in enumerate(bin_ranges):

#     v_max = binrange[0]
#     v_min = binrange[1]

#     print(v_max)
#     print(v_min)
    
#     meanBinvalue = ((v_max - v_min) / 2) + v_min
#     print('mean bin value:', meanBinvalue)

#     # expectedV += meanBinvalue * predictedTargetPosteriors[index]
#     expectedV += meanBinvalue * probabilities[index]

#     print('expected value:', expectedV)

# expectedV = expectedValue(bin_ranges, probabilities)
# print(expectedV)

# for posterior in predictedTargetPosteriors:
#     # print(posterior)
#     posteriorPDmeans.append(expectedValue(bin_ranges, predictedTargetPosteriors))
# # print('posteriorPDmeans:' , posteriorPDmeans)


# mse = mean_squared_error(testing_data, posteriorPDmeans)
# rmse = math.sqrt(mse)

# # loglossfunction = sklearn.metrics.log_loss(output_bin_means, probabilities,normalize=True, labels=range(0, len(bin_ranges)))
# norm_distance_errors = distribution_distance_error(binnedTestingData[target], predictedTargetPosteriors,testingData[target], binRanges[target], False)

# ##if we take the above code, we see that by taking target1, in the first row of posteriors 0th bin, the value is 0.2
# ##do the same for the second row, and the 1st bin is 0.3. 
# correct_bin_probabilities = []
# for p in range(len(testing_data['target1'])):
#     correct_bin_probabilities.append(predictedTargetPosteriors[p][binnedTestingData['target1'][p]])

# print('rmse:', float(rmse))
# # print('loglossfunction:', float(loglossfunction))
# print('correct bin probabilities', correct_bin_probabilities)

###implementing zacks code to plot the distance errors.  

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

###This argument is a list of actual output values from the testing dataset 
###which are used to compute the distance error between the predicted bin and the actual bin. 
###e.g, the first value should corrspond to the first bin in correct bin locations data. 
actual_values = [0.4, 1.2, 2.5, 3.6, 4.8, 2.5]

###bin_ranges is a list of lists, where each inner list contains the upper and lower bounds of each bin.
bins_dict = y_test_bins

###this function does a good job of extracting bin ranges from the test dictionary. 
def extract_bin_ranges(variable_name, bin_dict): # specify the variable name for which to extract bin ranges
    bins = bin_dict.get(f"{variable_name}_bins") # get the bin values for the specified variable
    bin_ranges = []
    for i in range(len(bins) - 1):
        bin_ranges.append([bins[i], bins[i+1]]) # calculate the bin ranges
    return bin_ranges

acceleration_bin_ranges = extract_bin_ranges('acceleration', bins_dict)
bin_ranges = extract_bin_ranges('acceleration', bins_dict)

# print('acceleration_bin_ranges: ', acceleration_bin_ranges) # Output: [[1, 2], [2, 3], [3, 4], [4, 5]]
# bin_ranges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]] ##this is the same as I used

###this is an example of how to call the funtction
###norm_distance_errors = distribution_distance_error(correct_bin_locations, predicted_bin_probabilities, actual_values, bin_ranges, plot)


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

        distance_errors.append(distance_error)
        norm_distance_errors.append(norm_distance_error*100) # remove 100 to normalise

        print('distance_error:', distance_error)
        print('max def value:', bin_ranges[len(bin_ranges) - 1][1])
        print('min def value:', bin_ranges[0][0])
        print('normalised distance error:', norm_distance_error)

    if plot == True:
        plt.hist(norm_distance_errors, bins=15)
        plt.xlim(-1, 1)
        plt.show()

    return norm_distance_errors, output_bin_means

output_bin_means, norm_distance_errors = distribution_distance_error(correct_bin_locations, predicted_bin_probabilities, actual_values, bin_ranges, plot=False)

###implementing zacks code via generateErrors function to calculate rmse
###this is how the different arguments could be taken. 
###In this example, predictedTargetPosteriors is a list of lists representing the posterior probabilities for each test data point, 
predictedTargetPosteriors = [[0.1, 0.4, 0.5], 
                             [0.7, 0.2, 0.1], 
                             [0.2, 0.2, 0.6]]

###testingData is a dictionary where the keys are the target variable names and the values are lists of actual target values for each test data point
testingData = {'target1': [0.8, 0.2, 0.5], 
               'target2': [0.1, 0.3, 0.6]}

###NOT LIKE THIS - THIS IS A PROBABILITY DISTRIBUTION, NOT HARD VALUES. 
testing_data2 = {'target1': [0.168, 0.168, 0.164, 0.168, 0.164, 0.168],
                'target2': [0.168, 0.168, 0.164, 0.168, 0.164, 0.168],
                'target3': [0.168, 0.168, 0.164, 0.168, 0.164, 0.168]}

###binnedTestingData is also a dictionary where the keys are the target variable names and the values are lists of bin numbers for each test data point, 
binnedTestingData = {'target1': [1, 0, 2], 
                     'target2': [0, 0, 2]}

###same as above
binRanges = {'target1': [(0, 0.3), (0.3, 0.6), (0.6, 1)], 
             'target2': [(0, 0.3), (0.3, 0.6), (0.6, 1)]}

bin_ranges = extract_bin_ranges('acceleration', bins_dict)

target = 'target1'


posteriorPDmeans = []

# probabilities = [0.0, 0.015, 0.035, 0.5, 0.10, 0.80]
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
        correct_bin_probabilities.append(predictedTargetPosteriors[p][binnedTestingData[target][p]])


    return float(rmse),float(loglossfunction),norm_distance_errors,correct_bin_probabilities

rmse, loglossfunction, norm_distance_errors, correct_bin_probabilities = generateErrors(predictedTargetPosteriors, testingData, binnedTestingData, binRanges, target)


testingData = {'target': [0.8, 0.2, 0.5]}

binRanges = {'target': [(0, 0.3), (0.3, 0.6), (0.6, 1)]}

test_value = testingData['target'][0]  # get the testing data value for 'target'
bin_ranges = binRanges['target']  # get the bin ranges for 'target'
bin_index = None
for i, (start, end) in enumerate(bin_ranges):
    if start <= test_value < end:
        bin_index = i
        break
if bin_index is None:
    bin_index = len(bin_ranges)  # if the value is outside all ranges, assign it to the last bin
print('target value bin index: ',bin_index)


def get_target_bin_indices(test_x, obs_dict, default_bin=5):
    bin_indices = []
    for col in test_x.columns:
        if col in obs_dict:
            bin_indices.append(obs_dict[col]['bin_index'])
        else:
            bin_indices.append(str(default_bin))

    return bin_indices


test_y = pd.DataFrame({'target': [0.25, 0.8, 0.4, 0.6, 0.9]})
obs_dict = {'mass_bins': {'bin_index': '1', 'val': 1.0},
            'force_bins': {'bin_index': '3', 'val': 1.0}}



