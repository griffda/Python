import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # for drawing graphs
from sklearn.metrics import mean_squared_error
import sklearn
import math
import operator

def get_correct_values(obs_dicts, output):
    """
    Get the bin location and the actual values of the target.

    Parameters
    ----------
    obs_dicts : list of dicts with the bin index and values
    output : str, name of the target variable

    Returns
    -------
    correct_bin_locations : list of ints, bin indices of the actual values
    actual_values : list of floats, actual values of the target variable
    """
    correct_bin_locations = []
    actual_values = []
    for d in obs_dicts:
        for k, v in d.items():
            if k == output:
                correct_bin_locations.append(int(v['bin_index']))
                if 'actual_value' in v:
                    actual_values.append(v['actual_value'])
                else:
                    actual_values.append(None)
    # print('bin_indices:', bin_indices)
    # print('actual_values:', actual_values)  
    return correct_bin_locations, actual_values


def extract_bin_ranges(variable_name, bin_dict):
    """
    Change the bin format from [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] to [[0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0]].

    Parameters
    ----------
    variable_name : str, name of the variable
    bin_dict : dict, dictionary of bin values

    Returns
    -------
    bin_ranges : list of lists, bin ranges
    """
    bins = bin_dict[variable_name] # get the bin values for the specified variable
    bin_ranges = []
    for i in range(len(bins) - 1):
        bin_ranges.append([round(bins[i],3), round(bins[i+1],3)]) # calculate the bin ranges
    # print(variable_name, bin_ranges)
    return bin_ranges


def distribution_distance_error(correct_bin_locations, predicted_bin_probabilities, actual_values, bin_ranges):

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

        # norm_distance_error = (distance_error - bin_ranges[0][0]) / (
        # bin_ranges[len(bin_ranges) - 1][1] - bin_ranges[0][0])
        norm_distance_error = distance_error/ (bin_ranges[len(bin_ranges) - 1][1] - bin_ranges[0][0])

        distance_errors.append(round(distance_error,3))
        # norm_distance_errors.append(round(norm_distance_error*100,3)) # remove 100 to normalise
        norm_distance_errors.append(round(norm_distance_error,3))

        # Calculate the average error
        average_error = sum(norm_distance_errors) / len(norm_distance_errors)

        # Calculate the prediction accuracy
        prediction_accuracy = 1 - average_error

    # Print the prediction accuracy
    #print("Prediction Accuracy: {:.2%}".format(prediction_accuracy))

    return norm_distance_errors, prediction_accuracy


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