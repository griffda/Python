import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import sklearn
import math
import operator

class ErrorCalculator:
    def __init__(self, binRanges):
        self.binRanges = binRanges

    def get_correct_values(self, obs_dicts, output):
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
                    correct_bin_locations.append(int(v['bin_index']) - 1)
                    if 'actual_value' in v:
                        actual_values.append(v['actual_value'])
                    else:
                        actual_values.append(None)
        return correct_bin_locations, actual_values

    def extract_bin_ranges(self, variable_name):
        """
        Change the bin format from [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] to [[0.0, 0.2], [0.2, 0.4], [0.4, 0.6], [0.6, 0.8], [0.8, 1.0]].

        Parameters
        ----------
        variable_name : str, name of the variable

        Returns
        -------
        bin_ranges : list of lists, bin ranges
        """
        bins = self.binRanges[variable_name]  # get the bin values for the specified variable
        bin_ranges = []
        for i in range(len(bins) - 1):
            bin_ranges.append([round(bins[i], 3), round(bins[i + 1], 3)])  # calculate the bin ranges
        return bin_ranges

    def distribution_distance_error(self, correct_bin_locations, predicted_bin_probabilities, actual_values, bin_ranges, calculate_d1=False):
        distance_errors = []
        norm_distance_errors = []
        output_bin_means = []
        actual_bins = []

        for i in range(0, len(bin_ranges)):
            max_bound = bin_ranges[i][1]
            min_bound = bin_ranges[i][0]
            output_bin_means.append(((max_bound - min_bound) * 0.5) + min_bound)

        for i in range(len(correct_bin_locations)):
            probabilities = predicted_bin_probabilities[i]
            index, value = max(enumerate(probabilities), key=operator.itemgetter(1))
            actual_bin = correct_bin_locations[i]

            if calculate_d1:
                distance_error = abs(output_bin_means[index] - actual_values[i])
            else:
                distance_error = abs(output_bin_means[index] - output_bin_means[actual_bin])

            norm_distance_error = distance_error / (bin_ranges[len(bin_ranges) - 1][1] - bin_ranges[0][0])

            distance_errors.append(round(distance_error, 3))
            norm_distance_errors.append(round(norm_distance_error, 3))

        average_error = sum(norm_distance_errors) / len(norm_distance_errors)
        prediction_accuracy = 1 - average_error

        return norm_distance_errors, prediction_accuracy

    def expectedValue(self, binRanges, probabilities):
        expectedV = 0.0

        for index, binrange in enumerate(binRanges):
            v_max = binrange[0]
            v_min = binrange[1]
            meanBinvalue = ((v_max - v_min) / 2) + v_min
            expectedV += meanBinvalue * probabilities[index]

        posteriorPDmeans.append(expectedV)

        return expectedV

    def generate_errors(self, predictedTargetPosteriors, testingData, binnedTestingData, target):
        posteriorPDmeans = []

        for posterior in predictedTargetPosteriors:
            posteriorPDmeans.append(self.expectedValue(self.binRanges[target], posterior))

        mse = mean_squared_error(testingData[target], posteriorPDmeans)
        rmse = math.sqrt(mse)

        loglossfunction = sklearn.metrics.log_loss(binnedTestingData[target], predictedTargetPosteriors, normalize=True, labels=range(0, len(self.binRanges[target])))
        norm_distance_errors = self.distribution_distance_error(binnedTestingData[target], predictedTargetPosteriors, testingData[target], self.binRanges[target], calculate_d1=False)

        correct_bin_probabilities = []
        for p in range(len(testingData[target])):
            correct_bin_probabilities.append(predictedTargetPosteriors[p][binnedTestingData[target][p]])

        return float(rmse), float(loglossfunction), norm_distance_errors, correct_bin_probabilities
