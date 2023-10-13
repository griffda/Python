import operator
import matplotlib.pyplot as plt

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

        #print 'distance_error', distance_error
        #print 'max def value ', bin_ranges[len(bin_ranges) - 1][1]
        #print 'min def value ', bin_ranges[0][0]
        #print 'normalised distance error ', norm_distance_error

    if plot == True:
        plt.hist(norm_distance_errors, bins=15)
        plt.xlim(-1, 1)
        plt.show()

    return norm_distance_errors