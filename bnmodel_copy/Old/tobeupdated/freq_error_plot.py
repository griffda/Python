import matplotlib.pyplot as plt


# Assume the dictionary of distance errors is called 'distance_errors_dict'
# distance_errors = list(distance_errors_dict.values())

# distance_errors = ([0.0, 0.05000000000000002, 0.30000000000000004], [0.0, 5.000000000000002, 30.000000000000004], [0.15, 0.44999999999999996, 0.8])
distance_errors = ([0.063, 0.087, 0.163, 0.15], [1.8504999999999998, 1.944, 1.9845000000000002, 2.0215, 2.065, 2.173])



# Plot the histogram of distance errors
plt.hist(distance_errors, bins=20, edgecolor='black')
plt.xlabel('Distance Error (%)')
plt.ylabel('Frequency')
plt.title('Histogram of Distance Errors')
plt.show()