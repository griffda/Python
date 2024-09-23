import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pickle
import pandas as pd

# Load the results from the pickle file
with open('sa_results_process_d2_0624.pkl', 'rb') as f:
    results = pickle.load(f)

# Convert the dictionary to a DataFrame
df = pd.DataFrame.from_dict(results, orient='index', columns=['nbins_input', 'nbins_output', 'prediction_accuracy', 'train_time', 'validate_time'])

# Save the DataFrame to a CSV file
df.to_csv('sa_results_process_d2_0624.csv')

# Load the data from the CSV file
data = pd.read_csv('sa_results_process_d1_0624.csv')

# Reshape the DataFrame
data = data.melt(id_vars=data.columns[0], var_name='inputs', value_name='accuracy')
data.columns = ['inputs', 'outputs', 'accuracy']
data[['inputs', 'outputs']] = data[['inputs', 'outputs']].apply(pd.to_numeric)

# Create grid values first.
xi = np.linspace(data['inputs'].min(), data['inputs'].max(), 100)
yi = np.linspace(data['outputs'].min(), data['outputs'].max(), 100)
xi, yi = np.meshgrid(xi, yi)

# Interpolate onto the grid
zi = griddata((data['inputs'], data['outputs']), data['accuracy'], (xi, yi), method='cubic')

# Create a figure for plotting a 3D surface plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Create a 3D surface plot
surf = ax.plot_surface(xi, yi, zi, rstride=1, cstride=1, cmap=plt.cm.coolwarm, linewidth=0.5, antialiased=True)


# Add a colorbar
cbar = fig.colorbar(surf, ax=ax, label='Prediction accuracy (%)', pad = 0.15)
cbar.ax.tick_params(labelsize=14) # set tick label font size
cbar.set_label('Prediction accuracy (%)', size=14) # set colorbar label font size

# Set labels
ax.set_xlabel('Number of bins: inputs', fontsize=14)
ax.set_ylabel('Number of bins: outputs', fontsize=14)
ax.invert_xaxis()

# Show the plot
plt.show()