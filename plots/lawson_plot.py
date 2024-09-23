import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm

# Data upload csv 
data = pd.read_csv('data/lawson_data.csv')

# Filter out non-positive values for 'Lawson'
data = data[data['Lawson'] > 0]

# Group data by 'Concept'
grouped_data = data.groupby('Concept')

# Create the plot
plt.figure(figsize=(10, 6))

# Create a colormap
colors = cm.get_cmap('tab20', len(grouped_data))

# Plot each group with a unique color
for idx, (concept, group) in enumerate(grouped_data):
    color = colors(idx)
    if concept in ['ITER', 'SPARC', 'NIF']:
        plt.plot(group['Year'], group['Lawson'], label=concept, color=color)
        plt.scatter(group['Year'], group['Lawson'], s=90, color=color)  # Larger dots for ITER, SPARC, and NIF
        if concept == 'ITER': # Add an annotation for ITER
            plt.annotate('ITER', (group['Year'].values[0], group['Lawson'].values[0]), textcoords='offset points', xytext=(5, 5), ha='center', fontsize=16)
        if concept == 'SPARC': # Add an annotation for SPARC
            plt.annotate('SPARC', (group['Year'].values[0], group['Lawson'].values[0]), textcoords='offset points', xytext=(5, -20), ha='center', fontsize=16)
        if concept == 'NIF': # Add an annotation for NIF
            plt.annotate('NIF', (group['Year'].values[0], group['Lawson'].values[0]), textcoords='offset points', xytext=(5, 5), ha='center', fontsize=16)
    else:
        plt.plot(group['Year'], group['Lawson'], label=concept, color=color)
        plt.scatter(group['Year'], group['Lawson'], s=50, color=color)  # Regular dots for other concepts

# Set y-axis to log scale
plt.yscale('log')

# Add a dashed horizontal line at 4.0e21
plt.axhline(y=4.5e21, color='red', linestyle='--')
plt.text(x=1970,y=5.0e21,s='CURRENT POSITION',color='red',fontsize=16)

# Add labels and title
plt.xlabel('Year', fontsize=16)
plt.ylabel(r'Triple Product $\eta T \tau_E$ (keV s m$^{-3}$)', fontsize=16)
# Set face color of the plot
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().set_facecolor('whitesmoke')
plt.legend(fontsize=14)
plt.grid(True)

# Show the plot
plt.show()