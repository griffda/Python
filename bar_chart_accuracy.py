import numpy as np
import matplotlib.pyplot as plt

# Data from the table
dataset_size = [1400, 5120, 10240]
bin_resolution = [5, 7, 10]
accuracy_d1 = [
    [92.73, 93.74, 89.37],
    [92.84, 94.08, 91.20],
    [93.13, 94.79, 90.86]
]
accuracy_d2 = [
    [96.36, 96.98, 88.25],
    [96.76, 96.79, 89.00],
    [96.73, 97.00, 89.40]
]
bar_width = 0.25
num_datasets = len(dataset_size)

# Define the x-axis positions for each group
x = np.arange(len(bin_resolution))

# Define the colors for the bars
colors = ['navy', 'royalblue', 'cornflowerblue']

# Plotting the grouped bar chart
fig, ax = plt.subplots()

# Loop over the dataset sizes
for i in range(num_datasets):
    # Plot the bars for D1 and D2
    rects1 = ax.bar(x + (i - (num_datasets - 1) / 2) * bar_width, accuracy_d1[i], bar_width,
                    label=f'D1 - Dataset Size: {dataset_size[i]}', color=colors[i], edgecolor='black')
    rects2 = ax.bar(x + (i - (num_datasets - 1) / 2) * bar_width, accuracy_d2[i], bar_width,
                    label=f'D2 - Dataset Size: {dataset_size[i]}', alpha=0.5, color=colors[i], edgecolor='black')

    # Function to label each bar with its value
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom')

    # Label each bar
    autolabel(rects1)
    autolabel(rects2)

# Determine the maximum value of accuracy data
max_accuracy = max([max(d1 + d2) for d1, d2 in zip(accuracy_d1, accuracy_d2)])

# Set x-axis labels and tick positions
ax.set_xlabel('Bin Resolution', fontsize=12)
ax.set_ylabel('Prediction Accuracy %',  fontsize=12)
ax.grid(color='0.2', linestyle=':', linewidth=0.1, dash_capstyle='round')
ax.set_facecolor('whitesmoke')
ax.set_xticks(x)
ax.set_xticklabels(bin_resolution)

# Set y-axis limits
ax.set_ylim([80, max_accuracy + 5])

# Add a legend
ax.legend()

# Set a title
plt.title('Prediction Accuracy by Dataset Size and Bin Resolution', fontsize=14, fontweight='bold')

# Display the chart
plt.tight_layout()
plt.show(block=False)



