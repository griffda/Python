import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from matplotlib.ticker import LinearLocator
from matplotlib import cm
import pickle
import seaborn as sns


def plot_errors(norm_distance_errors, histnbins, prediction_accuracy, av_prediction_accuracy, axperrow=2):
    """
    Plot the errors in a histogram.
    Each figure contains subplots, and each subplot is a fold.

    Parameters
    ----------
    norm_distance_errors : list of floats, normalised distance errors
    output_bin_means : list of floats, bin means
    target : str, name of the target variable
    histnbins : int, number of bins for the histogram (kfoldnbins)
    axperrow : int Number of axis per row

    """
    # Get the number of axes
    nax = len(norm_distance_errors)
    nrow = int(np.ceil(nax / axperrow))
    ncol = axperrow
    
    # Create the figure and axes objects
    fig, ax = plt.subplots(nrow, ncol, figsize=(12, 4), squeeze=False)
    ax = ax.flatten()  # Flatten the axes array into a 1D array
    
    fig.suptitle('Normalised distance error distributions {:.2%}'.format(av_prediction_accuracy), fontsize=16)

    for i in range(nax):
        ax[i].hist(norm_distance_errors[i], bins=histnbins, linewidth=0.2, edgecolor='black', color='black')
        ax[i].set_title('Fold {}, Prediction Accuracy: {:.2%}'.format(i+1, prediction_accuracy[i]))
        ax[i].set_xlim([0, 1])
        ax[i].grid(True, linestyle='--', alpha=0.5)
        ax[i].set_facecolor('whitesmoke')
        ax[i].set_ylabel('Frequency')
        ax[i].set_xlabel('Normalised distance error')
        ax[i].set_xlim([0, 1])

    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show(block=False)
    return ax



def plot_sensitivity_analysis(results):#3D scatter plot of sensitivity analysis
    with open('sa_results.pkl', 'rb') as f:
        results = pickle.load(f)
    bin_configs = list(results.keys())
    nbins_inputs = [results[config][0] for config in bin_configs]
    nbins_outputs = [results[config][1] for config in bin_configs]
    accuracies = [results[config][2] for config in bin_configs]
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(nbins_inputs, nbins_outputs, accuracies)
    ax.set_xlabel('Number of bins for inputs')
    ax.set_ylabel('Number of bins for outputs')
    ax.set_zlabel('Prediction accuracy')
    
    plt.show()

def plot_sensitivity_analysis(results):#2D line plot of sensitivity analysis, 2 subplots, change to scatter plot 
    bin_configs = list(results.keys())
    nbins_inputs = [results[config][0] for config in bin_configs]
    nbins_outputs = [results[config][1] for config in bin_configs]
    accuracies = [results[config][2] for config in bin_configs]
    
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(nbins_inputs, accuracies)
    plt.xlabel('Number of bins for inputs')
    plt.ylabel('Prediction accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(nbins_outputs, accuracies)
    plt.xlabel('Number of bins for outputs')
    plt.ylabel('Prediction accuracy')
    
    plt.tight_layout()
    plt.show()

def plot_sensitivity_analysis(results):#3D surface plot of sensitivity analysis
    with open('sa_results5k.pkl', 'rb') as f:
        results = pickle.load(f)
    # Convert dictionary to DataFrame
    data = pd.DataFrame(results.values(), columns=['inputs', 'outputs', 'accuracy'])

    # Convert accuracy to percentage
    data['accuracy'] = data['accuracy'] * 100

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
    fig.colorbar(surf, ax=ax, label='Prediction accuracy (%)')
    #increase space between colour bar and plot
    fig.subplots_adjust(right=1)

    # Set labels
    ax.set_xlabel('Number of bins for inputs', fontsize=14)
    ax.set_ylabel('Number of bins for outputs', fontsize=14)
    ax.set_zlabel('Prediction accuracy (%)', fontsize=14)
    ax.set_title('Bin-configuration sensitivity analysis', fontsize=16)
    ax.invert_xaxis()

    # Show the plot
    plt.show()

def plot_sensitivity_analysis(results):#2D contour plot of sensitivity analysis
    # Convert dictionary to DataFrame
    data = pd.DataFrame(results.values(), columns=['inputs', 'outputs', 'accuracy'])

    # Convert accuracy to percentage
    data['accuracy'] = data['accuracy'] * 100

    # Pivot the DataFrame
    pivot_table = data.pivot('inputs', 'outputs', 'accuracy')

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a contour plot
    contour = ax.contourf(pivot_table.columns, pivot_table.index, pivot_table.values)

    # Add a colorbar
    fig.colorbar(contour, ax=ax, label='Prediction accuracy (%)')

    # Invert y-axis
    #ax.invert_yaxis()

    # Set labels
    ax.set_xlabel('Number of bins for outputs')
    ax.set_ylabel('Number of bins for inputs')
    ax.set_title('Bin-configuration sensitivity analysis')

    # Show the plot
    plt.show()


def plot_sensitivity_analysis(results):#2D heatmap plot of sensitivity analysis
    # Convert dictionary to DataFrame
    data = pd.DataFrame(results.values(), columns=['inputs', 'outputs', 'accuracy'])

    # Convert accuracy to percentage
    data['accuracy'] = data['accuracy'] * 100

    # Pivot the DataFrame
    pivot_table = data.pivot('inputs', 'outputs', 'accuracy')

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a heatmap with a labeled colorbar
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cbar_kws={'label': 'Prediction accuracy (%)'}, ax=ax)

    # Invert y-axis
    ax.invert_yaxis()

    # Set labels
    ax.set_xlabel('Number of bins for outputs')
    ax.set_ylabel('Number of bins for inputs')
    ax.set_title('Bin-configuration sensitivity analysis')

    # Show the plot
    plt.show()


def plot_training(posteriors, edges, priors, inputs, outputs, obs2plot: int, axperrow: int = 5):
    """
    Plot the results of the inference in a figure with the prior and the posteriors

    Parameters
    ----------
    posteriors : dict Posteriors distributions
    edges : array Edges of the bins
    priors : dict Prior distributions
    inputs : list Input variables
    outputs : list Output variables
    obs2plot : int Number of the observation to plot
    axperrow : int Number of axis per row

    Returns
    -------
    ax : figure axis
    """
    # Get the bins parameters for all variables
    binwidth = {}
    bin_centers = {}
    for var in edges:
        binwidth[var] = edges[var][1:]-edges[var][:-1]
        bin_centers[var] = 0.5*(edges[var][1:] + edges[var][:-1])

    # Get the number of axis
    nax = len(posteriors.keys())
    nrow = int(np.ceil(nax/axperrow))
    ncol = axperrow
    
    fig, ax = plt.subplots(nrow, ncol, squeeze = False)
    ax.reshape((nrow, axperrow))
    colnames = list(edges.keys())

    i = 0
    j = 0
    for var in colnames:
        # Plot the prior dist
        ax[i,j].bar(bin_centers[var], priors[var], width = binwidth[var],
                    color = 'grey', alpha = 0.7, linewidth = 0.2, edgecolor = 'black')
        
        # Plot the posterior dist
        # need to plot the target posteriors using equidistant bins to make the plot look nice
        if var in inputs:
            colour = 'green'
        elif var in outputs:
            colour = 'red'
        ax[i,j].bar(bin_centers[var], posteriors[var][obs2plot], width = binwidth[var],
                    color = colour, alpha = 0.5, linewidth = 0.2, edgecolor = 'black')

        # Cosmetics
        ax[i,j].set_xlabel('Ranges')
        ax[i,j].set_ylabel('Probability')
        ax[i,j].grid(True, linestyle = '--', alpha = 0.5)
        ax[i,j].set_facecolor('whitesmoke')
        ax[i,j].set_title(var, fontweight="bold", fontsize = 10)
        ax[i,j].set_ylim([0, 1])

        j += 1
        if j == ncol:
            j = 0
            i += 1

    title = 'Prior and Posterior Distributions for observation: '+str(obs2plot)
    fig.suptitle(title, fontsize = 10)
    plt.show(block = False)
    plt.tight_layout()
    return ax      


def plot_meta(posteriors, edges, priors, inputs, outputs, axperrow: int = 3):
    """
    Plot the results of the meta model inference in a figure showing posteriors from observations

    Parameters
    ----------
    posteriors : dict Posteriors distributions
    edges : array Edges of the bins
    inputs : list Input variables
    outputs : list Output variables

    Returns
    -------
    ax : figure axis
    """
    # Get the bins parameters for all variables
    binwidth = {}
    bin_centers = {}
    for var in edges:
        edges[var] = np.array(edges[var])
        binwidth[var] = edges[var][1:]-edges[var][:-1]
        bin_centers[var] = 0.5*(edges[var][1:] + edges[var][:-1])

    # Get the number of axis
    nax = len(posteriors)
    nrow = int(np.ceil(nax/axperrow))
    ncol = axperrow
    
    fig, ax = plt.subplots(nrow, ncol, squeeze=False)
    ax.reshape((nrow, axperrow))
    colnames = list(edges.keys())

    i = 0
    j = 0
    for var in colnames:

        ax[i, j].bar(bin_centers[var], priors[var], width=binwidth[var],
                     color='grey', alpha=0.7, linewidth=0.2, edgecolor='black')

        # Plot the posterior dist
        # need to plot the target posteriors using equidistant bins to make the plot look nice
        if var in inputs:
            colour = 'green'
        elif var in outputs:
            colour = 'red'
        ax[i, j].bar(bin_centers[var], posteriors[var], width=binwidth[var],
                     color=colour, alpha=0.5, linewidth=0.2, edgecolor='black')

        # Cosmetics
        ax[i, j].grid(True, linestyle='--', alpha=0.5)
        ax[i, j].set_facecolor('whitesmoke')
        ax[i, j].set_title(var, fontweight="bold", fontsize=10)
        ax[i, j].set_ylim([0, 1])
        ax[i, j].xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))

        # Set x-axis label for the bottom row
        if i == nrow - 1:
            ax[i, j].set_xlabel('Ranges')

        # Set y-axis label for the leftmost column
        if j == 0:
            ax[i, j].set_ylabel('Probability')

        j += 1
        if j == ncol:
            j = 0
            i += 1

    title = 'Prior and Posterior Distributions'  #+str(obs2plot)
    fig.suptitle(title, fontsize=10)
    plt.show(block=False)
    plt.tight_layout()