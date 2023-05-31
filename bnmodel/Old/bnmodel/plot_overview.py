import numpy as np
import matplotlib.pyplot as plt 

"""
Plot results comparing the inputs and the posteriors distributions

by @griffda and @jhidalgosalaverri
"""


def plot_results(posteriors, edges, priors, inputs, outputs, obs2plot: int, axperrow: int = 3):
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
        if var in inputs:
            colour = 'green'
        elif var in outputs:
            colour = 'red'
        ax[i,j].bar(bin_centers[var], posteriors[var][obs2plot], width = binwidth[var],
                    color = colour, alpha = 0.7, linewidth = 0.2, edgecolor = 'black')
        


        # Cosmetics
        ax[i,j].set_xlabel('Ranges')
        ax[i,j].set_ylabel('Probability')
        ax[i,j].set_title(var, fontweight="bold", fontsize = 10)



        j += 1
        if j == ncol:
            j = 0
            i += 1

    title = 'Prior and Posterior Distributions for observation: '+str(obs2plot)
    fig.suptitle(title, fontsize = 10)
    plt.show(block = False)
    plt.tight_layout()
    return ax      

