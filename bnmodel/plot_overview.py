import numpy as np
import matplotlib.pyplot as plt 


"""
Plot results comparing the inputs and the posteriors distributions

by @griffda and @jhidalgosalaverri
15/05/2023

"""


def plot_results(posteriors, edges, prior_dist):
    """
    Plot the results of the inference in a figure with the prior and the posteriors

    Parameters
    ----------
    posteriors : dict
        Dictionary with the posteriors distributions
    edges : array
        Array with the edges of the bins
    prior_dist : dict
        Dictionary with the prior distributions

    Returns
    -------
    ax : figure axis

    """

    nrow = len(posteriors['mass']) # TODO: remove the hardcoded
    ncol = len(posteriors.keys())

    binwidth = {}
    bin_centers = {}

    # Get the bins parameters for all variables
    for var in edges:
        binwidth[var] = edges[var][1:]-edges[var][:-1]
        bin_centers[var] = 0.5*(edges[var][1:] + edges[var][:-1])
    
    fig, ax = plt.subplots(nrow, ncol)
    ax = ax.reshape(nrow, ncol)
    colnames = list(edges.keys())
    
    # Plot the prior dist
    for i in range(nrow):
        j = 0
        for var in colnames:
            ax[i,j].bar(bin_centers[var], prior_dist[var], width = binwidth[var], 
                        color = 'grey', alpha = 0.2, linewidth = 0.2)
            j += 1

    # Plot the posteriors    
    for i in range(nrow):
        j = 0
        for var in colnames:
            if var == 'acceleration':
                colour = 'red'
            elif var == 'force' or var == 'mass':
                colour = 'green'
            ax[i,j].bar(bin_centers[var], posteriors[var][i], width = binwidth[var],
                            color = colour, alpha = 0.2, linewidth = 0.2)
            j += 1

    # Cosmetics
    for i in range(nrow):
        j = 0
        for var in edges:
            ax[i,j].set_xlabel(var)
            ax[i,j].set_xlim(min(edges[var]), max(edges[var]))
            ax[i,j].set_ylim(0, 1)
            ax[i,j].set_xticks(edges[var], [np.round(e, 2) for e in edges[var]],
                rotation = 'vertical')
            j += 1
    plt.show(block = False)
    plt.tight_layout()

    return ax    

