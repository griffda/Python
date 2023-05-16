import numpy as np
import matplotlib.pyplot as plt 

"""
Plot results comparing the inputs and the posteriors distributions

by @griffda and @jhidalgosalaverri
15/05/2023

"""


def plot_results(posteriors, edges, prior_dist, inputs, outputs):
    """
    Plot the results of the inference in a figure with the prior and the posteriors

    Parameters
    ----------
    posteriors : dict Posteriors distributions
    edges : array Edges of the bins
    prior_dist : dict Prior distributions
    inputs : list Input variables
    outputs : list Output variables

    Returns
    -------
    ax : figure axis
    """

    nrow = len(posteriors[inputs[0]]) # TODO: this may fail if there's a single input
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
                        color = 'grey', alpha = 0.7, linewidth = 0.2, edgecolor = 'black')
            j += 1

    # Plot the posteriors  
    for i in range(nrow):
        j = 0
        for var in colnames:
            if var in outputs:
                colour = 'red'
            elif var in inputs:
                colour = 'green'
            ax[i,j].bar(bin_centers[var], posteriors[var][i], width = binwidth[var],
                            color = colour, alpha = 0.2, linewidth = 0.2, edgecolor = 'black')
            j += 1

    # Cosmetics
    fig.suptitle('Prior and Posterior Distributions', fontsize = 10)

    for i in range(nrow-1):
        j = 0
        for var in edges:
            ax[i,j].set_xticks(edges[var], [np.round(e, 2) for e in edges[var]])
            ax[i,j].axes.xaxis.set_ticklabels([])
            # ax[i,j].xaxis.set_tickparams(labelbottom = False)
            j += 1

    j = 0
    for var in edges:
        ax[-1,j].set_xlim(min(edges[var]), max(edges[var]))
        ax[-1,j].set_ylim(0, 1)
        ax[-1,j].set_xticks(edges[var], [np.round(e, 2) for e in edges[var]],
            rotation = 'vertical')
        ax[-1,j].set_xlabel('Ranges')
        j += 1
    
    j = 0    
    for var in edges:
        ax[0,j].set_title(var, fontweight="bold", fontsize = 10)
        j += 1

    for i in range(nrow):
        j = 0
        for var in edges:
            ax[i,j].set_ylabel('Probability')
            ax[i,j].axes.grid(True, linestyle = '--', alpha = 0.5) 
            j += 1       

    plt.show(block = False)
    plt.tight_layout()
    # fig.subplots_adjust(hspace=0.18)
    # fig.subplots_adjust(hspace=0.25)


    return ax      

