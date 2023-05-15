import numpy as np
import matplotlib.pyplot as plt 
from generate_posteriors import obs_posteriors_dict, predicted_posteriors_list
from discretisation import bin_edges_dict, prior_dict_xytrn 

"""
Plot results comparisson the inputs and the posteriors distributions
"""


def plot_results(posteriors, edges, prior_dist):
    """
    TODO: To be filled
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

    
plot_results(obs_posteriors_dict, bin_edges_dict, prior_dict_xytrn)
