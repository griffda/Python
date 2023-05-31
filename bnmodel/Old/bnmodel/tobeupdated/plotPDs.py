"""
Created on Weds Oct 19th 15:20 2022

@author: tomgriffiths

This script contains a function that will plot posterior probabilities for different BN nodes. 
 
- This is an update and modularised version of struc_data_mod
"""
import pandas as pd
from pybbn.graph.factory import Factory
from pybbn.pptc.inferencecontroller import InferenceController
import numpy as np
import matplotlib.pyplot as plt # for drawing graphs

###can you put this plotting code into a function where the arguments are the x-axis values for 'xticks' and the y-axis values are the probability distribution given by 'dataDict'


def create_figure(n_rows, n_cols):
    fig = plt.figure(figsize=((200 * n_cols) / 96, (200 * n_rows) / 96), dpi=96, facecolor='white')
    fig.suptitle('Posterior Probabilities', fontsize=8)
    return fig


def plot_posterior(ax, varName, index, join_tree):
    ax.set_facecolor("whitesmoke")

    edge = np.zeros((len(index), len(index[:])))
    binwidths = np.zeros((len(index), len(index[:-1])))
    xticksv = np.zeros((len(index), len(index[:-1])))

    for i in range(len(index)):
        edge[i, :] = index[i]

    for i in range(len(index)-1):
        binwidths[i, :] = (index[i+1] - index[i])
        xticksv[i, :] = ((index[i+1] - index[i]) / 2.) + index[i]

    dataDict = {}
    for node, posteriors in join_tree.get_posteriors().items():
        p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
        if varName == node:
            dataDict[node] = list(posteriors.values())
            if varName == 'acceleration_bins':
                ax.bar(xticksv, dataDict[node], align='center', width=binwidths, color='red', alpha=0.2, linewidth=0.2)
            elif varName == 'mass_bins' or 'force_bins':
                ax.bar(xticksv, dataDict[node], align='center', width=binwidths, color='green', alpha=0.2, linewidth=0.2)

    ax.set_xlim(min(edge), max(edge))
    ax.set_xticks([np.round(e, 2) for e in edge], rotation='vertical')
    ax.set_ylim(0, 1)

    ax.grid(color='0.2', linestyle=':', linewidth=0.1, dash_capstyle='round')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.set_title(varName, fontweight="bold", size=6)
    ax.set_ylabel('Probabilities', fontsize=7)
    ax.set_xlabel('Ranges', fontsize=7)


def plot_all_posteriors(structure, bin_edges_dict, join_tree):
    n_rows = 1
    n_cols = len(structure.keys())

    fig = create_figure(n_rows, n_cols)
    count = 0
    for varName, index in bin_edges_dict.items():
        ax = fig.add_subplot(n_rows, n_cols, count+1)
        plot_posterior(ax, varName, index, join_tree)
        count += 1

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    plt.show()
