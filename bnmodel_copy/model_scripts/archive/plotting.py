import numpy as np
import matplotlib.pyplot as plt # for drawing graphs
from join_tree_population import structure
from discretisation import bin_edges_dict, prior_dict_xytrn
from generate_posteriors import obs_posteriors_dict, predicted_posteriors_list

# ##This is for the figure parameters.
n_rows = len(predicted_posteriors_list)
n_cols = len(structure.keys()) ##the length of the BN i.e., five nodes

def create_figure(n_rows, n_cols):
    """
    Create a figure with specified number of rows and columns.
    """
    fig = plt.figure(figsize=((200 * n_cols) / 96, (200 * n_rows) / 96), dpi=96, facecolor='white')
    fig.suptitle('Posterior Probabilities', fontsize=8)
    return fig

# def create_subplot(fig, n_rows, n_cols, count, varName, bin_edges_dict, priors_dict, obs_posteriors):
    """
    Create a subplot with specified parameters.
    """
    ax = fig.add_subplot(n_rows, n_cols, count+1)
    ax.set_facecolor("whitesmoke")

    index = bin_edges_dict[varName]
    edge = np.zeros((len(bin_edges_dict.items()), len(index[:])))
    binwidths = np.zeros((len(bin_edges_dict.items()), len(index[:-1])))
    xticksv = np.zeros((len(bin_edges_dict.items()), len(index[:-1])))

    for i in range(len(index)):
        edge[count, i] = index[i]

    for i in range(len(index)-1):
        binwidths[count, i] = (index[i+1] - index[i])
        xticksv[count,i]  = ((index[i+1] - index[i]) / 2.) + index[i]

    dataDict = {}
    
    priorPDs_dict = {}
    for node, posteriors in obs_posteriors.items():
        varName2 = varName[:-5]
    
        if varName2 == node:
            dataDict[node] = posteriors
            if node == 'acceleration':
                print(dataDict[node])

                ax.bar(xticksv[count], np.isscalar(dataDict[node]), align='center', width=binwidths[count], color='red', alpha=0.2, linewidth=0.2)
            elif node == 'mass' or node == 'force':
                ax.bar(xticksv[count], np.isscalar(dataDict[node]), align='center', width=binwidths[count], color='green', alpha=0.2, linewidth=0.2)
    for var2, idx in priors_dict.items():
        varName3 = var2[:-13]
        if varName2 == varName3:
            priorPDs_dict[var2] = [idx[i] for i in range(1, len(idx) + 1)]
            ax.bar(xticksv[count], priorPDs_dict[var2], align='center', width=binwidths[count], color='black', alpha=0.2, linewidth=0.2)

    plt.xlim(min(edge[count]), max(edge[count]))
    plt.xticks([np.round(e, 2) for e in edge[count]], rotation='vertical')
    plt.ylim(0, 1)

    ax.grid(color='0.2', linestyle=':', linewidth=0.1, dash_capstyle='round')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.set_title(varName2, fontweight="bold", size=6)
    ax.set_ylabel('Probabilities', fontsize=7)
    ax.set_xlabel('Ranges', fontsize=7)
    return fig

def create_subplot(fig, n_rows, n_cols, count, varName, bin_edges_dict, priors_dict, obs_posteriors):
    """
    Create a subplot with specified parameters.
    """
    ax = fig.add_subplot(n_rows, n_cols, count+1)
    ax.set_facecolor("whitesmoke")

    index = bin_edges_dict[varName]
    edge = np.zeros((len(bin_edges_dict.items()), len(index[:])))
    binwidths = np.zeros((len(bin_edges_dict.items()), len(index[:-1])))
    xticksv = np.zeros((len(bin_edges_dict.items()), len(index[:-1])))

    for i in range(len(index)):
        edge[count, i] = index[i]

    for i in range(len(index)-1):
        binwidths[count, i] = (index[i+1] - index[i])
        xticksv[count,i]  = ((index[i+1] - index[i]) / 2.) + index[i]

    dataDict = {}
    
    priorPDs_dict = {}
    for node, posteriors in obs_posteriors.items():
        varName2 = varName[:-5]
    
        if varName2 == node:
            dataDict[node] = posteriors
            if node == 'acceleration':
                for value in dataDict[node]:
                    ax.bar(xticksv[count], value, align='center', width=binwidths[count], color='red', alpha=0.2, linewidth=0.2)
            elif node == 'mass' or node == 'force':
                for value in dataDict[node]:
                    ax.bar(xticksv[count], value, align='center', width=binwidths[count], color='green', alpha=0.2, linewidth=0.2)
    for var2, idx in priors_dict.items():
        varName3 = var2[:-7]
        if varName2 == varName3:
            priorPDs_dict[var2] = [idx[i] for i in range(1, len(idx) + 1)]
            for value2 in priorPDs_dict[var2]:
                ax.bar(xticksv[count], value2, align='center', width=binwidths[count], color='black', alpha=0.2, linewidth=0.2)

    plt.xlim(min(edge[count]), max(edge[count]))
    plt.xticks([np.round(e, 2) for e in edge[count]], rotation='vertical')
    plt.ylim(0, 1)

    ax.grid(color='0.2', linestyle=':', linewidth=0.1, dash_capstyle='round')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.set_title(varName2, fontweight="bold", size=6)
    ax.set_ylabel('Probabilities', fontsize=7)
    ax.set_xlabel('Ranges', fontsize=7)
    return fig


def create_all_subplots(fig, n_rows, n_cols, bin_edges_dict, priors_dict, obs_posteriors):
    """
    Create all subplots for the given figure.
    """
    count = 0
    for varName in bin_edges_dict:
        fig = create_subplot(fig, n_rows, n_cols, count, varName, bin_edges_dict, priors_dict, obs_posteriors)
        count += 1
    return fig

def plot_posterior_probabilities(n_rows, n_cols, bin_edges_dict, priors_dict, obs_posteriors, plot):
    """
    Plot posterior probabilities for each variable in bin_edges_dict.
    """
    fig = create_figure(n_rows, n_cols)
    fig = create_all_subplots(fig, n_rows, n_cols, bin_edges_dict, priors_dict, obs_posteriors)
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    if plot == True:
        plt.show(block=False)   

plot_posterior_probabilities(n_rows, n_cols, bin_edges_dict, prior_dict_xytrn, obs_posteriors_dict, plot=True) 


