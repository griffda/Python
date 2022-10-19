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

def BNstructure(df, df_binned, labels):
    ##Create new data frame - call it binned and fill with the values and then use structure syntax below. 

    bin_edges_dict = {}

    ###This is a more modular approach which uses qcut and a for loop to loop through each of the columns for the nodes
    ###providing bins, and calculating bin edges.
    for name in df.columns:
        name_bins = name + '_bins'
        df[name_bins], bin_edges = pd.qcut(df[name], 4, labels=labels, retbins=True)
        bin_edges_dict[name_bins]=bin_edges

    df_binned = df.drop(['m', 'theta','v0', 'vf', 'KE'], axis=1)
    ###Pybbn only reads data types as strings, so this line converts the data in the csv from int64 to string 
    df_binned = df_binned.applymap(str)
    #print(df_binned.head(10))

    ###This is telling us how the network is structured between parent nodes and posteriors. 
    ###Using this method, we avoid having to build our own conditional probability tables using maximum likelihood estimation. 
    structure = {
        'm_bins':[],
        'theta_bins': [],
        'v0_bins':[],
        'vf_bins': ['theta_bins', 'v0_bins', 'm_bins'],
        'KE_bins': ['theta_bins', 'v0_bins', 'm_bins']
    }

    ###Here we are calling the Factory function from pybbn and applying the above structure and data frame from csv as arguments. 
    bbn = Factory.from_data(structure, df_binned)

    ###this line performs inference on the data 
    join_tree = InferenceController.apply(bbn)

    return join_tree, structure

def plotPDs(maintitle, xlabel, ylabel, displayplt = False): # plots the probability distributions)
    
    structure, join_tree = BNstructure(df=[], df_binned=[], labels=[])###These arguments would need to be filled. 

    bin_edges_dict = {} ### creates empty dict 
    
    ###Code to automatically set the number of columns and rows and dimensions of the figure
    n_rows = 1
    n_cols = len(structure.keys()) ##the length of the BN i.e., five nodes


    ###instantiate a figure as a placaholder for each distribution (axes)
    fig = plt.figure(figsize=((200 * n_cols) / 96, (200 * n_rows) / 96), dpi=96, facecolor='white')
    fig.suptitle(maintitle, fontsize=8) ###Replaced 'Posterior Probabilities with maintitle

    ###Creating an array of zeros that will be filled with values when going through the dictionary loop below
    edge = np.zeros((len(bin_edges_dict.items()), len(list(bin_edges_dict.items())[:-1])))
    binwidths = np.zeros((len(bin_edges_dict.items()), len(list(bin_edges_dict.items())[:-1])))
    xticksv = np.zeros((len(bin_edges_dict.items()), len(list(bin_edges_dict.items())[:-1]))) 


    ###Loop goes through a dictionary which contains a key and a value
    ###The first variable i.e., node will correspond to the key and the second i.e., posteriors, will correspond to the value. 
    for node, posteriors in join_tree.get_posteriors().items():
        p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])

    ###This creates a variable that corresponds to key (varname) and another variable which corresponds to the value (index)
    for varName, index in bin_edges_dict.items():

        ax = fig.add_subplot(n_rows, n_cols, count+1) ###subplot with three arguments taken from above, including count
        ax.set_facecolor("whitesmoke") ###sets the background colour of subplot

        for i in range(len(index)-1): ###This for loop find the edges, binwidths and midpoints (xticksv) for each of the bins in the dict
            edge[count, i] = index[i]
            binwidths[count, i] = (index[i+1] - index[i])
            xticksv[count,i]  = ((index[i+1] - index[i]) / 2.) + index[i]
            
        ###This line plots the bars using xticks on x axis, probabilities on the y and binwidths as bar widths. 
        ###It counts through them for every loop within the outer for loop 
        ax.bar(xticksv[count], posteriors.values(), align='center', width=binwidths[count], color='black', alpha=0.2, linewidth=0.2)

        ###These lines plot the limits of each axis. 
        plt.xlim(min(edge[count]), max(edge[count]))
        plt.xticks([np.round(e, 4) for e in edge[count]], rotation='vertical')
        plt.ylim(0, 1) 

        ###These lines set labels and formatting style for the plots. 
        ax.grid(color='0.2', linestyle=':', linewidth=0.1, dash_capstyle='round')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.set_title(varName, fontweight="bold", size=6)
        ax.set_ylabel(ylabel, fontsize=7)  # Y label
        ax.set_xlabel(xlabel, fontsize=7)  # X label

        count+=1

    fig.tight_layout()  # Improves appearance a bit.
    fig.subplots_adjust(top=0.85)  # white spacing between plots and title   
    #plt.show()
    if displayplt == True: plt.show()

    