"""
Created on Fri Aug 12 13:58:57 2022

@author: tomgriffiths

In this script we will attepmt to create a BN using the library pyBBN 
for the BOS model.

BBNs structure is updated: 
- input nodes: theta, m, v0
- output nodes: vf, KE.
- Larger data set of 500 rows.
- Data will use preset bin widths and will test how changing this affects the outputs.
- posterior probabilities will be calculated.
- posterior probabilities will be plotted. 
"""

import pandas as pd
from pybbn.graph.factory import Factory
from pybbn.pptc.inferencecontroller import InferenceController
import matplotlib.pyplot as plt # for drawing graphs
import numpy as np

df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/output2.csv',
                 index_col=False,
                 usecols=['m', 'theta','v0', 'vf', 'KE'],
                 encoding=('utf-8')
                 )

##Create new data frame - call it binned and fill with the values and then use structure syntax below. 

labels = [1,2,3,4]
labels2 = [1,2,3,4,5,6]

bin_edges_dict = {}

###This is a more modular approach which uses qcut and a for loop to loop through each of the columns for the nodes
###providing bins, and calculating bin edges.
for name in df.columns:
    name_bins = name + '_bins'
    df[name_bins], bin_edges = pd.qcut(df[name], 4, labels=labels, retbins=True)
    bin_edges_dict [name_bins]=bin_edges
 
###Pybbn only reads data types as strings, so this line converts the data in the csv from int64 to string 
df = df.applymap(str)

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
bbn = Factory.from_data(structure, df)

###
print(structure.keys())

###this line performs inference on the data 
join_tree = InferenceController.apply(bbn)

p = []

###Function to calculate the posterior probabilities and put into a dataframe/array/dictionary/f-string
def calculate_posteriors():
    for node, posteriors in join_tree.get_posteriors().items():
        ###This is useful to display them as f-strings 
        # p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
        # print(p)
        
        ###This method passes the key value pairs. 
        p = posteriors
        df = pd.DataFrame(p.items(), columns = [node, 'Posterior Probability'])
        ###These are the posterior probabilities
        b = df.iloc[:,1].values
        ###these are the bin values 
        c = df.iloc[:,0].values
        print(b)
        print(c)

        ####This method passes the Series constructor
        s = pd.Series(p, name='Posterior Probability')
        s.index.name = 'Bins'
        s.reset_index()
        #print(s)
    return p, b ,c, node

###Function to draw a barchart
def draw_barchartpd(xlabel, ylabel, binranges, probabilities):

    """
    combined =[]
    for range in binranges:
        for val in range:
            combined.append(val)
    # Convert to a set and back into a list.
    print combined
    sett = set(combined)
    xticks = list(sett)
    xticks.sort()
    """

    xticksv = []
    widths = []
    edge = []

    #edge.append(binranges[0][len(binranges[0])-1])
    for index, range in enumerate(binranges):
        edge.append(range[0])
        widths.append(range[1]-range[0])
        xticksv.append(((range[1]-range[0])/2)+range[0])
        if index ==len(binranges)-1: edge.append(range[1])

    print('xticks '), xticksv
    print('probabilities '), probabilities
    print('edge '), edge

    d = plt.bar(xticksv, probabilities, align='center', width=widths, color='black', alpha=0.2)

    #plt.bar(xticksv, posterior, align='center', width=widths, color='red', alpha=0.2)
    #plt.xlim(edge[0], max(edge))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(edge)
    plt.ylim(0, 1)
    plt.show()

    return d
# p, b ,c, node = calculate_posteriors()    
# draw_barchartpd('Ranges', 'Probabilities',c ,b)

### This is a function to calculate the posterior probabilities and then plot them. 
def plotPDs(pybbn, maintitle, xlabel, ylabel, displayplt = False, **kwargs):


    # ###code to automatically set the number of columns and rows and dimensions of the figure
    # jt = JoinTree()
    # n_totalplots = len(structure)
    # n_totalplots = jt.nodes

    # if n_totalplots <= 4:
    #     n_cols = n_totalplots
    #     n_rows = 1

    # else: 
    #     n_cols = 4
    #     n_rows = n_totalplots % 4
    # if n_rows == 0: n_rows = n_totalplots/4

    ###change this to the length or number of nodes
    n_rows = 1
    n_cols = len(structure.keys())


    priorsPDs = {}

    # instantiate a figure as a placaholder for each distribution (axes)
    fig=plt.figure(figsize=((750*n_cols)/220, (750*n_rows)/220  ), dpi=220)
    t = fig.suptitle(maintitle, fontsize=4)

    i = 0 

    binwidths = []

    ###This is for drawing histograms. 
    p, b ,c, node = calculate_posteriors()

    ###THIS IS THE CODE THAT NEEDS CHANGING 
    for var_name in list(structure.keys()):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        ax.set_facecolor("whitesmoke") 
        
        if isinstance(binwidths, int) == True:
            minv = min(df['posteriorPD'])
            maxv = max(df['posteriorPD'])
            df[var_name].hist(bins=np.arange(minv, maxv + binwidths, binwidths),ax=ax)
            #df[var_name].hist(bins=binwidths[var_name],ax=ax, color='black')

        # ###This is meant to consider evidence. 
        # if 'posteriorPD' in kwargs:
    
        #     if len(kwargs['posteriorPD']) > 1:
        #             if node in join_tree.get_posteriors().values():
        #                 ax.bar(xticksv, kwargs['posteriorPD'][var_name], align='center', width=binwidths, color='green', alpha=0.2, linewidth=0.2)

        #             else:
        #                 ax.bar(xticksv, kwargs['posteriorPD'][var_name], align='center', width=binwidths, color='red', alpha=0.2, linewidth=0.2) 

     
        plt.xticks([round(e, 4) for e in b], rotation='vertical')
        plt.ylim(0, 1)

    for spine in ax.spines:
        ax.spines[spine].set_linewidth(0)

    ax.grid(color='0.2', linestyle=':', linewidth=0.1, dash_capstyle='round')
    # ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.set_title(node, fontweight="bold", size=6)
    ax.set_ylabel(ylabel, fontsize=7)  # Y label
    ax.set_xlabel(xlabel, fontsize=7)  # X label
    ax.xaxis.set_tick_params(labelsize=6, length =0)
    ax.yaxis.set_tick_params(labelsize=6, length = 0)

    i += 1
    
    if displayplt == True: plt.show()
    return node, p

#plotPDs('Posteriors', 'Ranges', 'Probabilities', displayplt=True, posteriorPD=p)



