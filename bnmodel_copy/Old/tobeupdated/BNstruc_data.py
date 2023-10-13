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
- posterior probabilities will be calculated AND PLOTTED
"""
import pandas as pd
from pybbn.graph.factory import Factory
from pybbn.pptc.inferencecontroller import InferenceController
import numpy as np
import matplotlib.pyplot as plt # for drawing graphs

df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/output2.csv',
                 index_col=False,
                 usecols=['m', 'theta','v0', 'vf', 'KE'],
                 encoding=('utf-8')
                 )

##Create new data frame - call it binned and fill with the values and then use structure syntax below. 

labels = [1,2,3,4]
labels2 = [1,2,3,4,5,6]

bin_edges_dict = {}

###Equidistant binning for the inputs 
for name in df.iloc[:,[0,1,2]]:
    name_bins = name + '_bins'
    df[name_bins], bin_edges = pd.cut(df[name], 4, labels=labels, retbins=True)
    bin_edges_dict[name_bins]=bin_edges
    

###Percentile binning for the outputs
for name in df.iloc[:,[3,4]]:
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
    'vf_bins': ['theta_bins', 'v0_bins'],
    'KE_bins': ['vf_bins', 'm_bins']
}

###Here we are calling the Factory function from pybbn and applying the above structure and data frame from csv as arguments. 
bbn = Factory.from_data(structure, df_binned)

###this line performs inference on the data 
join_tree = InferenceController.apply(bbn) 

###Try to fill an empty array with the posteriors and plot them from there. 
p=[]
arr = []

##This is for the figure parameters. 
n_rows = 1
n_cols = len(structure.keys()) ##the length of the BN i.e., five nodes

###instantiate a figure as a placaholder for each distribution (axes)
fig = plt.figure(figsize=((200 * n_cols) / 96, (200 * n_rows) / 96), dpi=96, facecolor='white')
fig.suptitle('Posterior Probabilities', fontsize=8) # title


i = 0

###Creating an array of zeros that will be filled with values when going through the dictionary loop below
edge = np.zeros((len(bin_edges_dict.items()), len(list(bin_edges_dict.items())[:-1])))
binwidths = np.zeros((len(bin_edges_dict.items()), len(list(bin_edges_dict.items())[:-1])))
xticksv = np.zeros((len(bin_edges_dict.items()), len(list(bin_edges_dict.items())[:-1]))) 

count = 0

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
    ax.set_ylabel('Probabilities', fontsize=7)  # Y label
    ax.set_xlabel('Ranges', fontsize=7)  # X label

    count+=1
    
fig.tight_layout()  # Improves appearance a bit.
fig.subplots_adjust(top=0.85)  # white spacing between plots and title   
plt.show()



