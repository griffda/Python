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
from bnmodel.graph.factory import Factory
from bnmodel.pptc.inferencecontroller import InferenceController
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

###When building network you can get the names of the nodes using bbn.nodes or something to that affect. 

###Try to fill an empty array with the posteriors and plot them from there. 
p=[]
arr = []

##This is for the figure parameters. 
n_rows = 1
n_cols = len(structure.keys()) ##the length of the BN i.e., five nodes

###instantiate a figure as a placaholder for each distribution (axes)
fig = plt.figure(figsize=((200 * n_cols) / 96, (200 * n_rows) / 96), dpi=96, facecolor='white')
fig.suptitle('Posterior Probabilities', fontsize=8) # title

###This is a function that enables indexing through a dictionary. 
# def get_nth_key(dictionary, n=0):
#     if n < 0:
#         n += len(dictionary)
#     for i, key in enumerate(dictionary.keys()):
#         if i == n:
#             return key
#     raise IndexError("dictionary index out of range") 

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

# print(edge)
# print(binwidths)


###THESE ARE VARIOUS OTHER WAYS YOU COULD FOR LOOP TO FIND XTICKSV, EDGES AND BINWIDTHS, DEPENDING ON WHAT DATA TYPE YOU HAVE
###this makes a list of ranges 
# for i in range(len(bin_edges_dict.values()-range[1])):
#     varName = bin_edges_dict.keys()
#     print(varName, range)
#     bin_edges_dict.values()
#     edge.append(range[0]) 
#     binwidths.append(bin_edges_dict.values()[i+1] - bin_edges_dict.values()[i])
#     xticksv.append(((bin_edges_dict.values()[i+1] - bin_edges_dict.values()[i]) / 2) + bin_edges_dict.items()[i])
#     if varName == len(bin_edges_dict[varName]) - 1: edge.append(bin_edges_dict.items()[i+1]) 

#i = 0

# for i in range(len(bin_edges_dict)):
#     edge.append(i) 
#     binwidths.append(bin_edges_dict.values()[i+1] - bin_edges_dict.values()[i])
#     xticksv.append(((bin_edges_dict.values()[i+1] - bin_edges_dict.values()[i]) / 2) + bin_edges_dict.items()[i])

# edge.append(range[0]) 
# binwidths.append(range[1] - range[0])
# xticksv.append(((range[1] - range[0]) / 2) + range[0])
# if varName == len(bin_edges_dict[varName]) - 1: edge.append(range[1])

# for varName, range in enumerate(bin_edges[:]):
#     print(bin_edges[:])
    # edge.append(range[0])
    # binwidths.append(range[1] - range[0])
    # xticksv.append(((range[1] - range[0]) / 2) + range[0])
    # if varName == len(bin_edges[varName]) - 1: edge.append(range[1])


