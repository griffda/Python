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
from pybbn.graph.jointree import EvidenceBuilder
import numpy as np
import matplotlib.pyplot as plt # for drawing graphs

df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/output2.csv',
                 index_col=False,
                 usecols=['m', 'theta','v0', 'vf', 'KE'],
                 encoding=('utf-8')
                 )

##Create new data frame - call it binned and fill with the values and then use structure syntax below. 

labels = [1,2,3,4]
labels2 = [1,2,3,4,5]

bin_edges_dict = {}
prior_dict = {}

###Equidistant binning for the inputs 
for name in df.iloc[:,[0,1,2]]:
    name_bins = name + '_bins'
    df[name_bins], bin_edges = pd.cut(df[name], 4, labels=labels, retbins=True)
    bin_edges_dict[name_bins]=bin_edges
    
###This is storing the priorPDs so we can plot them
    name_priors = name + '_priors'
    prior = df[name_bins].value_counts(normalize=True).sort_index()
    priorPDs = prior.to_dict()
    prior_dict[name_priors] = priorPDs
    #print(prior_dict)
      

###Percentile binning for the outputs
for name in df.iloc[:,[3,4]]:
    name_bins = name + '_bins'
    df[name_bins], bin_edges = pd.qcut(df[name], 4, labels=labels, retbins=True)
    bin_edges_dict[name_bins]=bin_edges

###This is storing the priorPDs so we can plot them
    name_priors = name + '_priors'
    prior = df[name_bins].value_counts(normalize=True).sort_index()
    priorPDs = prior.to_dict()
    prior_dict[name_priors] = priorPDs
    #print(prior_dict)

df_binned = df.drop(['m', 'theta','v0', 'vf', 'KE'], axis=1)


###Pybbn only reads data types as strings, so this line converts the data in the csv from int64 to string 
df_binned = df_binned.applymap(str)


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

###inserting observation evidence
###This is saying that if there is evidence submitted in the arguments for the function to fill evidenceVars with that evidence
###e.g., evidence = {'deflection':[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'weight':[1.0, 0.0, 0.0, 0.0, 0.0, 0.0] }
###where this is a dictionary with a list of bin ranges, setting hard probability of one bin to 100%. 
#evidenceVars = {'v0_bins':[1.0, 0.0, 0.0, 0.0]}
#evidenceVars = {'m_bins':[1.0, 0.0, 0.0, 0.0]}
evidenceVars = {'theta_bins':[1.0, 0.0, 0.0, 0.0]}

###This inserts observation evidence.
###This needs to be plotted as it's own plot and then superimposed onto the marginal probability distributions you already have. 
for bbn_evid in evidenceVars:
    ev = EvidenceBuilder() \
        .with_node(join_tree.get_bbn_node_by_name(bbn_evid)) \
        .with_evidence('1', 1.0) \
        .build()
    join_tree.set_observation(ev)


###Try to fill an empty array with the posteriors and plot them from there. 
p=[]
arr = []

##This is for the figure parameters. 
n_rows = 1
n_cols = len(structure.keys()) ##the length of the BN i.e., five nodes

###instantiate a figure as a placaholder for each distribution (axes)
fig = plt.figure(figsize=((200 * n_cols) / 96, (200 * n_rows) / 96), dpi=96, facecolor='white')
fig.suptitle('Posterior Probabilities', fontsize=8) # title

###Instantiate a counter
i = 0
count = 0

###Creating an array of zeros that will be filled with values when going through the dictionary loop below
edge = np.zeros((len(bin_edges_dict.items()), len(list(bin_edges_dict.items())[:-1])))
binwidths = np.zeros((len(bin_edges_dict.items()), len(list(bin_edges_dict.items())[:-1])))
xticksv = np.zeros((len(bin_edges_dict.items()), len(list(bin_edges_dict.items())[:-1]))) 

###Loop goes through a dictionary which contains a key and a value
###The first variable i.e., node will correspond to the key and the second i.e., posteriors, will correspond to the value. 


    #print(posteriors) ###this a dictionary within the list
    
    ####make fict ordered for plotting.


###This is to plot the priorPDs that we stored above.
priorPDs_dict = {}

for var2, idx in prior_dict.items():
    priorPDs_dict[var2] = list(idx.values())


###This creates a variable that corresponds to key (varname) and another variable which corresponds to the value (index)
for varName, index in bin_edges_dict.items():

    ax = fig.add_subplot(n_rows, n_cols, count+1) ###subplot with three arguments taken from above, including count
    ax.set_facecolor("whitesmoke") ###sets the background colour of subplot

    dataDict = {}

    ###Loop goes through a dictionary which contains a key and a value
    ##The first variable i.e., node will correspond to the key and the second i.e., posteriors, will correspond to the value.      
    for node, posteriors in join_tree.get_posteriors().items(): ### this is a list of dictionaries 
        p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
        #print(posteriors) #So Dict(str,List[float]) and Dict(str,Dict(str,float))
        #print(node) Can you make the second dict into the same data types as the first
        dataDict[node] = list(posteriors.values())
                

    for i in range(len(index)-1): ###This for loop find the edges, binwidths and midpoints (xticksv) for each of the bins in the dict
        edge[count, i] = index[i]
        binwidths[count, i] = (index[i+1] - index[i])
        xticksv[count,i]  = ((index[i+1] - index[i]) / 2.) + index[i]

    ###this is saying: if there is posteriorPD in the arguments then also ask if there is evidence. 
    ###if there is evidence, then plot the posteriorPD as green superimposed on the orginial plot. 
    ###if there is no evidence then to plot the posteriorPD as red superimposed on the green plot. 
    # if 'posteriorPD' in kwargs:

    #     if len(kwargs['posteriorPD'][varName]) > 1:
    #         if varName in evidenceVars:
    #             ax.bar(xticksv, kwargs['posteriorPD'][varName], align='center', width=binwidths, color='green', alpha=0.2, linewidth=0.2)

    #         else:
    #             ax.bar(xticksv, kwargs['posteriorPD'][varName], align='center', width=binwidths, color='red', alpha=0.2, linewidth=0.2)
    
    ###This line plots the bars using xticks on x axis, probabilities on the y and binwidths as bar widths. 
    ###It counts through them for every loop within the outer for loop 
    ###posteriotrs.values() is a dict and therefore not ordered, so need to a way to make it ordered for future use. 
    ###This line plots the priorPDs that we stored in the forloop above. 
    ax.bar(xticksv[count], priorPDs_dict[var2], align='center', width=binwidths[count], color='black', alpha=0.2, linewidth=0.2)
   
    ###sorting evidence variables to be in the beginning of the list and then plotting evidence if it exists. 
    for key in evidenceVars:
        if varName == key:
            for idx, x in enumerate(evidenceVars[varName]):
                if x == 1:
                    bin_index = idx
            ax.bar(xticksv[count][bin_index], evidenceVars[varName][bin_index], align='center', width=binwidths[count][bin_index], color='green', alpha=0.2, linewidth=0.2)
        else: 
            ax.bar(xticksv[count], dataDict[node], align='center', width=binwidths[count], color='red', alpha=0.2, linewidth=0.2)    

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
