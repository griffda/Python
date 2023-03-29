"""
Created on Wed Nov 16 13:58:57 2022

@author: tomgriffiths

In this script we will attepmt to create a BN using the library pyBBN 
for f = ma linear model.

BBNs structure is updated: 
- Input nodes: f, m 
- Output nodes: a.
- Larger data set of 500 rows.
- Data will use preset bin widths and will test how changing this affects the outputs.
- priorPDs will be plotted.
- posterior probabilities will be calculated.
- evidence can be supplied to inform on posterior probabilities.
- evidence, posteriors, and priorPDs will all be super-imposed on each other for same plot. 

- Each step followed in the script will follow flow diagram from Zack Quereb Conti thesis on page 71.  
- Steps: have all been carried out in other scripts. 
    1a (select design ranges), 
    1b (generate input samples), 
    1c (run analytical model i.e., f=ma) 


--THIS SCRIPT SHOULD FIX THE ISSUE OF PLOTTING -1 NUMBER OF BINS USED
"""
import pandas as pd
import inspect
from pybbn.graph.factory import Factory
from pybbn.pptc.inferencecontroller import InferenceController
from pybbn.graph.jointree import EvidenceBuilder
import numpy as np
import matplotlib.pyplot as plt # for drawing graphs
from sklearn.model_selection import train_test_split


###Steps 2a (store inoput values), and 2b (store response values) into csv columns. 
###This loads csv into a dataframe to be manipulated. 
df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/outputv3.csv',
                 index_col=False,
                 usecols=['mass', 'force','acceleration'],
                 encoding=('utf-8')
                 )

###This is getting the input data from the model
x_df = df.iloc[:,[0,1]]
# print(x_df.head())

###This is getting the output data from the model i.e., the target.
y_df = df.iloc[:,[2]]
# print(y_df.head())

###Step 3 is to split data into training and testing sets. 
###Here the data is split 50/50
###Target (i.e., y_data are the outputs from the model)
x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.5)
# x_train, x_test = train_test_split(df, test_size=0.5)

###Bin labels: 
labels = [1,2,3]
# labels = [1,2,3,4]
# labels = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
# labels = [1,2,3,4,5,6,7,8,9,10]
# labels2 = [1,2,3,4,5,6,7,8]

number_of_bins = 3

###Empty dicts to fill 
bin_edges_dict = {}
prior_dict = {}


###Step 4a 
###Replace altl 'df' with 'x_train' to ensure sampling from training data
###Equidistant binning for the inputs 
for name in x_train:
    name_bins = name + '_bins'
    x_train[name_bins], bin_edges = pd.cut(x_train[name], number_of_bins, labels=labels, retbins=True)
    bin_edges_dict[name_bins]=bin_edges
    # print(bin_edges_dict.items())
    
###This is storing the priorPDs so we can plot them
    name_priors = name + '_priors'
    prior = x_train[name_bins].value_counts(normalize=True).sort_index()
    priorPDs = prior.to_dict()
    prior_dict[name_priors] = priorPDs
    # print(prior_dict.items())
      
###Percentile binning for the outputs
###Step 4b
for name in y_train:
    name_bins = name + '_bins'
    y_train[name_bins], bin_edges = pd.qcut(y_train[name], number_of_bins, labels=labels, retbins=True)
    # y_train[name_bins], bin_edges = pd.cut(y_train[name], 4, labels=labels, retbins=True)
    bin_edges_dict[name_bins]=bin_edges
    # for i in range(len(y_train[name]-1)):
    print(bin_edges_dict.items())
    

###This is storing the priorPDs so we can plot them
    name_priors = name + '_priors'
    prior = y_train[name_bins].value_counts(normalize=True).sort_index()
    priorPDs = prior.to_dict()
    prior_dict[name_priors] = priorPDs
    print(prior_dict)


###Must join x_train and y_train and pass to the BN:
# df_binned = x_train.drop(['m', 'theta','v0', 'vf', 'KE'], axis=1)
df_binned = x_train.drop(['force', 'mass'], axis=1)
df_binned_y = y_train.drop(['acceleration'], axis=1)
df_binned_xy = pd.concat([df_binned, df_binned_y], axis=1)

###Pybbn only reads data types as strings, so this line converts the data in the csv from int64 to string 
df_binned = df_binned.applymap(str)
df_binned_y = df_binned_y.applymap(str)
df_binned_xy = df_binned_xy.applymap(str)

###Step 5
###This is telling us how the network is structured between parent nodes and posteriors. 
###Using this method, we avoid having to build our own conditional probability tables using maximum likelihood estimation. 
structure = {
    'mass_bins':[],
    'force_bins': [],
    'acceleration_bins': ['force_bins', 'mass_bins']
}

###Step 5
##Here we are calling the Factory function from pybbn and applying the above structure and data frame from csv as arguments. 
bbn = Factory.from_data(structure, df_binned_xy)
###Step 5 
###this line performs inference on the data 
join_tree = InferenceController.apply(bbn)

###Write a function for adding evidence:
def evidence(nod, bin_index, val):
    ev = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name(nod)) \
    .with_evidence(bin_index, val) \
    .build()
    return ev 


"""
<<THESE ARE MARGINALS FROM ABOVE TRAINED NET AT STEP 5: LEARN BAYESIAN NET>>
force_bins : 1=0.08000, 2=0.68400, 3=0.23600
acceleration_bins : 1=0.51374, 2=0.31068, 3=0.17558
mass_bins : 1=0.20400, 2=0.66000, 3=0.13600
"""

# ###TESTING: 
# ###Step 4a 
# ###Replace all 'df' with 'x_test' to ensure sampling from testing data
# ###Equidistant binning for the inputs 
for name in x_test:
    name_bins = name + '_bins'
    x_test[name_bins], bin_edges = pd.cut(x_test[name], number_of_bins, labels=labels, retbins=True)
    bin_edges_dict[name_bins]=bin_edges
    # print(bin_edges_dict.items())
    
###This is storing the priorPDs so we can plot them
    name_priors = name + '_priors'
    prior = x_test[name_bins].value_counts(normalize=True).sort_index()
    priorPDs = prior.to_dict()
    prior_dict[name_priors] = priorPDs
#     # print(prior_dict.items())


###Percentile binning for the outputs
###Step 4b
for name in y_test:
    name_bins = name + '_bins'
    y_test[name_bins], bin_edges = pd.qcut(y_test[name], number_of_bins, labels=labels, retbins=True)
    bin_edges_dict[name_bins]=bin_edges

###This is storing the priorPDs so we can plot them
    name_priors = name + '_priors'
    prior = y_test[name_bins].value_counts(normalize=True).sort_index()
    priorPDs = prior.to_dict()
    prior_dict[name_priors] = priorPDs
#     # print(prior_dict)

# ###Must join x_test and y_test and pass to the BN:
# ###We are testing for outputs and therefore do not want to use any of the training data: 
# # df_binned2 = x_test.drop(['m', 'theta','v0', 'vf', 'KE'], axis=1)
df_test_x = x_test.drop(['force', 'mass'], axis=1)
df_test_y = y_test.drop(['acceleration'], axis=1)
df_test_xy = pd.concat([df_test_x, df_test_y], axis=1)

# ###Pybbn only reads data types as strings, so this line converts the data in the csv from int64 to string 
df_test_x = df_test_x.applymap(str)
df_test_y = df_test_y.applymap(str)
df_test_xy = df_test_xy.applymap(str)
# print(df_test_x.head())

###inserting observation evidence
###This is saying that if there is evidence submitted in the arguments for the function to fill evidenceVars with that evidence
###e.g., evidence = {'deflection':[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'weight':[1.0, 0.0, 0.0, 0.0, 0.0, 0.0] }
ev_dict = {}

for col in df_test_x: ###col is the column header i.e., nod in the function.
    ev_dict = {'nod':col, 'bin_index':'1', 'val': 1.0}
    # ev = evidence(col, '2', 1.0) ###we want to apply hard evidence (100% prob) to the first bin, BUT THIS ISN'T SELECTING THE FIRST BIN, MENTION TO ZACK. 
    ev = evidence(**ev_dict)
    join_tree.set_observation(ev)
    # print(ev_dict)

# for node, posteriors in join_tree.get_posteriors().items(): ### this is a list of dictionaries 
#     p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
#     # print(f'{node} : {p}')

"""
<<THESE ARE THE NEW POSTERIORS HAVING SUPPLIED EVIDENCE TO FORCE and MASS>>
force_bins : 1=0.00000, 2=1.00000, 3=0.00000
acceleration_bins : 1=0.22430, 2=0.55140, 3=0.22430
mass_bins : 1=0.00000, 2=1.00000, 3=0.00000
"""

##This is for the figure parameters. 
n_rows = 1
n_cols = len(structure.keys()) ##the length of the BN i.e., five nodes

###instantiate a figure as a placaholder for each distribution (axes)
fig = plt.figure(figsize=((200 * n_cols) / 96, (200 * n_rows) / 96), dpi=96, facecolor='white')
fig.suptitle('Posterior Probabilities', fontsize=8) # title

###Instantiate a counter
i = 0
count = 0

###This is to plot the priorPDs that we stored above.
priorPDs_dict = {}

for var2, idx in prior_dict.items():
    priorPDs_dict[var2] = list(idx.values())

###This creates a variable that corresponds to key (varname) and another variable which corresponds to the value (index)
for varName, index in bin_edges_dict.items(): 
    ax = fig.add_subplot(n_rows, n_cols, count+1) ###subplot with three arguments taken from above, including count
    ax.set_facecolor("whitesmoke") ###sets the background colour of subplot
    # print(index)

    edge = np.zeros((len(bin_edges_dict.items()), len(index[:])))
    binwidths = np.zeros((len(bin_edges_dict.items()), len(index[:-1])))
    xticksv = np.zeros((len(bin_edges_dict.items()), len(index[:-1])))

    ev2 = list(ev_dict.values()) 

    for i in range(len(index)): ###This for loop find the edges, binwidths and midpoints (xticksv) for each of the bins in the dict      
        edge[count, i] = index[i]       
                
    for i in range(len(index)-1): ###This for loop find the edges, binwidths and midpoints (xticksv) for each of the bins in the dict      
        binwidths[count, i] = (index[i+1] - index[i])
        xticksv[count,i]  = ((index[i+1] - index[i]) / 2.) + index[i]

           

    ###This line plots the bars using xticks on x axis, probabilities on the y and binwidths as bar widths. 
    ###It counts through them for every loop within the outer for loop 
    ###posteriotrs.values() is a dict and therefore not ordered, so need to a way to make it ordered for future use. 
    ###This line plots the priorPDs that we stored in the forloop above. 
    ax.bar(xticksv[count], priorPDs_dict[var2], align='center', width=binwidths[count], color='black', alpha=0.2, linewidth=0.2)  
   
    dataDict = {}

    ###Loop goes through a dictionary which contains a key and a value
    ##The first variable i.e., node will correspond to the key and the second i.e., posteriors, will correspond to the value.      
    for node, posteriors in join_tree.get_posteriors().items(): ### this is a list of dictionaries 
        p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
        # print(posteriors) #So Dict(str,List[float]) and Dict(str,Dict(str,float))
        # print(node) ##Can you make the second dict into the same data types as the first
        if varName == node:
            dataDict[node] = list(posteriors.values())
            print(dataDict[node])
            if varName == 'acceleration_bins':
                ax.bar(xticksv[count], dataDict[node], align='center', width=binwidths[count], color='red', alpha=0.2, linewidth=0.2)         
            elif varName == 'mass_bins' or 'force_bins':

                ax.bar(xticksv[count], dataDict[node], align='center', width=binwidths[count], color='green', alpha=0.2, linewidth=0.2)
                
    ###These lines plot the limits of each axis. 
    plt.xlim(min(edge[count]), max(edge[count]))
    plt.xticks([np.round(e, 2) for e in edge[count]], rotation='vertical')
    plt.ylim(0, 1) 

    ###These lines set labels and formatting style for the plots. 
    ax.grid(color='0.2', linestyle=':', linewidth=0.1, dash_capstyle='round')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
    ax.set_title(varName, fontweight="bold", size=6)
    ax.set_ylabel('Probabilities', fontsize=7)  # Y label
    ax.set_xlabel('Ranges', fontsize=7)  # X label
    i+=1
    count+=1

# print(edge)
# print(binwidths)
# print(xticksv)
    
fig.tight_layout()  # Improves appearance a bit.
fig.subplots_adjust(top=0.85)  # white spacing between plots and title   
# plt.show()
