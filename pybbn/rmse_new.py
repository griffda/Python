import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # for drawing graphs
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle

posteriorDict_training = {}
priorDict_training = {}
posteriorDict_testing = {}
y_testing_probsDict = {}

###Results of STEP 5 Learn Bayes Net:
###These are the training set probabilities for the inputs and outputs with NO evidence applied. 
with open('xy_train_priors.pkl', 'rb') as f:
    xy_train_priors = pickle.load(f)
    # print(xy_train_priors.items())
for node, posteriors in xy_train_priors.items(): ### this is a list of dictionaries 
    p_no_ev = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
    print(f'{node} : {p_no_ev}')
    if node == 'acceleration_bins':
        priorDict_training[node] = (list(posteriors.values()))
        prior_mean = np.mean(priorDict_training[node])
        # print(prior_dict[node])
        # print(prior_mean)



###Step 6: Validate Bayes Net requires: 
###input testing set, which through inference gives predicted output set
###output testing set
###These are the testing set probability distribution for the output (y target) (output testing set)
with open('y_testing_probs.pkl', 'rb') as f:
    y_testing_probs = pickle.load(f)
    print(y_testing_probs)



for node, posteriors in y_testing_probs.items(): ### this is a list of dictionaries 
    print(node)
    print(posteriors)
    # y_testing_probsDict[node] = list(posteriors)
    # print(y_testing_probs[node])
    # p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
    # print(f'{node} : {p}')
    # if node == 'acceleration_bins':
    #     y_testing_probsDict[node] = list(posteriors.values())
    #     print(posteriorDict_training[node])


###These are the testing set probabilities for inputs and outputs with evidence applied. 
####Predicted output set
with open('posteriors_evidence.pkl', 'rb') as f:
    posteriors_evidence = pickle.load(f)
for node, posteriors in posteriors_evidence.items(): ### this is a list of dictionaries 
    p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
    print(f'{node} : {p}')
    posteriorDict_testing[node] = list(posteriors.values())
        # print(dataDict[node])
    # print(posteriorDict_testing['acceleration_bins'])

expectedV = 0.0
count = 0
i = 0 

###These are the bins for the training set inputs and outputs:
with open('bin_edges_dict_train.pkl', 'rb') as f:
    bin_edges_train = pickle.load(f)
for varName, index in bin_edges_train.items():
    # print(bin_edges_dict_test.items())
    if varName == 'acceleration_bins':
        mean_bin_value = np.zeros((len(bin_edges_train['acceleration_bins'][:-6]), len(index[:-1])))
        mean2 = np.mean(index)
        # print(mean2)
        for i in range(len(index)-1): ###This for loop find the edges, binwidths and midpoints (xticksv) for each of the bins in the dict      
            # print(index)
            mean_bin_value[count,i] = ((index[i+1] - index[i]) / 2.) + index[i]
        expectedV += mean_bin_value * priorDict_training['acceleration_bins'] ###this 
        # print(varName)
        # print(index)
        # print(mean_bin_value)
        # print(expectedV)


###These are the bins for the testing set outputs:
with open('y_test_bins.pkl', 'rb') as f:
    y_test_bins = pickle.load(f)
for varName, index in y_test_bins.items():  
    # print(bin_edges_dict_test.items())
    if varName == 'acceleration_bins':
        mean_bin_value = np.zeros((len(y_test_bins['acceleration_bins'][:-6]), len(index[:-1])))
        mean2 = np.mean(index)
        # print(mean2)
        for i in range(len(index)-1): ###This for loop find the edges, binwidths and midpoints (xticksv) for each of the bins in the dict      
            # print(index)
            mean_bin_value[count,i] = ((index[i+1] - index[i]) / 2.) + index[i]
        expectedV += mean_bin_value * posteriorDict_testing['acceleration_bins'] ###this 
        print(varName)
        print(index)
        print(mean_bin_value)
        print(expectedV)

##This is for the figure parameters. 
n_rows = 1
# n_cols = len(structure.keys()) ##the length of the BN i.e., five nodes
n_cols = 1

###instantiate a figure as a placaholder for each distribution (axes)
fig = plt.figure(figsize=((200 * n_cols) / 96, (200 * n_rows) / 96), dpi=96, facecolor='white')
fig.suptitle('Posterior Probabilities', fontsize=8) # title

###This creates a variable that corresponds to key (varname) and another variable which corresponds to the value (index)
for varName, index in y_test_bins.items(): 
    # ax = fig.add_subplot(n_rows, n_cols, count+1) ###subplot with three arguments taken from above, including count
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("whitesmoke") ###sets the background colour of subplot
    # print(index)

    edge = np.zeros((len(y_test_bins.items()), len(index[:])))
    binwidths = np.zeros((len(y_test_bins.items()), len(index[:-1])))
    xticksv = np.zeros((len(y_test_bins.items()), len(index[:-1])))

    # ev2 = list(ev_dict.values()) 

    for i in range(len(index)): ###This for loop find the edges, binwidths and midpoints (xticksv) for each of the bins in the dict      
        edge[count, i] = index[i]       
                
    for i in range(len(index)-1): ###This for loop find the edges, binwidths and midpoints (xticksv) for each of the bins in the dict      
        binwidths[count, i] = (index[i+1] - index[i])
        xticksv[count,i]  = ((index[i+1] - index[i]) / 2.) + index[i]

    ###This line plots the bars using xticks on x axis, probabilities on the y and binwidths as bar widths. 
    ###It counts through them for every loop within the outer for loop 
    ###posteriotrs.values() is a dict and therefore not ordered, so need to a way to make it ordered for future use. 
    ###This line plots the priorPDs that we stored in the forloop above. 


    ###Loop goes through a dictionary which contains a key and a value
    ##The first variable i.e., node will correspond to the key and the second i.e., posteriors, will correspond to the value.      
    for node, posteriors in posteriors_evidence.items(): ### this is a list of dictionaries 
        p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
        # print(posteriors) #So Dict(str,List[float]) and Dict(str,Dict(str,float))
        # print(node) ##Can you make the second dict into the same data types as the first
        if varName == node:
            posteriorDict_testing[node] = list(posteriors.values())
            # print(dataDict[node])
            if varName == 'acceleration_bins':
                ax.bar(xticksv[count], posteriorDict_testing[node], align='center', width=binwidths[count], color='red', alpha=0.2, linewidth=0.2)         
            # elif varName == 'mass_bins' or 'force_bins':
            #     ax.bar(xticksv[count], dataDict[node], align='center', width=binwidths[count], color='green', alpha=0.2, linewidth=0.2)
    
    for node, posteriors in y_testing_probs.items(): ### this is a list of dictionaries 
        # print(node)
        # print(posteriors)
        # y_testing_probsDict[node] = list(posteriors)
        # print(y_testing_probs[node])
        ax.bar(xticksv[count], posteriors, align='center', width=binwidths[count], color='black', alpha=0.2, linewidth=0.2)  


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

    
fig.tight_layout()  # Improves appearance a bit.
fig.subplots_adjust(top=0.85)  # white spacing between plots and title  
plt.show()