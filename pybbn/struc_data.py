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
"""

import pandas as pd
from pybbn.graph.factory import Factory
from pybbn.pptc.inferencecontroller import InferenceController
import numpy as np

###Reading csv file of data
df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/binned_data.csv')
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

###this line performs inference on the data 
join_tree = InferenceController.apply(bbn)

###When building network you can get the names of the nodes using bbn.nodes or something to that affect. 

###Try to fill an empty array with the posteriors and plot them from there. 
p=[]
arr = []

###This tells us the posterior probabilities 
for node, posteriors in join_tree.get_posteriors().items():
    ###This is useful to display them as f-strings 
    p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
    print(p)
    
    ###This method passes the key value pairs. 
    # p = posteriors
    # df = pd.DataFrame(p.items(), columns = [node, 'Posterior Probability'])
    # b = df.iloc[:,1].values
    # c = df.iloc[:,0].values
    # print(b)
    # print(c)

    
    ####This method passes the Series constructor
    # s = pd.Series(p, name='Posterior Probability')
    # s.index.name = 'Bins'
    # s.reset_index()
    # print(s)

    ###This method uses numpy to pass data into an array:
    # # data = posteriors.items()
    # # arr = np.array(data)
    # # print(data)