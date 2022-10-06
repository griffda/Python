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
"""

import pandas as pd
from pybbn.graph.factory import Factory
from pybbn.pptc.inferencecontroller import InferenceController
from create_plot import CreatePlot
import numpy as np

###Reading csv file of data
df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/binned_data.csv')
###Pybbn only reads data types as strings, so this line converts the data in the csv from int64 to string 
df = df.applymap(str)

b=CreatePlot2

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


###This tells us the posterior probabilities 
for node, posteriors in join_tree.get_posteriors().items():
    p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
    print(f'{node} : {p}')

evidence = {'m':[1.0,0.0,0,0], 'theta':[1.0,0.0,0,0], 'v0':[1.0,0.0,0,0]}

b.plotPDs(b, 'Posteriors', 'Ranges', 'Probabilities', displayplt=True, posteriorPD=posteriors, evidence=evidence.keys())