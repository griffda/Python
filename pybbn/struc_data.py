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
import numpy as np

df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/output2.csv',
                 usecols=['m', 'theta','v0', 'vf', 'KE'],
                 encoding=('utf-8')
                 )

#print(df.head(10))

##Create new data frame - call it binned and fill with the values and then use structure syntax below. 


df['m_bins']= np.digitize(df['m'], np.linspace(df['m'].min(), df['m'].max(), 10))
df['theta_bins']= np.digitize(df['theta'], np.linspace(df['theta'].min(), df['theta'].max(), 10))
df['v0_bins']= np.digitize(df['v0'], np.linspace(df['v0'].min(), df['v0'].max(), 10))
df['vf_bins']= np.digitize(df['vf'], np.linspace(df['vf'].min(), df['vf'].max(), 10))
df['KE_bins']= np.digitize(df['KE'], np.linspace(df['KE'].min(), df['KE'].max(), 10))

df_binned = df.drop(['m', 'theta','v0', 'vf', 'KE'], axis=1)

print(df_binned.head(10))

structure = {
    'theta_bins': [],
    'v0_bins':[],
    'm_bins':[],
    'vf_bins': ['theta_bins', 'v0_bins', 'm_bins'],
    'KE_bins': ['theta_bins', 'v0_bins', 'm_bins']
}

bbn = Factory.from_data(structure, df_binned)

join_tree = InferenceController.apply(bbn)

# for node, posteriors in join_tree.get_posteriors().items():
#     p = ', '.join([f'{val}={prob:.5f}' for val, prob in posteriors.items()])
#     print(f'{node} : {p}')

    