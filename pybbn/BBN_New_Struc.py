#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 13:58:57 2022

@author: tomgriffiths

In this script we will attepmt to create a BN using the library pyBBN 
for the BOS model.

BBNs structure is updated: 
- input nodes: theta, m, v0
- output nodes: vf, KE.
- Larger data set of 200 rows.
- Data will use preset bin widths and will test how changing this affects the outputs.
"""

import networkx as nx  # for drawing graphs
import matplotlib.pyplot as plt  # for drawing graphs
import pandas as pd
from pybbn.graph.factory import Factory
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
import numpy as np


##Ingest data from csv fiel path and derive variables for use in the model into a dataframe
df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/output2.csv',
                 usecols=['m', 'theta', 'v0', 'vf', 'KE'],
                 encoding=('utf-8')
                 )
##we can print the dataframe and check that the above line has worked
# print(df)

df['m_bins'] = pd.cut(x=df['m'],
                      bins=6,
                      labels=["small", "small/medium", "medium", "medium/large", "large", "very large"]
                      )

df['theta_bins'] = pd.cut(x=df['theta'],
                      bins=6,
                      labels=["small", "small/medium", "medium", "medium/large", "large", "very large"]
                      )

df['v0_bins'] = pd.cut(x=df['v0'],
                      bins=6,
                      labels=["small", "small/medium", "medium", "medium/large", "large", "very large"]
                      )  

df['vf_bins'] = pd.cut(x=df['vf'],
                      bins=6,
                      labels=["small", "small/medium", "medium", "medium/large", "large", "very large"]
                      )  

df['KE_bins'] = pd.cut(x=df['KE'],
                      bins=6,
                      labels=["small", "small/medium", "medium", "medium/large", "large", "very large"]
                      )



#plt.hist(df["KE_bins"],bins=6)
#plt.hist(df["theta_bins"],bins=6)
#plt.hist(df["v0_bins"],bins=6)
#plt.hist(df["vf_bins"],bins=6)
plt.hist(df["KE_bins"],bins=20)
plt.show()