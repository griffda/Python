"""
Created on Tues 23 Aug 10:18:36 2022

@author: tomgriffiths

In this script we will attepmt to create a BN using the library pyBBN
for the BOS model.

This one is an update of BBN_3 and will attempt to bring all inputs and outputs together for full BN structure for a larger data set of 200 rows.
Data will use preset bin widths and will test how changing this affects the outputs.
The method of binning and generating the input probability distributions will be different from Pandas.
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 13:58:57 2022

@author: tomgriffiths

In this script we will attepmt to create a BN using the library pyBBN
for the BOS model.

This one is an update of BBN_2 and will attempt to bring all inputs and outputs together for full BN structure for a larger data set of 200 rows.
Data will use preset bin widths and will test how changing this affects the outputs.
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


# Ingest data from csv fiel path and derive variables for use in the model into a dataframe
df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/output2.csv',
                 usecols=['m', 'r', 'mu', 'theta', 'l','vf', 'af', 'KE'],
                 encoding=('utf-8')
                 )
# we can print the dataframe and check that the above line has worked
# print(df)
#print(df.head(5))

###This method of binning will use numpy digitize to create probability distributions
###np.linspace is used to automate the bin ranges for the specific column data range
### have not been able to label bins yet.

###This checks that the linspace works
# m_bins = np.linspace(df['m'].min(), df['m'].max(), 3)
# print(m_bins)

##binning
df['m_bins']= np.digitize(df['m'], np.linspace(df['m'].min(), df['m'].max(), 6))


# df['r_bins']= np.digitize(df['r'], np.linspace(df['r'].min(), df['r'].max(), 6))


df['vf_bins']= np.digitize(df['vf'], np.linspace(df['vf'].min(), df['vf'].max(), 6))
print(df.head(5))


###We can print the dataframe and check that the above line has worked
# print(df['r_bins'].value_counts(normalize=True).sort_index())
# print(df['r_bins'].value_counts().sort_index())


###Create nodes for BBN
m = BbnNode(Variable(0, 'm', ['1','2','3','4','5','6']),
            df['m_bins'].value_counts(normalize=True).sort_index()
            )

# r = BbnNode(Variable(1, 'r', ['1','2','3','4','5','6']),
#             df['r_bins'].value_counts(normalize=True).sort_index()
#             )

vf = BbnNode(Variable(2, 'vf', ['1','2','3','4','5','6']),
            df['vf_bins'].value_counts(normalize=True).sort_index()
            )


###Create network:
bbn = Bbn() \
    .add_node(m) \
    .add_node(vf) \
    .add_edge(Edge(m, vf, EdgeType.DIRECTED)) \

###Create the BBN to a join tree
join_tree = InferenceController.apply(bbn)

###Create function for drawing the graph related to bbn.
###Set node positions
def drawbn(bbn):
    pos = {0: (-1, 2), 1: (-2, 2), 2: (-1.5, 1)}

    # Set options for graph looks
    options = {
        "font_size": 16,
        "node_size": 4000,
        "node_color": "white",
        "edgecolors": "black",
        "edge_color": "red",
        "linewidths": 5,
        "width": 5, }

    # Generate graph
    n, d = bbn.to_nx_graph()
    nx.draw(n, with_labels=True, labels=d, pos=pos, **options)

    # Update margins and print the graph
    ax = plt.gca()
    ax.margins(0.10)
    plt.axis("off")
    plt.show()
    return plt

#drawbn(bbn)

# Define a function for printing marginal probabilities
def print_probs():
    for node in join_tree.get_bbn_nodes():
        potential = join_tree.get_bbn_potential(node)
        print("Node:", node)
        print("Values:")
        print(potential)
        print('----------------')

# Use the above function to print marginal probabilities
print_probs()
# plt.hist(df["KE_bins"], bins =8)
# plt.show()