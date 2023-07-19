
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 13:58:57 2022
@author: tomgriffiths
In this script we will attepmt to create a BN using the library pyBBN 
for the BOS model.
This one is an update of BBN_1 and will attempt to dicretize the data into bins before being put through the BN. 
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
                 usecols=['m', 'vf', 'KE'],
                 encoding=('utf-8')
                 )
# we can print the dataframe and check that the above line has worked
# print(df)

# Now we can use the pd.cut function to dicretise the data into bins.
# These have been specified depending on the mass of the ball: very small, small etc.
df['m_bins'] = pd.cut(x=df['m'],
                      bins=8,
                      labels=["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]
                      )

df['vf_bins'] = pd.cut(x=df['vf'],
                       bins=8,
                       labels=["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]
                       )

df['KE_bins'] = pd.cut(x=df['KE'],
                       bins=8,
                       labels=["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]
                       )
#print(df)

# we can print the dataframe and check that the above line has worked
#print(df['vf_bins'].value_counts(normalize=True).sort_index())

# calculate probability distributions for the dataset


def probs(data, child, parent1=None, parent2=None):
    if parent1 == None:
        # Calculate probabilities
        prob = pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index(
        ).to_numpy().reshape(-1).tolist()
    elif parent1 != None:
        # Check if child node has 1 parent or 2 parents
        if parent2 == None:
            # Calculate probabilities
            prob = pd.crosstab(data[parent1], data[child], margins=False,
                               normalize='index').sort_index().to_numpy().reshape(-1).tolist()
        else:
            # Calculate probabilities
            prob = pd.crosstab([data[parent1], data[parent2]], data[child], margins=False,
                               normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else:
        print("Error in Probability Frequency Calculations")
    return prob

print(probs(df, child='m_bins'))
print(probs(df, child='KE_bins', parent1='vf_bins'))

# create nodes for BBN
m = BbnNode(Variable(0, 'm', ["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]), probs(df, child='m_bins'))
vf = BbnNode(Variable(1, 'vf', ["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]), probs(df, child='vf_bins', parent1='m_bins'))
KE = BbnNode(Variable(2, 'KE', ["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]), probs(df, child='KE_bins', parent1='vf_bins'))

# create network:
bbn = Bbn() \
    .add_node(m) \
    .add_node(vf) \
    .add_node(KE) \
    .add_edge(Edge(m, vf, EdgeType.DIRECTED)) \
    .add_edge(Edge(vf, KE, EdgeType.DIRECTED))


# create the BBN to a join tree
join_tree = InferenceController.apply(bbn)

# Create function for drawing the graph related to bbn.

# Set node positions


def drawbn(bbn):
    pos = {0: (-1, 2), 1: (-1, 0.5), 2: (1, 0.5), 3: (0, -1)}

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
#print_probs()
#plt.hist(df["KE_bins"], bins =8)
#plt.show()