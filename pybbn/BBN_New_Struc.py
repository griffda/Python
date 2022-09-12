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

from urllib.request import proxy_bypass
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

df['vf_bins']= np.digitize(df['vf'], np.linspace(df['vf'].min(), df['vf'].max(), 10))
df['m_bins']= np.digitize(df['m'], np.linspace(df['m'].min(), df['m'].max(), 10))
df['theta_bins']= np.digitize(df['theta'], np.linspace(df['theta'].min(), df['theta'].max(), 10))
df['v0_bins']= np.digitize(df['v0'], np.linspace(df['v0'].min(), df['v0'].max(), 10))
df['KE_bins']= np.digitize(df['KE'], np.linspace(df['KE'].min(), df['KE'].max(), 10))

# print(df.head(5))
##We can print the dataframe and check that the above line has worked
#print(df['m_bins'].value_counts(normalize=True).sort_index())
m_npbins = df['m_bins'].value_counts(normalize=True).sort_index().to_list()
# theta_npbins = df['theta_bins'].value_counts(normalize=True).sort_index()
# v0_npbins = df['v0_bins'].value_counts(normalize=True).sort_index()
vf_npbins = df['vf_bins'].value_counts(normalize=True).sort_index().to_list()
# KE_npbins = df['KE_bins'].value_counts(normalize=True).sort_index()


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

prob = pd.crosstab([df['theta_bins'], df['v0_bins']], df['KE_bins'], margins=False,
                               normalize='index').sort_index().to_numpy().reshape(-1).tolist()

# print(probs(df, child='m_bins'))
# print(probs(df, chil='KE_bins', parent1='vf_bins'))
# print(m_npbins)

###Create nodes for BBN
# m = BbnNode(Variable(0, 'm', ['1','2','3','4','5','6','7','8','9','10']),
#             probs(df, child='m_bins')
#             )

theta = BbnNode(Variable(1, 'theta', ['1','2','3','4','5','6','7','8','9','10']),
            probs(df, child='theta_bins')
            )

v0 = BbnNode(Variable(2, 'v0', ['1','2','3','4','5','6','7','8','9','10']),
            probs(df, child='v0_bins')
            )

vf = BbnNode(Variable(3, 'vf', ['1','2','3','4','5','6','7','8','9','10']),
            pd.crosstab([df['theta_bins'], df['v0_bins']], df['vf_bins'], margins=False,
                               normalize='index').sort_index().to_numpy().reshape(-1).tolist()
            )

# KE = BbnNode(Variable(4, 'KE', ['1','2','3','4','5','6','7','8','9','10']),
#             probs(df, child='KE', parent1='vf', parent2='m')
#             )            


###Create network:
bbn = Bbn() \
    .add_node(theta) \
    .add_node(v0) \
    .add_node(vf) \
    .add_edge(Edge(theta, vf, EdgeType.DIRECTED)) \
    .add_edge(Edge(v0, vf, EdgeType.DIRECTED)) \
    


###Create function for drawing the graph related to bbn.
###Set node positions
def drawbn(bbn):
    pos = {0: (1, 0), 1: (-1, 1), 2: (1, 1), 3: (0, 0), 4: (0, -1)}

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

###Create the BBN to a join tree
join_tree = InferenceController.apply(bbn)

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

###Histograms of binned values
# plt.hist(df["m_bins"])
# plt.hist(df["v0_bins"])
#plt.hist(df["theta_bins"])
# plt.hist(df["KE_bins"])
# plt.hist(df["vf_bins"])
#plt.show()