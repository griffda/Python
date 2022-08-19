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
                 usecols=['m', 'r', 'mu', 'theta', 'l','vf', 'af', 'KE'],
                 encoding=('utf-8')
                 )
# we can print the dataframe and check that the above line has worked
# print(df)

# Now we can use the pd.cut function to dicretise the data into bins.
# These have been specified depending on the mass of the ball: very small, small etc.
## This method of binning manually tells the function what widths the bins are. 
# df['m_bins'] = pd.cut(x=df['m'],
#                       bins=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
#                       labels=["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]
#                       )

# df['r_bins'] = pd.cut(x=df['r'],
#                        bins=[-7, -5.25, -3.5, -1.75, 0, 1.75, 3.5, 5.25, 7],
#                        labels=["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]
#                        )


# df['vf_bins'] = pd.cut(x=df['vf'],
#                        bins=[0, 4, 8, 12, 16, 20, 24, 28, 33],
#                        labels=["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]
#                        )

# df['KE_bins'] = pd.cut(x=df['KE'],
#                        bins=[-7, -5.25, -3.5, -1.75, 0, 1.75, 3.5, 5.25, 7],
#                        labels=["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]
#                        )


# This method of binning automatically splits into specified number of bins. 
df['m_bins'] = pd.cut(df['m'], 8, 
                     labels=["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]
                     )

df['r_bins'] = pd.cut(df['r'], 8,
                       labels=["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]
                       )

df['mu_bins'] = pd.cut(df['mu'], 8,
                       labels=["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]
                       )

df['theta_bins'] = pd.cut(df['theta'], 8,
                       labels=["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]
                       )

df['l_bins'] = pd.cut(df['l'], 8,
                     labels=["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]
                     )
                    

df['vf_bins'] = pd.cut(df['vf'], 8,
                       labels=["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]
                       )                       

df['af_bins'] = pd.cut(df['af'], 8,
                       labels=["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]
                       )

df['KE_bins'] = pd.cut(df['KE'], 8,
                       labels=["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]
                       )                       

#print(df)

###We can print the dataframe and check that the above line has worked
print(df['m_bins'].value_counts(normalize=True).sort_index())
print(df['r_bins'].value_counts(normalize=True).sort_index())
print(df['mu_bins'].value_counts(normalize=True).sort_index())
print(df['theta_bins'].value_counts(normalize=True).sort_index())
print(df['l_bins'].value_counts(normalize=True).sort_index())
print(df['vf_bins'].value_counts(normalize=True).sort_index())
print(df['af_bins'].value_counts(normalize=True).sort_index())
print(df['KE_bins'].value_counts(normalize=True).sort_index())

print(df['m_bins'].value_counts().sort_index())
print(df['r_bins'].value_counts().sort_index())
print(df['mu_bins'].value_counts().sort_index())
print(df['theta_bins'].value_counts().sort_index())
print(df['l_bins'].value_counts().sort_index())
print(df['vf_bins'].value_counts().sort_index())
print(df['af_bins'].value_counts().sort_index())
print(df['KE_bins'].value_counts().sort_index())

###Calculate probability distributions for the dataset
def probs(data, child, parent1=None, parent2=None, parent3=None, parent4=None, parent5=None):
    if parent1 == None:
        # Calculate probabilities
        prob = pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index(
        ).to_numpy().reshape(-1).tolist()
    elif parent1 != None:
        # Check if child node parents between 1 and 5
        if parent2 == None:
            # Calculate probabilities
            prob = pd.crosstab(data[parent1], data[child], margins=False,
                               normalize='index').sort_index().to_numpy().reshape(-1).tolist()
        if parent3 == None:
            prob = pd.crosstab(data[parent2], data[child], margins=False,
                               normalize='index').sort_index().to_numpy().reshape(-1).tolist()
        if parent4 == None:
            prob = pd.crosstab(data[parent3], data[child], margins=False,
                               normalize='index').sort_index().to_numpy().reshape(-1).tolist()
        if parent5 == None:
            prob = pd.crosstab(data[parent4], data[child], margins=False,
                               normalize='index').sort_index().to_numpy().reshape(-1).tolist()                                              
        else:
            # Calculate probabilities
            prob = pd.crosstab([data[parent1], data[parent2], data[parent3], data[parent4], data[parent5]], data[child], margins=False,
                               normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else:
        print("Error in Probability Frequency Calculations")
    return prob


###Create nodes for BBN
m = BbnNode(Variable(0, 'm', ["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]), 
            df['m_bins'].value_counts(normalize=True).sort_index()
            )

r = BbnNode(Variable(1, 'r', ["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]), 
            df['r_bins'].value_counts(normalize=True).sort_index()
            )
            
mu = BbnNode(Variable(2, 'mu', ["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]), 
            df['mu_bins'].value_counts(normalize=True).sort_index()
            )

theta = BbnNode(Variable(3, 'theta', ["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]), 
                df['theta_bins'].value_counts(normalize=True).sort_index()
                )
                
l = BbnNode(Variable(4, 'l', ["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]), 
            df['l_bins'].value_counts(normalize=True).sort_index()
            )

vf = BbnNode(Variable(5, 'vf', ["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]), 
            df['vf_bins'].value_counts(normalize=True).sort_index()
            )

af = BbnNode(Variable(6, 'af', ["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]), 
            df['af_bins'].value_counts(normalize=True).sort_index()
            )
            
KE = BbnNode(Variable(7, 'KE', ["very very small", "very small", "small", "small/medium", "medium", "medium/large", "large", "very large"]), 
            df['KE_bins'].value_counts(normalize=True).sort_index()
            )



# print(probs(df, child='KE_bins', parent1='m_bins', parent2='r_bins', parent3='mu_bins', parent4='theta_bins', parent5='l_bins'))
# print(probs(df, child='r_bins'))

###Create network:
###This one throws an error "IndexError: index 8 is out of bounds for axis 0 with size 8" as soon as edges are added.
bbn = Bbn() \
    .add_node(m) \
    .add_node(r) \
    .add_node(mu) \
    .add_node(theta) \
    .add_node(l) \
    .add_node(vf) \
    .add_node(af) \
    .add_node(KE) \
    .add_edge(Edge(m, vf, EdgeType.DIRECTED)) \
    .add_edge(Edge(r, vf, EdgeType.DIRECTED)) \
    .add_edge(Edge(mu, vf, EdgeType.DIRECTED)) \
    .add_edge(Edge(theta, vf, EdgeType.DIRECTED)) \
    .add_edge(Edge(l, vf, EdgeType.DIRECTED)) \
    .add_edge(Edge(m, af, EdgeType.DIRECTED)) \
    .add_edge(Edge(r, af, EdgeType.DIRECTED)) \
    .add_edge(Edge(mu, af, EdgeType.DIRECTED)) \
    .add_edge(Edge(theta, af, EdgeType.DIRECTED)) \
    .add_edge(Edge(l, af, EdgeType.DIRECTED)) \
    .add_edge(Edge(m, KE, EdgeType.DIRECTED)) \
    .add_edge(Edge(r, KE, EdgeType.DIRECTED)) \
    .add_edge(Edge(mu, KE, EdgeType.DIRECTED)) \
    .add_edge(Edge(theta, KE, EdgeType.DIRECTED)) \
    .add_edge(Edge(l, KE, EdgeType.DIRECTED)) 


###Create the BBN to a join tree
join_tree = InferenceController.apply(bbn)

###Create function for drawing the graph related to bbn.
###Set node positions
def drawbn(bbn):
    pos = {0: (-1, 2), 1: (-2, 2), 2: (-3, 2), 3: (-4, 2), 4: (-5, 2), 5: (-2, 1), 6: (-3, 1), 7:(-4, 1)}

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