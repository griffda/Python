#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 13:58:57 2022

@author: tomgriffiths

In this script we will attepmt to create a BN using the library pyBBN 
for the BOS model 
"""

import networkx as nx # for drawing graphs
import matplotlib.pyplot as plt # for drawing graphs
import pandas as pd
from pybbn.graph.factory import Factory
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
import numpy as np


#Ingest data from csv fiel path and derive variables for use in the model into a dataframe
df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/output2.csv',usecols = ['m','vf', 'KE'],
encoding=('utf-8'))
#we can print the dataframe and check that the above line has worked
#print(df)


#calculate probability distributions for the dataset
def probs(data, child, parent1=None, parent2=None):
    if parent1==None:
        # Calculate probabilities
        prob=pd.crosstab(data[child], 'Empty', margins=False, normalize='columns').sort_index().to_numpy().reshape(-1).tolist()
    elif parent1!=None:
            # Check if child node has 1 parent or 2 parents
            if parent2==None:
                # Calculate probabilities
                prob=pd.crosstab(data[parent1],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
            else:    
                # Calculate probabilities
                prob=pd.crosstab([data[parent1],data[parent2]],data[child], margins=False, normalize='index').sort_index().to_numpy().reshape(-1).tolist()
    else: print("Error in Probability Frequency Calculations")
    return prob 

print(probs(df, child='vf', parent1='m'))

#create nodes for BBN 
##NOTES the values these lines are searching for in the blue brackets are not correct.
##need to figure out a way of telling it to search for certain values within a range. 
m = BbnNode(Variable(0, 'm', ['4'] ), probs(df, child='m'))
vf = BbnNode(Variable(1, 'vf', ['6']), probs(df, child='vf', parent1='m'))
KE = BbnNode(Variable(2, 'KE', ['8']), probs(df, child='KE', parent1='vf'))

#create network: 
bbn = Bbn() \
    .add_node(m) \
    .add_node(vf) \
    .add_node(KE) \
    .add_edge(Edge(m, vf, EdgeType.DIRECTED)) \
    .add_edge(Edge(vf, KE, EdgeType.DIRECTED)) \

#create the BBN to a join tree 
join_tree = InferenceController.apply(bbn)      


## Create function for drawing the graph related to bbn. 
#Set node positions
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
            "width": 5,}
            
        # Generate graph
        n, d = bbn.to_nx_graph()
        nx.draw(n, with_labels=True, labels=d, pos=pos, **options)

        # Update margins and print the graph
        ax = plt.gca()
        ax.margins(0.10)
        plt.axis("off")
        plt.show()
        return plt


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