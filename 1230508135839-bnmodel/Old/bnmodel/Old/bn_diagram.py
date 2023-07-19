"""
Created on Wed 17 Aug 16:00 2022

@author: tomgriffiths

In this script we will attepmt to create a script that displays our BN plot
NOTES: THIS WILL NOT RUN UNLESS LOCATED WITHIN ANOTHER SCRIPT OR YOU HAVE CREATED A FUNCTION TO CALL
"""

import networkx as nx # for drawing graphs
import matplotlib.pyplot as plt # for drawing graphs

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

