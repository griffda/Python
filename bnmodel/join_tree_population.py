# This file contains functions for creating a join tree from a given structure and data set.
from pybbn.graph.factory import Factory
from pybbn.pptc.inferencecontroller import InferenceController
from pybbn.graph.jointree import EvidenceBuilder

###Step 5
###This is telling us how the network is structured between parent nodes and posteriors.
###Using this method, we avoid having to build our own conditional probability tables using maximum likelihood estimation.
structure = {
    'mass':[],
    'force': [],
    'acceleration': ['force', 'mass']
}

##Write a function for probability dists:
def prob_dists(structure, data):
    bbn = Factory.from_data(structure, data)
    join_tree = InferenceController.apply(bbn)
    return join_tree

# join_tree = prob_dists(structure, df_train_binned) ###Use the function:

##Write a function for adding evidence:
def evidence(nod, bin_index, val, join_tree):
    ev = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name(nod)) \
    .with_evidence(str(bin_index), val) \
    .build()
    # print("setting evidence")
    return ev

