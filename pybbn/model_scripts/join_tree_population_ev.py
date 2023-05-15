from join_tree_population import join_tree
from pybbn.graph.factory import Factory
from pybbn.pptc.inferencecontroller import InferenceController
from pybbn.graph.jointree import EvidenceBuilder

##Write a function for adding evidence:
def evidence(nod, bin_index, val):
    ev = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name(nod)) \
    .with_evidence(bin_index, val) \
    .build()
    return ev
