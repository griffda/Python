# This file contains functions for creating a join tree from a given structure and data set.
from pybbn.graph.factory import Factory
from pybbn.pptc.inferencecontroller import InferenceController
from pybbn.graph.jointree import EvidenceBuilder


def prob_dists(structure, data):
    """
    This is telling us how the network is structured between parent nodes and posteriors.
    Using this function, we avoid having to build our own conditional probability tables using maximum likelihood estimation.
    Corresponds to step 5 in Zac's thesis.

    Parameters
    ----------
    structure : dict 
    data : pandas dataframe with the trainned data

    Returns
    ------- 
    join_tree : conditional probability table
    """
    bbn = Factory.from_data(structure, data)
    join_tree = InferenceController.apply(bbn)
    return join_tree


def evidence(nod, bin_index, val, join_tree):
    """
    Prepare an evidence to be used as input for the join tree.

    Parameters
    ----------
    nod : str node name
    bin_index : str
    val : float 
    join_tree : conditional probability table

    Returns
    -------
    ev : evidence object 
    """

    ev = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name(nod)) \
    .with_evidence(str(bin_index), val) \
    .build()
    # print("setting evidence")
    return ev

