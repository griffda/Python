from pybbn.graph.factory import Factory
from pybbn.pptc.inferencecontroller import InferenceController
from pybbn.graph.jointree import EvidenceBuilder

class JoinTreeBuilder:
    def __init__(self, structure, data):
        self.structure = structure
        self.data = data

    def prob_dists(self):
        """
        This is telling us how the network is structured between parent nodes and posteriors.
        Using this function, we avoid having to build our own conditional probability tables using maximum likelihood estimation.
        Corresponds to step 5 in Zac's thesis.

        Returns
        ------- 
        join_tree : conditional probability table
        """
        bbn = Factory.from_data(self.structure, self.data)
        join_tree = InferenceController.apply(bbn)
        return join_tree

    def evidence(self, nod, bin_index, val, join_tree):
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
        return ev
