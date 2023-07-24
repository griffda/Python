from bnmodel.join_tree_population import evidence
import pandas as pd

class ObservationGenerator:
    def __init__(self, test_binned, output, data):
        self.test_binned = test_binned
        self.output = output
        self.data = data

    def generate_obs_dict(self):
        """
        Generate a single observation from the test dataset
        
        Changes:
        Need to change this so that we can, if desired, generate observations from another dataset 
        (e.g., if new data is available) This will require changing the function signature to include
        kwargs for the dataset and the output variable. 
        If the dataset has different bin ranges, we will need to change the bin ranges in the function  

        Returns
        -------
        obs_dict : observation dictionary
        """
        # choose a random row from the test_binned
        row = self.test_binned.sample()

        # generate an obs_dict from the chosen row
        obs_dict = {}
        for col in self.test_binned.columns:
            if col == self.output:
                obs_dict[col] = {'bin_index': str(row[col].values[0]), 'actual_value': self.data[self.output][row.index.values].values[0]}
            else:
                obs_dict[col] = {'bin_index': str(row[col].values[0]), 'val': 1.0}
        return obs_dict

    def generate_multiple_obs_dicts(self, num_samples):
        """
        Generate num_samples observations from the test dataset

        Parameters
        ----------
        num_samples : int number of samples to generate

        Returns
        -------
        obs_dicts : list of observation dictionaries
        """
        obs_dicts = []
        for i in range(num_samples):
            obs_dict = self.generate_obs_dict()
            obs_dicts.append(obs_dict)
        return obs_dicts

    def gen_ev_list(self, obs_dicts):
        """
        Parameters
        ----------
        obs_dicts : list of observation dictionaries
        """
        self.test_binned = self.test_binned.drop([self.output], axis=1) 
        all_ev_list = []
        for obs in obs_dicts:
            ev_list = []
            for col in self.test_binned.columns:
                bin_index = obs[col]['bin_index']
                val = obs[col]['val']
                ev_dict = {'nod': col, 'bin_index': bin_index, 'val': val}
                ev_list.append(ev_dict)
            all_ev_list.append(ev_list)
        return all_ev_list

    def get_posteriors(self, join_tree):
        """
        Get the posteriors for the observations included in the join tree.

        Parameters
        ----------
        join_tree : conditional probability table
        """
        obs_posteriors = {}
        predictedTargetPosteriors = []
        for node, posteriors_raw in join_tree.get_posteriors().items():
            obs_posteriors[node] = [posteriors_raw[val] for val in posteriors_raw]
            if node == self.output:  # check if the observation corresponds to the specified target variable
                predictedTargetPosteriors = [posteriors_raw[val] for val in posteriors_raw]
        return obs_posteriors, predictedTargetPosteriors

    def get_all_posteriors(self, all_ev_list, join_tree):
        """
        Get the posteriors for all the observations in all_ev_list for the corresponding join_tree.

        Parameters
        ----------
        all_ev_list : list of observations
        join_tree : conditional probability table

        Returns
        -------
        obs_posteriors : dict of observations posteriors
        predicted_posteriors : list of predicted posteriors
        """
        obs_posteriors = {}
        predicted_posteriors = []

        for observation in all_ev_list:
            join_tree.unobserve_all()
            # Do a duplicate of join_tree to avoid modifying the original one
            for ev in observation:
                # Modify the join_tree using this case evidences
                ev4jointree = evidence(ev['nod'], ev['bin_index'], ev['val'], join_tree)
                join_tree.set_observation(ev4jointree)
            # Get the posteriors for this observation
            aux_obs, aux_prd = self.get_posteriors(join_tree)

            # Store the posteriors for this case
            for node_id, posterior in aux_obs.items():
                if node_id not in obs_posteriors:
                    obs_posteriors[node_id] = []
                obs_posteriors[node_id].append(posterior)
            predicted_posteriors.append(aux_prd)
        
        # Ensure that the join tree is unmodified
        join_tree.unobserve_all()

        return obs_posteriors, predicted_posteriors
