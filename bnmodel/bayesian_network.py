import bnmodel as bn
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

class BayesianNetwork:
    def __init__(self, inputs):
        self.inputs = inputs  
        self.__load_inputs()
        self.join_tree = None
    
    def __load_inputs(self):
        """
        Prepare the inputs for the training

        """
        if self.inputs['method'] == 'uniform':
            x, y, bin_edges, prior_xytrn = bn.discretisation.binning_data(self.inputs['data'],
                                                                          self.inputs['nbins'],
                                                                          self.inputs['inputs'],
                                                                          self.inputs['output'])
            self.bin_edges = bin_edges
            self.prior_xytrn = prior_xytrn

            # Split the data into training and testing sets
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.inputs['train_test_split'])
        
            # Combine the binned data into a single DataFrame for each set
            # Pybbn only reads data types as strings, so this line converts the data in the csv from int64 to string
            train_binned = pd.concat([x_train, y_train], axis=1)
            train_binned = train_binned.astype(str)

            self.train_binned = train_binned
            self.test_binned = pd.concat([x_test, y_test], axis=1)

        else:
            # TODO: implement kfold method
            raise ValueError('Invalid method for discretisation')

        self.struct = bn.utilities.df2struct(self.inputs['data'], self.inputs['inputs'], self.inputs['output'])

    def train(self):
        if self.inputs['method'] == 'uniform':
            self.join_tree = bn.join_tree_population.prob_dists(self.struct, self.train_binned)
            
        else:
            # TODO: implement kfold method
            raise ValueError('Invalid method for discretisation')
        
    def validate(self):
        if self.join_tree is None:
            raise ValueError('Model not trained yet')
        
        # Get the posteriors for the testing subset
        # TODO: create a run_model function. Obs_posteriors should be in a different function
        obs_dicts = bn.generate_posteriors.generate_multiple_obs_dicts(self.test_binned, 
                                                                       self.inputs['output'], 
                                                                       self.inputs['data'])
        all_ev_list = bn.generate_posteriors.gen_ev_list(self.test_binned,
                                                         obs_dicts,
                                                         self.inputs['output'])
        obs_posteriors, predicted_posteriors = bn.generate_posteriors.get_all_posteriors(all_ev_list,
                                                                                         self.join_tree,
                                                                                         self.inputs['output'])

        self.test = {'obs_posteriors': obs_posteriors,
                     'predicted_posteriors': predicted_posteriors}
        
        # Error evaluation
        correct_bin_locations, actual_values = bn.evaluate_errors.get_correct_values(obs_dicts, self.test)
        bin_ranges = bn.evaluate_errors.extract_bin_ranges(self.test, self.bin_edges)
        distance_errors, norm_distance_errors, output_bin_means = bn.evaluate_errors.distribution_distance_error(correct_bin_locations,
                                                                                                                predicted_posteriors,
                                                                                                                actual_values,
                                                                                                                bin_ranges,
                                                                                                                plot=False,
                                                                                                                nbins=self.inputs['nbins'])
        
        self.errors = {'distance_errors': distance_errors,
                       'norm_distance_errors': norm_distance_errors,
                       'output_bin_means': output_bin_means}
        
    def save(self, path):
        with open(path, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
