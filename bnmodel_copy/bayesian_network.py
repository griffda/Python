import bnmodel_copy as bn
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
from sklearn.model_selection import KFold

class BayesianNetwork:
    def __init__(self, inputs):
        self.inputs = inputs  
        self.__load_inputs()
        if isinstance(self.inputs['output'], list):
            self.output = self.inputs['output'][0]

    def __load_inputs(self):
        """
        Prepare the inputs for the training

        """
        # Load the data
        data = bn.utilities.prepare_csv(self.inputs['data'])

        if self.inputs['method'] == 'uniform':
            x, y, bin_edges, prior_xytrn = bn.discretisation.binning_data(data,
                                                                          self.inputs['nbins'],
                                                                          self.inputs['inputs'],
                                                                          self.inputs['output'])
            self.bin_edges = bin_edges
            self.prior_xytrn = prior_xytrn
            self.data = data
    

            # Split the data into training and testing sets
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.inputs['train_test_split'])
        
            # Combine the binned data into a single DataFrame for each set
            # Pybbn only reads data types as strings, so this line converts the data in the csv from int64 to string
            train_binned = pd.concat([x_train, y_train], axis=1)
            train_binned = train_binned.astype(str)

            self.train_binned = train_binned
            self.test_binned = pd.concat([x_test, y_test], axis=1)

        elif self.inputs['method'] == 'kfold':
            self.data = data # TODO: Check if this can be out of the if
            x, y, bin_edges, prior_xytrn = binning_data(data, nbins=nbins, y_cols=[output])
            self.bin_edges = bin_edges
            self.prior_xytrn = prior_xytrn

            kf = KFold(n_splits=self.inputs['nfolds'], 
                    shuffle=True, 
                    random_state=42) 
            kf.get_n_splits(x)

            self.folds = {}

            fold_counter = 0
            for train_index, test_index in kf.split(x):
                x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Combine the binned data into a single DataFrame for each set
                train_binned = pd.concat([x_train, y_train], axis=1)
                test_binned = pd.concat([x_test, y_test], axis=1)


                fold_name = 'fold_' + str(fold_counter)
                self.folds[fold_counter] = {'train_binned': train_binned, 'test_binned': test_binned}
                fold_counter += 1

        else:
            raise ValueError('Invalid method for discretisation')

        self.struct = bn.utilities.df2struct(data, self.inputs['inputs'], self.inputs['output'])

    def train(self):
        if self.inputs['method'] == 'uniform':
            self.join_tree = bn.join_tree_population.prob_dists(self.struct, self.train_binned)

        elif self.inputs['method'] == 'kfold':
            for fold in self.folds:
                self.folds[fold]['join_tree'] = bn.join_tree_population.prob_dists(self.struct, self.folds[fold]['train'])

        else:
            raise ValueError('Invalid method for discretisation')
        
    def validate(self):
        if hasattr(self, 'join_tree') is False:
            raise ValueError('Model not trained yet')

        if self.inputs['method'] == 'uniform':
            # Get the posteriors for the testing subset
            # TODO: create a run_model function. Obs_posteriors should be in a different function
            obs_dicts = bn.generate_posteriors.generate_multiple_obs_dicts(self.test_binned, 
                                                                        self.inputs['output'], 
                                                                        self.data)
            all_ev_list = bn.generate_posteriors.gen_ev_list(self.test_binned,
                                                            obs_dicts,
                                                            self.inputs['output'])
            obs_posteriors, predicted_posteriors = bn.generate_posteriors.get_all_posteriors(all_ev_list,
                                                                                            self.join_tree,
                                                                                            self.inputs['output'])

            # Error evaluation
            correct_bin_locations, actual_values = bn.evaluate_errors.get_correct_values(obs_dicts, self.inputs['output'])
            bin_ranges = bn.evaluate_errors.extract_bin_ranges(self.inputs['output'], self.bin_edges)
            norm_distance_errors, prediction_accuracy = bn.evaluate_errors.distribution_distance_error(correct_bin_locations,
                                                                                                        predicted_posteriors,
                                                                                                        actual_values,
                                                                                                        bin_ranges)
                                                                                                                    
            self.errors = {'norm_distance_errors': norm_distance_errors,
                        'prediction_accuracy': prediction_accuracy}
        
        elif self.inputs['method'] == 'kfold':
            for fold in self.folds:
                obs_dicts = bn.generate_posteriors.generate_multiple_obs_dicts(self.folds[fold]['test_binned'], 
                                                                        self.inputs['output'], 
                                                                        self.data)
                all_ev_list = bn.generate_posteriors.gen_ev_list(self.folds[fold]['test_binned'],
                                                                obs_dicts,
                                                                self.inputs['output'])
                obs_posteriors, predicted_posteriors = bn.generate_posteriors.get_all_posteriors(all_ev_list,
                                                                                                self.folds[fold]['join_tree'],
                                                                                                self.inputs['output'])

                # Error evaluation
                correct_bin_locations, actual_values = bn.evaluate_errors.get_correct_values(obs_dicts, self.inputs['output'])
                bin_ranges = bn.evaluate_errors.extract_bin_ranges(self.inputs['output'], self.bin_edges)
                norm_distance_errors, prediction_accuracy = bn.evaluate_errors.distribution_distance_error(correct_bin_locations,
                                                                                                            predicted_posteriors,
                                                                                                            actual_values,
                                                                                                            bin_ranges)

                self.folds[fold]['errors'] = {'norm_distance_errors': norm_distance_errors,
                                            'prediction_accuracy': prediction_accuracy}
            

            self.errors = {'norm_distance_errors': np.mean([self.folds[fold]['errors']['norm_distance_errors'] for fold in self.folds]),
                        'prediction_accuracy': np.mean([self.folds[fold]['errors']['prediction_accuracy'] for fold in self.folds])}

        else:
            raise ValueError('Invalid method for discretisation')

    def run_model(self, data, input_names, output_names):
        """
        data_path: path to csv file


        TODO: add any inputs and outputs to the model
        TODO: ranges for inputs (no hard evidence)
        TODO: clean this function
        """
        data = bn.utilities.prepare_csv(data_path)

        if hasattr(self, 'join_tree') is False:
            raise ValueError('Model not trained yet')
        
        if self.inputs['method'] == 'uniform':
            observations = data['input_names']

            obs_dicts = bn.generate_posteriors.generate_multiple_obs_dicts(observations, 
                                                                            output_names, 
                                                                            data)
            all_ev_list = bn.generate_posteriors.gen_ev_list(observations,
                                                            obs_dicts,
                                                            output_names)
            obs_posteriors, predicted_posteriors = bn.generate_posteriors.get_all_posteriors(all_ev_list,
                                                                                            self.join_tree,
                                                                                            output_names)

            return predicted_posteriors

        elif self.inputs['method'] == 'kfold':
            aux = {}
            for fold in self.folds:
                obs_dicts = bn.generate_posteriors.generate_multiple_obs_dicts(observations, 
                                                                        output_names, 
                                                                        data)
                all_ev_list = bn.generate_posteriors.gen_ev_list(observations,
                                                                obs_dicts,
                                                                output_names)
                obs_posteriors, predicted_posteriors = bn.generate_posteriors.get_all_posteriors(all_ev_list,
                                                                                                self.folds[fold]['join_tree'],
                                                                                                output_names)
                aux[fold] = predicted_posteriors

            return np.mean([aux[fold] for fold in aux], axis=0)


        else:
            raise ValueError('Invalid method for discretisation')


    def save(self, path):
        with open(path, 'wb') as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)
