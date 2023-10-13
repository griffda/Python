import bnmodel_copy as bn
from bnmodel_copy.join_tree_population import evidence
from pybbn.pptc.inferencecontroller import InferenceController
from sklearn.model_selection import train_test_split
from pybbn.graph.jointree import JoinTree
from pybbn.graph.jointree import Evidence, EvidenceBuilder, EvidenceType
import pandas as pd
import pickle
import json
from sklearn.model_selection import KFold
import numpy as np
import os

class BayesianNetwork:
    def __init__(self, inputs):
        self.inputs = inputs
        # self.__load_inputs()
        ## CASE: load model from already built BN
        if self.inputs['data'] != None:
            print('model data supplied')
            self.__load_inputs()
        ## CASE: build new model from data supplied via BNdata and netstructure
        else:
            print('no model data supplied')
            self.__load_inputs()
        if isinstance(self.inputs['output'], list):
            self.inputs['output'] = self.inputs['output'][0]
    
        

    def __load_inputs(self):
        """
        Prepare the inputs for the training

        'uniform' method: this is when no k-fold cross-validation is used
        'kfold' method: this is when k-fold cross-validation is used
        'meta' method: this is when the model is used to predict the output of a real case

        """
        # If model data is supplied, load the data
        data = bn.utilities.prepare_csv(self.inputs['data'])

        if self.inputs['method'] == 'uniform': # No k-fold cross-validation
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

        elif self.inputs['method'] == 'kfold': # k-fold cross-validation
            x, y, bin_edges, prior_xytrn = bn.discretisation.binning_data(data,
                                                                          self.inputs['nbins'],
                                                                          self.inputs['inputs'],
                                                                          self.inputs['output'])
            self.bin_edges = bin_edges
            self.prior_xytrn = prior_xytrn
            self.data = data 
            
            # k-fold cross-validation
            kf = KFold(n_splits=self.inputs['nfolds'], 
                    shuffle=True, 
                    random_state=42) 
            kf.get_n_splits(x)

            self.folds = {}

            fold_counter = 0
            for train_index, test_index in kf.split(x): # Split the data into training and testing sets
                x_train, x_test = x.iloc[train_index], x.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]

                # Combine the binned data into a single DataFrame for each set
                train_binned = pd.concat([x_train, y_train], axis=1)
                train_binned = train_binned.astype(str)
                test_binned = pd.concat([x_test, y_test], axis=1)


                fold_name = 'fold_' + str(fold_counter)
                self.folds[fold_name] = {'train_binned': train_binned,
                                         'test_binned': test_binned}
                fold_counter += 1
            
        elif self.inputs['method'] == 'meta': # This is when the model is used to predict the output of a real case
            # discretise the data
            x, y, bin_edges, prior_xytrn = bn.discretisation.binning_data(data,
                                                                self.inputs['nbins'],
                                                                self.inputs['inputs'],
                                                                self.inputs['output'])
            self.bin_edges = bin_edges
            self.prior_xytrn = prior_xytrn
            self.data = data
            #concatenate inputd and outputs to create a single dataframe
            self.train_binned = pd.concat([x, y], axis=1)
            self.train_binned = self.train_binned.astype(str)
        else:
            raise ValueError('Invalid method for discretisation')

        self.struct = bn.utilities.df2struct(data, self.inputs['inputs'], self.inputs['output'])

    def train(self):
        """
        train the model depending on the method of validation chosen

        """
        if self.inputs['method'] == 'uniform':
            self.join_tree, self.bbn = bn.join_tree_population.prob_dists(self.struct, self.train_binned)

        elif self.inputs['method'] == 'kfold':
            for fold in self.folds:
                self.folds[fold]['join_tree'], self.bbn = bn.join_tree_population.prob_dists(self.struct, self.folds[fold]['train_binned'])
        elif self.inputs['method'] == 'meta': #need to add a section for meta where we use all the data to train the model and no validation is done
            #populate the join tree
            self.join_tree, self.bbn = bn.join_tree_population.prob_dists(self.struct, self.train_binned)
        else:
            raise ValueError('Invalid method for discretisation')
        
    def validate(self):
        """
        validate the model depending on the method of validation chosen

        """
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
            # self.obs_posteriors = obs_posteriors
            self.all_ev_list = all_ev_list
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
            

            # self.errors = {'norm_distance_errors': np.mean([self.folds[fold]['errors']['norm_distance_errors'] for fold in self.folds]),
            #             'prediction_accuracy': np.mean([self.folds[fold]['errors']['prediction_accuracy'] for fold in self.folds])}

        else:
            raise ValueError('Invalid method for discretisation')


    def save(self, path): #this is where we want to save the join_tree
        """
        Saves the join_tree to a json file
        Saves the bin_edges to a json file 
        Saves the model to a pickle file or json file - need to decide which one

        """
        # Convert the JoinTree object to a serializable format
        with open(self.inputs['save_join_tree_path'], 'w') as f:
            d = JoinTree.to_dict(self.join_tree, self.bbn) #self.bbn is the bayesian network object 
            j = json.dumps(d, sort_keys=True, indent=2)
            f.write(j)

        
        # Convert NumPy arrays to lists in the dictionary
        bin_edges_serializable = {key: self.bin_edges[key].tolist() for key in self.bin_edges}

        # Save bin_ranges as JSON
        bin_ranges_path = os.path.join(path, self.inputs['save_bin_edges_path'])
        with open(bin_ranges_path, 'w') as bins_json:
            json.dump(bin_edges_serializable, bins_json, ensure_ascii=False, indent=4)

        # Save the model to a pickle file
        # with open(path, 'wb') as outp:
        #     pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

    def run_model(self):
        """
        data_path: path to csv file - it should not need this. 

        what to get out: 
        1. posteriors based on evidence

        what needs to go in:
        1. data path
        2. join_tree from the training
        3. evidence (inputs)
        4. output (to get the posteriors)
        

        TODO: add any inputs and outputs to the model
        TODO: ranges for inputs (no hard evidence)
        TODO: clean this function
        """
        # ask which method is used from inputs
        if self.inputs['method'] == 'uniform':
            pass

        elif self.inputs['method'] == 'kfold':
            pass

        elif self.inputs['method'] == 'meta':
            # load the evidence
            # observations = self.inputs['evidence']
            
            # load the join tree from json file
            with open(self.inputs['load_join_tree_path'], 'r') as f:
                j = f.read()
                d = json.loads(j)
                jt = JoinTree.from_dict(d)
                join_tree = InferenceController.apply_from_serde(jt)
                
            # load the bin edges from json file
            with open(self.inputs['load_bin_edges_path'], 'r') as json_file:
                bin_edges = json.load(json_file)

            if self.inputs['evidence'] != None: #this is when we want to run with hard evidence
                observations = self.inputs['evidence']
                # inference using the evidence supplied in the inputs
                for observation in observations:
                    print(observation)
                    node_name = observation['nod']
                    bin_index = observation['bin_index']
                    value = observation['val']
                    ev4jointree = bn.join_tree_population.evidence(node_name, int(bin_index), value, join_tree)
                    join_tree.set_observation(ev4jointree)
                    join_tree.unobserve_all
                    self.join_tree = join_tree
                # update the join tree with the evidence
                aux_obs, aux_prd = bn.generate_posteriors.get_posteriors(join_tree, self.inputs['output'])
                self.obs_posteriors = aux_obs
    
                # plot the posteriors
                bn.plotting.plot_meta(self.obs_posteriors, bin_edges, self.prior_xytrn, self.inputs['inputs'], self.inputs['output'], 5)
            

            else: #this is when we want to run with soft evidence across multiple bins
                # Define evidence for multiple bins on the same node
                evidence = {
                    'capcost': [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    # Add more nodes and evidence values as needed
                }

                # evidence = self.inputs['evidence2']

                # Loop through the evidence dictionary and set evidence for each node
                for node_name, values in evidence.items():
                    node = join_tree.get_bbn_node_by_name(node_name)
                    if node:
                        for bin_index, value in enumerate(values):
                            evidence_builder = EvidenceBuilder().with_node(node).with_type(EvidenceType.OBSERVATION)
                            evidence_builder = evidence_builder.with_evidence(bin_index, value)
                            evidence = evidence_builder.build()
                            join_tree.set_observation(evidence)
                            join_tree.unobserve_all
                            self.join_tree = join_tree
                    else:
                        print(f"Node '{node_name}' not found in the join tree.")
                
                aux_obs, aux_prd = bn.generate_posteriors.get_posteriors(join_tree, self.inputs['output'])
                self.obs_posteriors = aux_obs

                join_tree.update_evidences([evidence])

                # plot the posteriors
                bn.plotting.plot_meta(self.obs_posteriors, bin_edges, self.inputs['inputs'], self.inputs['output'])
        else:
            raise ValueError('Invalid method')