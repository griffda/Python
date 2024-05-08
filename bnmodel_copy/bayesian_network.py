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
import json
import ast

class BayesianNetwork:
    def __init__(self, inputs):
        if isinstance(inputs, str):
            self.inputs = json.loads(inputs)
        else:
            self.inputs = inputs

        if self.inputs['data'] != None:
            print('model data supplied')
            self.__load_inputs()
        else:
            print('no model data supplied')
            self.__load_inputs()

        if isinstance(self.inputs['output'], list):
            self.inputs['output'] = self.inputs['output']
            pass

    def __load_inputs(self):
        """
        Prepare the inputs for the training

        'uniform' method: this is when no k-fold cross-validation is used
        'kfold' method: this is when k-fold cross-validation is used
        'meta' method: this is when the model is used to predict the output of a real case

        """
        if isinstance(self.inputs['data'], str):
            data = bn.utilities.prepare_csv(self.inputs['data'], self.inputs['inputs'], self.inputs['output'])
        elif isinstance(self.inputs['data'], dict):
            data = pd.DataFrame(self.inputs['data'])
        else:
            raise ValueError('Invalid data format')

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
            test_binned = pd.concat([x_test, y_test], axis=1)
            self.test_binned = test_binned

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
            self.train_binned = train_binned


        elif self.inputs['method'] == 'meta': # This is when the model is used to predict the output of a real case
            # discretise the data
            # x, y, bin_edges, prior_xytrn = bn.discretisation.binning_data(data,
            #                                                     self.inputs['nbins'],
            #                                                     self.inputs['inputs'],
            #                                                     self.inputs['output'])
            # self.bin_edges = bin_edges
            # self.prior_xytrn = prior_xytrn
            # self.data = data

            # #concatenate inputd and outputs to create a single dataframe
            # self.train_binned = pd.concat([x, y], axis=1)
            # self.train_binned = self.train_binned.astype(str)

            x, y, bin_edges, prior_xytrn = bn.discretisation.binning_data(data,
                                                                          self.inputs['nbins'],
                                                                          self.inputs['inputs'],
                                                                          self.inputs['output'])
            self.bin_edges = bin_edges
            self.prior_xytrn = prior_xytrn
            self.data = data
            # print('data: ', data)   


            # Split the data into training and testing sets
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=self.inputs['train_test_split'])

            # Combine the binned data into a single DataFrame for each set
            # Pybbn only reads data types as strings, so this line converts the data in the csv from int64 to string
            train_binned = pd.concat([x_train, y_train], axis=1)
            train_binned = train_binned.astype(str)

            self.train_binned = train_binned
            test_binned = pd.concat([x_test, y_test], axis=1)
            self.test_binned = test_binned


        else:
            raise ValueError('Invalid method for discretisation')

        self.struct = bn.utilities.df2struct(data, self.inputs['inputs'], self.inputs['output'])
        # print('inputs: ', self.inputs['inputs'])
        # print('output: ', self.inputs['output'])
        # print(self.struct)

    def train(self):
        """
        train the model depending on the method of validation chosen

        """
        if self.inputs['method'] == 'uniform':
            self.join_tree, self.bbn = bn.join_tree_population.prob_dists(self.struct, self.train_binned)
            print('Join-tree created, BN configured')

        elif self.inputs['method'] == 'kfold':
            for fold in self.folds:
                self.folds[fold]['join_tree'], self.bbn = bn.join_tree_population.prob_dists(self.struct, self.folds[fold]['train_binned'])
                print(f'Join tree for fold {fold} created')
            self.join_tree, self.bbn = bn.join_tree_population.prob_dists(self.struct, self.train_binned)
        elif self.inputs['method'] == 'meta': #need to add a section for meta where we use all the data to train the model and no validation is done
            #populate the join tree
            self.join_tree, self.bbn = bn.join_tree_population.prob_dists(self.struct, self.train_binned)
            obs_dicts = bn.generate_posteriors.generate_multiple_obs_dicts(self.test_binned,
                                                                            self.inputs['output'],
                                                                            self.data)
            all_ev_list = bn.generate_posteriors.gen_ev_list(self.test_binned,
                                                                obs_dicts,
                                                                self.inputs['output'])
            self.all_ev_list = all_ev_list
            print('Join-tree created, BN configured')
        else:
            raise ValueError('Invalid method for discretisation')

    def validate(self):
        """
        validate the model depending on the method of validation chosen

        """
        norm_distance_errors_list = []  # Create an empty list to store all norm_distance_errors arrays
        prediction_accuracy_list = []  # Create an empty list to store all prediction_accuracy values

        if self.inputs['method'] == 'uniform':
            # Get the posteriors for the testing subset
            # TODO: create a run_model function. Obs_posteriors should be in a different function
            obs_dicts = bn.generate_posteriors.generate_multiple_obs_dicts(self.test_binned,
                                                                        self.inputs['output'],
                                                                        self.data)
            all_ev_list = bn.generate_posteriors.gen_ev_list(self.test_binned,
                                                            obs_dicts,
                                                            self.inputs['output'])
            obs_posteriors, predicted_posteriors, target_predictions = bn.generate_posteriors.get_all_posteriors(all_ev_list,
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
                                                                                                        bin_ranges, self.inputs['error_type'])

            self.errors = {'norm_distance_errors': norm_distance_errors,
                        'prediction_accuracy': prediction_accuracy}

        elif self.inputs['method'] == 'kfold':
            for fold in self.folds:
                obs_dicts = bn.generate_posteriors.generate_multiple_obs_dicts(self.folds[fold]['test_binned'],
                                                                        self.inputs['output'],
                                                                        self.data)
                # print('obs_dicts: ', obs_dicts)
                all_ev_list = bn.generate_posteriors.gen_ev_list(self.folds[fold]['test_binned'],
                                                                obs_dicts,
                                                                self.inputs['output'])
                self.all_ev_list = all_ev_list

                #print('all_ev_list: ', all_ev_list)
                obs_posteriors, predicted_posteriors, target_predictions = bn.generate_posteriors.get_all_posteriors(all_ev_list,
                                                                                                self.folds[fold]['join_tree'],
                                                                                                self.inputs['output'])

                # Error evaluation
                correct_bin_locations, actual_values = bn.evaluate_errors.get_correct_values(obs_dicts, self.inputs['output'])


                bin_ranges = bn.evaluate_errors.extract_bin_ranges(self.inputs['output'], self.bin_edges)


                norm_distance_errors, prediction_accuracy = bn.evaluate_errors.distribution_distance_error(correct_bin_locations,
                                                                                                            predicted_posteriors,
                                                                                                            actual_values,
                                                                                                            bin_ranges, self.inputs['error_type'])

                self.folds[fold]['errors'] = {'norm_distance_errors': norm_distance_errors,
                                            'prediction_accuracy': prediction_accuracy}

                norm_distance_errors_list.append(norm_distance_errors)
                prediction_accuracy_list.append(prediction_accuracy)

                # print("norm_distance_errors: ", norm_distance_errors_list)
                # print("prediction_accuracy: ", prediction_accuracy_list)

            self.errors = {
                'norm_distance_errors': np.mean([error for fold in self.folds for error in self.folds[fold]['errors']['norm_distance_errors']]),
                'prediction_accuracy': np.mean([self.folds[fold]['errors']['prediction_accuracy'] for fold in self.folds])
            }

            bn.plotting.plot_errors(norm_distance_errors_list, self.inputs['kfoldnbins'], prediction_accuracy_list, self.errors['prediction_accuracy'])

        else:
            raise ValueError('Invalid method for discretisation')

    # def sensitivity_analysis(self): #this is where we want to run the model with different bin sizes and see how the prediction accuracy changes
    #     accuracies = []
    #     start, end = self.inputs['nbins_sensitivity_range']
    #     for nbins in range(start, end):
    #         self.inputs['nbins'] = [{'inputs': nbins, 'output': nbins}]
    #         self.train()
    #         self.validate()
    #         accuracies.append(self.errors['prediction_accuracy'])
    #         print(accuracies)
    #     return accuracies

    # def sensitivity_analysis(self):
    #     results = {}
    #     start, end = self.inputs['nbins_sensitivity_range']
    #     for nbins in range(start, end):
    #         self.inputs['nbins'] = [{'inputs': nbins, 'output': nbins}]
    #         self.train()
    #         self.validate()
    #         bin_config = f'inputs: {nbins}, output: {nbins}'
    #         # accuracies[bin_config] = self.errors['prediction_accuracy']
    #         results[bin_config] = (nbins, nbins, self.errors['prediction_accuracy'])
    #         # print(bin_config, results[bin_config])
    #     with open('sa_results5k.pkl', 'wb') as f:
    #         pickle.dump(results, f)
    #     return results

    def sensitivity_analysis(self):
        results = {}
        start, end = self.inputs['nbins_sensitivity_range']
        for nbins_input in range(start, end):
            for nbins_output in range(start, end):
                self.inputs['nbins'] = [{'inputs': nbins_input, 'output': nbins_output}]
                self.train()
                self.validate()
                bin_config = f'inputs: {nbins_input}, output: {nbins_output}'
                results[bin_config] = (nbins_input, nbins_output, self.errors['prediction_accuracy'])
                # print(bin_config, results[bin_config])
        with open('sa_results_process_data_D2.pkl', 'wb') as f:
            pickle.dump(results, f)
        return results

    # bn.plotting.plot_sensitivity_analysis(results)

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

        # Save all_ev_list to a pickle file
        with open(self.inputs['save_evidence_path'], 'wb') as f:
            pickle.dump(self.all_ev_list, f)

    def run_model(self): #old routine
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
                evidence_vars = []
                # inference using the evidence supplied in the inputs
                for observation in observations:
                    node_name = observation['nod']
                    bin_index = observation['bin_index']
                    value = observation['val']
                    ev4jointree = bn.join_tree_population.evidence(node_name, int(bin_index), value, join_tree)
                    join_tree.set_observation(ev4jointree)
                    join_tree.unobserve_all
                    self.join_tree = join_tree
                    evidence_vars.append(node_name)
                # update the join tree with the evidence
                aux_obs, aux_prd = bn.generate_posteriors.get_posteriors(join_tree, self.inputs['output'])
                self.obs_posteriors = aux_obs


                # plot the posteriors
                bn.plotting.plot_meta(self.obs_posteriors, bin_edges, self.prior_xytrn, self.inputs['inputs'], self.inputs['output'], evidence_vars, 3)



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
                bn.plotting.plot_meta2(self.obs_posteriors, bin_edges, self.inputs['inputs'], self.inputs['output'])
        else:
            raise ValueError('Invalid method')

    def run_model2(self):#latest routine
        """
        data_path: path to csv file - it should not need this.
        also trying new plotting routine

        what to get out:
        1. posteriors based on evidence

        what needs to go in:
        1. should only be used in meta method
        2. join_tree from the training - load from json file
        3. evidence (inputs) - hard, soft, optimal_config or hybrid
            - hard: evidence for a single bin
            - soft: evidence for multiple bins:
                inputs = {
                    'evidence': {
                        'node1': [0.1, 0.2, 0.7],  # Evidence for node1
                        'node2': [0.3, 0.3, 0.4],  # Evidence for node2
                        # Add more nodes and evidence values as needed
                    },
                        # Other inputs...
                    }
            - optimal_config: evidence for the highest probability prediction for the lowest bin possible
        4. output (to get the posteriors)

        """
                # ask which method is used from inputs
        if self.inputs['method'] == 'uniform':
            pass

        elif self.inputs['method'] == 'kfold':
            pass

        elif self.inputs['method'] == 'meta':

            # load the join tree from json file
            with open(self.inputs['load_join_tree_path'], 'r') as f:
                j = f.read()
                d = json.loads(j)
                jt = JoinTree.from_dict(d)
                join_tree = InferenceController.apply_from_serde(jt)
                print('Join tree loaded from file')

            # load the bin edges from json file
            with open(self.inputs['load_bin_edges_path'], 'r') as json_file:
                bin_edges = json.load(json_file)
                print('Bin edges loaded from file')

            outputs = self.inputs['output']
            inputs = self.inputs['inputs']
            # inference_type = self.inputs['inference_type']
            
            evidence_type = self.inputs['evidence_type']

            if evidence_type == 'hard':
                print(evidence_type + ' evidence type chosen, runnning model for a single case')
                if 'evidence' in self.inputs and self.inputs['evidence'] is not None:
                    # Hard evidence
                    observations = self.inputs['evidence']
                    evidence_vars = []
                    for observation in observations:
                        node_name = observation['nod']
                        bin_index = observation['bin_index']
                        value = observation['val']
                        ev4jointree = bn.join_tree_population.evidence(node_name, int(bin_index), value, join_tree)
                        join_tree.set_observation(ev4jointree)
                        join_tree.unobserve_all
                        self.join_tree = join_tree
                        evidence_vars.append(node_name)
                    # Generate posteriors and plot
                    aux_obs, aux_prd = bn.generate_posteriors.get_posteriors(join_tree, self.inputs['output'])
                    self.obs_posteriors = aux_obs
                    print(self.obs_posteriors)
                    # print(self.inputs['inputs'], self.inputs['output'])
                    # print(evidence_vars)
                    bn.plotting.plot_meta3(self.obs_posteriors, bin_edges, self.prior_xytrn, self.inputs['inputs_plotting'], self.inputs['output_plotting'], evidence_vars, 3)
                else:
                    print("'evidence' not provided or is None. Skipping sequence.")

            elif evidence_type == 'soft': # this uses a different format for the evidence
                if 'evidence_soft' in self.inputs and self.inputs['evidence_soft'] is not None:
                    # Soft evidence
                    evidence = self.inputs['evidence_soft']
                    for node_name, values in evidence.items():
                        node = join_tree.get_bbn_node_by_name(node_name)
                        if node:
                            for bin_index, value in enumerate(values, start=1):
                                evidence_builder = EvidenceBuilder().with_node(node).with_type(EvidenceType.OBSERVATION)
                                evidence_builder = evidence_builder.with_evidence(str(bin_index), value)
                                evidence = evidence_builder.build()
                                print(bin_index, value, node_name, values)
                                join_tree.set_observation2(evidence)
                                join_tree.unobserve_all
                                self.join_tree = join_tree
                        else:
                            print(f"Bin index '{bin_index}' not found in the potentials for node '{node_name}'.")
                    # Generate posteriors and plot
                    aux_obs, aux_prd = bn.generate_posteriors.get_posteriors(join_tree, self.inputs['output'])
                    self.obs_posteriors = aux_obs

                    bn.plotting.plot_meta(self.obs_posteriors, bin_edges, self.prior_xytrn, self.inputs['inputs'], self.inputs['output'], 3)
                else:
                    print("'evidence_soft' not provided or is None. Skipping sequence.")


            elif evidence_type == 'soft2': # trying to use same dict format as hard evidence
                if 'evidence' in self.inputs and self.inputs['evidence'] is not None:
                # Soft evidence
                    observations = self.inputs['evidence']
                    for observation in observations:
                        node_name = observation['nod']
                        bin_index = observation['bin_index']
                        value = observation['val']
                        ev4jointree = bn.join_tree_population.evidence(node_name, int(bin_index), value, join_tree)
                        print(bin_index, value, node_name)
                        join_tree.set_observation2(ev4jointree)
                        print(ev4jointree)
                        #join_tree.unobserve_all  # Unobserve all after setting all the evidence
                        self.join_tree = join_tree
                    for node, posteriors_raw in join_tree.get_posteriors().items():
                        posteriors = {int(k): v for k, v in posteriors_raw.items()}
                        print(node, posteriors)
                    aux_obs, aux_prd = bn.generate_posteriors.get_posteriors(join_tree, self.inputs['output'])
                    print(aux_obs)
                    self.obs_posteriors = aux_obs
                    bn.plotting.plot_meta(self.obs_posteriors, bin_edges, self.prior_xytrn, self.inputs['inputs'], self.inputs['output'], 3)
                else:
                    print("'evidence' not provided or is None. Skipping sequence.")     
            
            # elif 'optimal_config' in evidence_type:
            elif evidence_type == 'optimal_config': # this uses a different format for the evidence
                analysis_type = self.inputs['analysis_type'][0]  # Get the first item from the analysis_type list

                ###forward inference optimal config type
                if 'optimal_config' in analysis_type and analysis_type['optimal_config'] is not None: 
                    print('Optimal config type chosen: ' + str(analysis_type))
                    # Generate all evidence combinations
                    all_evidence_combinations = bn.generate_posteriors.generate_all_evidence_combinations2(self.inputs)
                    print('All evidence combinations: ', len(all_evidence_combinations))

                    _, _, all_target_predictions = bn.generate_posteriors.get_all_posteriors_new(all_evidence_combinations, join_tree, self.inputs['output'])             
                    print(all_target_predictions)

                    # Extract the optimal_config dictionary from the analysis_type
                    optimal_config = analysis_type['optimal_config']

                    # Iterate over the optimal_config dictionary
                    for output_variable, config_type in optimal_config.items():
                        # Check if the output_variable exists in all_target_predictions
                        if output_variable in all_target_predictions:
                            best_evidence = bn.generate_posteriors.find_best_evidence_for_bin(all_target_predictions[output_variable], config_type)
                            print(f"Best evidence for {output_variable} ({config_type}): {best_evidence}")
                        else:
                            print(f"No predictions found for {output_variable}")
                    
                    best_evidence = ast.literal_eval(best_evidence)
                    print(best_evidence)
                    observations = best_evidence
                    
                    # Generate posteriors of best evidence and plot
                    evidence_vars = []
                    join_tree.unobserve_all()
                    for observation in observations:
                        
                        node_name = observation['nod']
                        bin_index = observation['bin_index']
                        value = observation['val']
                        # print(node_name, bin_index, value)
                        ev4jointree = bn.join_tree_population.evidence(node_name, int(bin_index), value, join_tree)
                        join_tree.set_observation(ev4jointree)
                        self.join_tree = join_tree
                        evidence_vars.append(node_name)

                        # Generate posteriors and plot
                    aux_obs, _, = bn.generate_posteriors.get_posteriors(self.join_tree, self.inputs['output'])
                    self.obs_posteriors = aux_obs
                    print(self.obs_posteriors)
                    bn.plotting.plot_meta2(self.obs_posteriors, bin_edges, self.prior_xytrn, self.inputs['inputs'], self.inputs['output'], evidence_vars, 3)
                else:
                    print("Optimal config not provided or is None. Skipping sequence.")

                ### Reverse inference optimal config        
                if 'reverse_optimal' in analysis_type: 
                    print('Optimal config type chosen: ' + str(analysis_type))

                    all_evidence_combinations = bn.generate_posteriors.generate_all_evidence_combinations2(self.inputs)
                    print('All evidence combinations: ', len(all_evidence_combinations))

                    _, _, all_target_predictions = bn.generate_posteriors.get_all_posteriors_new(all_evidence_combinations, join_tree, self.inputs['inputs'])
                    print(all_target_predictions)

                    # Extract the strongest posteriors from the all_target_predictions
                    for input_variable in inputs:
                        if input_variable in all_target_predictions:
                            max_probability = float('-inf')
                            best_evidence = None
                            for evidence, (probability, bin_index) in all_target_predictions[input_variable].items():
                                ##there is no min or max config type for inputs so we can just use the max config type
                                if probability > max_probability:
                                    max_probability = probability
                                    best_evidence = evidence
                                    print(f"Best evidence for {input_variable} ({analysis_type}): {best_evidence}")
                    best_evidence = ast.literal_eval(best_evidence)
                    print(best_evidence)
                    observations = best_evidence
                
                # Generate posteriors of best evidence and plot
                evidence_vars = []
                join_tree.unobserve_all()
                for observation in observations:
                    
                    node_name = observation['nod']
                    bin_index = observation['bin_index']
                    value = observation['val']
                    # print(node_name, bin_index, value)
                    ev4jointree = bn.join_tree_population.evidence(node_name, int(bin_index), value, join_tree)
                    join_tree.set_observation(ev4jointree)
                    self.join_tree = join_tree
                    evidence_vars.append(node_name)

                    # Generate posteriors and plot
                aux_obs, _, = bn.generate_posteriors.get_posteriors(self.join_tree, self.inputs['output'])
                self.obs_posteriors = aux_obs
                print(self.obs_posteriors)
                bn.plotting.plot_meta2(self.obs_posteriors, bin_edges, self.prior_xytrn, self.inputs['inputs'], self.inputs['output'], evidence_vars, 3)
            else:
                raise ValueError('Invalid evidence type')
            
