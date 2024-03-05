from bnmodel.join_tree_population import evidence
import pandas as pd
import itertools


def generate_obs_dict(test_binned, output, data):
    """
    Generate a single observation from the test dataset

    Parameters
    ----------
    test_binned : pandas dataframe discretised test dataset
    output : str target/output variable
    data : pandas dataframe with all the data

    Returns
    -------
    obs_dict : observation dictionary
    """

    # choose a random row from the test_binned
    row = test_binned.sample()

    # generate an obs_dict from the chosen row
    obs_dict = {}

    if isinstance(output, list):
        output = output[0]  # Convert the list to a single string if it is a list

    for col in test_binned.columns:
        if col == output:
            obs_dict[col] = {'bin_index': str(row[col].values[0]), 'actual_value': data[output][row.index.values].values[0]}
        else:
            obs_dict[col] = {'bin_index': str(row[col].values[0]), 'val': 1.0}

    # print("Observation dictionary:", obs_dict)
    return obs_dict

def generate_multiple_obs_dicts(test_binned, output, data):
    """
    Generate num_samples observations form the test dataset

    Parameters
    ----------
    test_binned : pandas dataframe discretised test dataset
    output : str target/output variable
    data : pandas dataframe with all the data

    Returns
    -------
    obs_dicts : list of observation dictionaries
    """
    obs_dicts = []

    if isinstance(output, list):
        output = output[0]  # Convert the list to a single string if it is a list

    for i in range(len(test_binned)):
        obs_dict = generate_obs_dict(test_binned, output, data)
        obs_dicts.append(obs_dict)
    # print("Observation dictionaries:", obs_dicts)
    return obs_dicts


def gen_ev_list(test_binned, obs_dicts, output):
    """
    Parameters
    ----------
    test_binned : pandas dataframe discretised test dataset
    obs_dicts : list of observation dictionaries
    output : str target/output variable
    """
    test_binned = test_binned.drop(output, axis=1)
    all_ev_list = []

    for obs in obs_dicts:
        ev_list = []
        for col in test_binned.columns:
            bin_index = obs[col]['bin_index']
            val = obs[col]['val']
            ev_dict = {'nod':col, 'bin_index':bin_index, 'val': val}
            ev_list.append(ev_dict)
        all_ev_list.append(ev_list)
    # print("All evidence lists:", all_ev_list)
    return all_ev_list


def get_posteriors(join_tree, output):
    """
    Get the posteriors for the observations included in the join tree.

    Parameters
    ----------
    join_tree : conditional probability table
    output : str target/output variable
    """
    obs_posteriors = {}
    predictedTargetPosteriors = []

    if isinstance(output, list):
        output = output[0]  # Convert the list to a single string if it is a list
    for node, posteriors_raw in join_tree.get_posteriors().items():
        obs_posteriors[node] = [posteriors_raw[val] for val in posteriors_raw]
        # print(node)
        # print(output)
        if node == output:  # check if the observation corresponds to the specified target variable
            # print(node)
            # print(output)
            predictedTargetPosteriors = [posteriors_raw[val] for val in posteriors_raw]
    # print("obs_posteriors:", obs_posteriors)
    return obs_posteriors, predictedTargetPosteriors



def get_all_posteriors(all_ev_list, join_tree, output):
    """
    Get the posteriors for all the observations in all_ev_list for the corresponding join_tree.

    Parameters
    ----------
    all_ev_list : list of observations
    join_tree : conditional probability table
    output : str target/output variable

    Returns
    -------
    obs_posteriors : dict of observations posteriors
    predicted_posteriors : list of predicted posteriors
    """
    obs_posteriors = {}
    predicted_posteriors = []
    target_predictions = {}

    if isinstance(output, list):
        output = output[0]  # Convert the list to a single string if it is a list

    for observation in all_ev_list:
        # join_tree.unobserve_all()
        # Do a duplicate of join_tree to avoid modifying the original one
        for ev in observation:
            # Modify the join_tree using this case evidences
            ev4jointree = evidence(ev['nod'], ev['bin_index'], ev['val'], join_tree)
            join_tree.set_observation(ev4jointree)
        # Get the posteriors for this observation
        aux_obs, aux_prd = get_posteriors(join_tree, output)
        # print("aux_obs:", aux_obs)
        # print("aux_prd:", aux_prd)

        # Store the posteriors for this case
        for node_id, posterior in aux_obs.items():
            if node_id not in obs_posteriors:
                obs_posteriors[node_id] = []
            obs_posteriors[node_id].append(posterior)
        predicted_posteriors.append(aux_prd)
        # print("predicted_posteriors:", predicted_posteriors)

        # Extract highest probability and corresponding bin index and store them
        target_probabilities = aux_prd  # aux_prd is the capcost prediction probabilities
        highest_probability = max(target_probabilities)
        corresponding_bin_index = target_probabilities.index(highest_probability)
        target_predictions[str(observation)] = (highest_probability, corresponding_bin_index)

    # Ensure that the join tree is unmodified
    join_tree.unobserve_all()

    return obs_posteriors, predicted_posteriors, target_predictions

def get_all_posteriors_new(all_ev_list, join_tree, outputs):
    """
    Get the posteriors for all the observations in all_ev_list for the corresponding join_tree.

    Parameters
    ----------
    all_ev_list : list of observations
    join_tree : conditional probability table
    outputs : list of str target/output variables

    Returns
    -------
    all_obs_posteriors : dict of observations posteriors for each output
    all_predicted_posteriors : list of predicted posteriors for each output
    all_target_predictions : dict of target predictions for each output
    """
    all_obs_posteriors = {}
    all_predicted_posteriors = {}
    all_target_predictions = {}

    for output in outputs:
        obs_posteriors, predicted_posteriors, target_predictions = get_posteriors_for_single_output(all_ev_list, join_tree, output)
        all_obs_posteriors[output] = obs_posteriors
        all_predicted_posteriors[output] = predicted_posteriors
        all_target_predictions[output] = target_predictions

    return all_obs_posteriors, all_predicted_posteriors, all_target_predictions


def get_posteriors_for_single_output(all_ev_list, join_tree, output):
    """
    Helper function to get the posteriors for a single output variable.
    """
    obs_posteriors = {}
    predicted_posteriors = []
    target_predictions = {}

    for observation in all_ev_list:
        # join_tree.unobserve_all()
        # Do a duplicate of join_tree to avoid modifying the original one
        for ev in observation:
            # Modify the join_tree using this case evidences
            ev4jointree = evidence(ev['nod'], ev['bin_index'], ev['val'], join_tree)
            join_tree.set_observation(ev4jointree)
        # Get the posteriors for this observation
        aux_obs, aux_prd = get_posteriors(join_tree, output)
        # print("aux_obs:", aux_obs)
        # print("aux_prd:", aux_prd)

        # Store the posteriors for this case
        for node_id, posterior in aux_obs.items():
            if node_id not in obs_posteriors:
                obs_posteriors[node_id] = []
            obs_posteriors[node_id].append(posterior)
        predicted_posteriors.append(aux_prd)

        # Extract highest probability and corresponding bin index and store them
        target_probabilities = aux_prd  # aux_prd is the capcost prediction probabilities
        # print("target_probabilities:", target_probabilities )
        highest_probability = max(target_probabilities)
        corresponding_bin_index = target_probabilities.index(highest_probability)
        target_predictions[str(observation)] = (highest_probability, corresponding_bin_index)

    # Ensure that the join tree is unmodified
    join_tree.unobserve_all()

    return obs_posteriors, predicted_posteriors, target_predictions


def find_best_evidence_for_bin(target_predictions, config_type):
    """
    Find the evidence configuration with the highest probability for the specified configuration type.

    Parameters
    ----------
    target_predictions : dict
        Dictionary where the keys are strings representing lists of dictionaries (evidence configurations) 
        and the values are tuples containing the probability and the corresponding bin index.
    config_type : str
        The configuration type to find the best evidence configuration for. Can be 'min' or 'max'.

    Returns
    -------
    best_evidence : str
        The best evidence configuration for the specified configuration type.
    """
    # Initialize max_probability and best_evidence
    max_probability = float('-inf') if config_type == 'max' else float('inf')
    best_evidence = None

    # Iterate over the items of target_predictions
    for evidence, (probability, bin_idx) in target_predictions.items():
        # If config_type is 'min' and probability is less than max_probability
        # or if config_type is 'max' and probability is greater than max_probability
        if (config_type == 'min' and probability < max_probability) or (config_type == 'max' and probability > max_probability):
            max_probability = probability
            best_evidence = evidence

    return best_evidence

def find_best_evidence_for_bin2(target_predictions, config_type):
    """
    Find the evidence configuration with the highest probability for the specified configuration type.

    Parameters
    ----------
    target_predictions : dict
        Dictionary where the keys are strings representing lists of dictionaries (evidence configurations) 
        and the values are tuples containing the probability and the corresponding bin index.
    config_type : str
        The configuration type to find the best evidence configuration for. Can be 'min' or 'max'.

    Returns
    -------
    best_evidence : str
        The best evidence configuration for the specified configuration type.
    """
    # Initialize max_probability and best_evidence
    max_probability = float('-inf')
    best_evidence = None

    # Initialize max_bin_idx and min_bin_idx
    max_bin_idx = float('-inf')
    min_bin_idx = float('inf')

    # Iterate over the items of target_predictions
    for evidence, (probability, bin_idx) in target_predictions.items():
        # If probability is greater than max_probability
        if probability > max_probability:
            max_probability = probability
            best_evidence = evidence
            max_bin_idx = bin_idx
            min_bin_idx = bin_idx
        # If probability is equal to max_probability and config_type is 'min' and bin_idx is less than min_bin_idx
        # or if probability is equal to max_probability and config_type is 'max' and bin_idx is greater than max_bin_idx
        elif probability == max_probability and ((config_type == 'min' and bin_idx < min_bin_idx) or (config_type == 'max' and bin_idx > max_bin_idx)):
            best_evidence = evidence
            max_bin_idx = max(max_bin_idx, bin_idx)
            min_bin_idx = min(min_bin_idx, bin_idx)

    return best_evidence

def generate_all_evidence_combinations(inputs):
    input_vars = inputs['inputs']
    nbins = inputs['nbins'][0]['inputs']  # Assuming the number of bins for inputs is the same for all
    print("chosen input variables:", input_vars)    

    all_evidence_combinations = []
    for input_var in input_vars:
        input_var_evidence = []
        for bin_index in range(1, nbins + 1):  # Start from 1 instead of 0
            for val in [0.0, 1.0]:  # Assuming val can only be 0.0 or 1.0
                input_var_evidence.append({'nod': input_var, 'bin_index': str(bin_index), 'val': val})
        all_evidence_combinations.append(input_var_evidence)
    # print("length of all evidence combinations:", len(all_evidence_combinations), all_evidence_combinations)
    # Convert each tuple to a list
    return [list(combination) for combination in itertools.product(*all_evidence_combinations)]

def generate_all_evidence_combinations2(inputs):
    inference_type = inputs['inference_type']

    if inference_type == 'forward':
        print('Inference type chosen: ' + inference_type)
        nbins = inputs['nbins'][0]['inputs']
        variables = inputs['inputs']
    elif inference_type == 'reverse':
        print('Inference type chosen: ' + inference_type)
        nbins = inputs['nbins'][0]['output']
        variables = inputs['output']
        if isinstance(variables, str):  # If variables is a string, convert it to a list
            variables = [variables]
        print('chosen variables: ' + ', '.join(map(str, variables)))  # Convert each variable to a string before joining
    else:
        raise ValueError(f"Invalid inference_type: {inference_type}")

    all_evidence_combinations = []
    for var in variables:
        var_evidence = []
        for bin_index in range(1, nbins + 1):  # Start from 1 instead of 0
            var_evidence.append({'nod': var, 'bin_index': str(bin_index), 'val': 1.0})
        all_evidence_combinations.append(var_evidence)

    # Generate all combinations of evidence, ensuring each variable appears only once in each combination
    return [list(combination) for combination in itertools.product(*all_evidence_combinations) if len(set(evidence['nod'] for evidence in combination)) == len(variables)]


# def get_posteriors(all_ev_list, join_tree):
#     obs_posteriors_dict = {}
#     predicted_posteriors_list = []

#     for ev_list in all_ev_list:
#         unique_evidence = [dict(ev_dict) for ev_dict in ev_list]  # Convert tuples to dictionaries
#         for ev_dict in unique_evidence:
#             ev = evidence(ev_dict['nod'], int(ev_dict['bin_index']), ev_dict['val'])
#             join_tree.set_observation(ev)
#             obs_posteriors, predictedTargetPosteriors = get_obs_and_pred_posteriors(join_tree, "acceleration")

#         for node_id, posterior in obs_posteriors.items():
#             if node_id not in obs_posteriors_dict:
#                 obs_posteriors_dict[node_id] = []
#             obs_posteriors_dict[node_id].append(posterior)

#         predicted_posteriors_list.append(predictedTargetPosteriors)

#     print("Observation posteriors:", obs_posteriors_dict)
#     print("Predicted target posteriors:", predicted_posteriors_list)

#     return obs_posteriors_dict, predicted_posteriors_list

# obs_posteriors_dict, predicted_posteriors_list = get_posteriors(all_ev_list, join_tree)


# dict_names = ['force_bins', 'mass_bins']
# all_ev_list = set_multiple_observations(df_test_binned, obs_dicts, dict_names, default_bin=5)

# def extract_data_from_dict_list(dict_list, target_dict):
#     """

#     """
#     bin_indices = []
#     actual_values = []
#     for d in dict_list:
#         for k, v in d.items():
#             if k == target_dict:
#                 bin_indices.append(int(v['bin_index']))
#                 if 'actual_value' in v:
#                     actual_values.append(v['actual_value'])
#                 else:
#                     actual_values.append(None)
#     print('bin_indices:', bin_indices)
#     print('actual_values:', actual_values)
#     return bin_indices, actual_values

# target_dict = 'acceleration_bins'
# bin_indices, actual_values = extract_data_from_dict_list(obs_dicts, target_dict)
