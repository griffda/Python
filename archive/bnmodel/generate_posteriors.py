from bnmodel.join_tree_population import evidence
import pandas as pd


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
    test_binned = test_binned.drop([output], axis=1) 
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
    for node, posteriors_raw in join_tree.get_posteriors().items():
        obs_posteriors[node] = [posteriors_raw[val] for val in posteriors_raw]
        if node == output:  # check if the observation corresponds to the specified target variable
            predictedTargetPosteriors = [posteriors_raw[val] for val in posteriors_raw]

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

    for observation in all_ev_list:
        join_tree.unobserve_all()
        # Do a duplicate of join_tree to avoid modifying the original one
        for ev in observation:
            # Modify the join_tree using this case evidences
            ev4jointree = evidence(ev['nod'], ev['bin_index'], ev['val'], join_tree)
            join_tree.set_observation(ev4jointree)
        # Get the posteriors for this observation
        aux_obs, aux_prd = get_posteriors(join_tree, output)

        # Store the posteriors for this case
        for node_id, posterior in aux_obs.items():
            if node_id not in obs_posteriors:
                obs_posteriors[node_id] = []
            obs_posteriors[node_id].append(posterior)
        predicted_posteriors.append(aux_prd)
    
    # Ensure that the join tree is unmodified
    join_tree.unobserve_all()

    return obs_posteriors, predicted_posteriors

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
