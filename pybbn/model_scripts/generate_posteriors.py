from join_tree_population import join_tree, evidence
from discretisation import  df_test_binned

def generate_obs_dict(test_df, target):
    # choose a random row from the test_df
    row = test_df.sample()
    print("Selected row index:", row.index[0])

    # generate an obs_dict from the chosen row
    obs_dict = {}
    for col in test_df.columns:
        if col == target:
            obs_dict[col] = {'bin_index': str(row[col].values[0]), 'actual_value': round(row['acceleration'].values[0],2)}
        elif col.endswith('_bins'):
            obs_dict[col] = {'bin_index': str(row[col].values[0]), 'val': 1.0}

    # print("Observation dictionary:", obs_dict)
    return obs_dict

def generate_multiple_obs_dicts(test_df, num_samples, target):
    obs_dicts = []
    for i in range(num_samples):
        obs_dict = generate_obs_dict(test_df, target)
        obs_dicts.append(obs_dict)
    print("Observation dictionaries:", obs_dicts)
    return obs_dicts

obs_dicts = generate_multiple_obs_dicts(df_test_binned, 3, 'acceleration_bins')

def set_multiple_observations(df, obs_dicts, dict_names, default_bin=5):
    df = df.drop(['acceleration', 'acceleration_bins'], axis=1) ###this will need to change can instead specify which ones to drop in args
    all_ev_list = []
    for obs_dict in obs_dicts:
        if obs_dict.keys() & dict_names:  # if obs_dict contains any of the dict_names
            ev_list = []
            for col in df.columns:
                if col in obs_dict:
                    bin_index = obs_dict[col]['bin_index']
                    val = obs_dict[col]['val']
                else:
                    bin_index = str(default_bin)
                    val = 1.0
                ev_dict = {'nod':col, 'bin_index':bin_index, 'val': val}
                ev_list.append(ev_dict)
            all_ev_list.append(ev_list)
    print("All evidence lists:", all_ev_list)
    return all_ev_list

dict_names = ['force_bins', 'mass_bins']
all_ev_list = set_multiple_observations(df_test_binned, obs_dicts, dict_names, default_bin=5)

def extract_data_from_dict_list(dict_list, target_dict):
    bin_indices = []
    actual_values = []
    for d in dict_list:
        for k, v in d.items():
            if k == target_dict:
                bin_indices.append(int(v['bin_index']))
                if 'actual_value' in v:
                    actual_values.append(v['actual_value'])
                else:
                    actual_values.append(None)
    print('bin_indices:', bin_indices)
    print('actual_values:', actual_values)  
    return bin_indices, actual_values

target_dict = 'acceleration_bins'
bin_indices, actual_values = extract_data_from_dict_list(obs_dicts, target_dict)

def get_obs_and_pred_posteriors(join_tree, target_variable):
    obs_posteriors = {}

    for node, posteriors in join_tree.get_posteriors().items():
        obs = node[:-5]  # remove the "_bins" suffix from the node name to get the observation name
        obs_posteriors[obs] = [round(posteriors[val],2) for val in sorted(posteriors)]  # sort the posteriors by value and add them to the dictionary

    # print("Observation posteriors:", obs_posteriors)

    predictedTargetPosteriors = []

    for node, posteriors in join_tree.get_posteriors().items():
        obs = node[:-5]  # remove the "_bins" suffix from the node name to get the observation name
        if obs == target_variable:  # check if the observation corresponds to the specified target variable
            predictedTargetPosteriors = [round(posteriors[val],2) for val in sorted(posteriors)]  # sort the posteriors by value and add them to the list

    # print("Predicted target posteriors:", predictedTargetPosteriors)

    return obs_posteriors, predictedTargetPosteriors


###The issue arises from having identical evidence lists. 
# As a result, the same evidence is applied twice, leading to the same posteriors being computed for both instances. 
# To obtain different posteriors, you need to provide distinct evidence for each case. 
# For example, you can add a unique suffix to the evidence node names in each evidence list,
# and need to tell 

def get_posteriors(all_ev_list, join_tree):
    obs_posteriors_dict = {}
    predicted_posteriors_list = []

    for ev_list in all_ev_list:
        for ev_dict in ev_list:
            ev = evidence(ev_dict['nod'], ev_dict['bin_index'], ev_dict['val'])
            join_tree.set_observation(ev)
            obs_posteriors, predictedTargetPosteriors = get_obs_and_pred_posteriors(join_tree, "acceleration")

        for node_id, posterior in obs_posteriors.items():
            if node_id not in obs_posteriors_dict:
                obs_posteriors_dict[node_id] = []
            obs_posteriors_dict[node_id].append(posterior)

        predicted_posteriors_list.append(predictedTargetPosteriors)

    print("Observation posteriors:", obs_posteriors_dict)
    print("Predicted target posteriors:", predicted_posteriors_list)

    return obs_posteriors_dict, predicted_posteriors_list

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

obs_posteriors_dict, predicted_posteriors_list = get_posteriors(all_ev_list, join_tree)



