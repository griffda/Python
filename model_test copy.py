import bnmodel as bn

#%% INPUTS
csv_path = 'outputv4.csv'
test_bin_size = 0.4
inputs = ['force', 'mass']
output = 'acceleration'

structure = {
    'mass':[],
    'force': [],
    'acceleration': ['force', 'mass']
}

n_obs = 100
nbins = 15

#%% Run the model
train_binned, test_binned, bin_edges, prior_xytrn = bn.discretisation.binning_data(csv_path, test_bin_size,
                                                                                    x_cols = inputs, y_cols = [output])

join_tree = bn.join_tree_population.prob_dists(structure, train_binned)

obs_dicts = bn.generate_posteriors.generate_multiple_obs_dicts(test_binned, n_obs, output, csv_path)

all_ev_list = bn.generate_posteriors.gen_ev_list(test_binned, obs_dicts, output)

obs_posteriors_dict, predicted_posteriors_list = bn.generate_posteriors.get_all_posteriors(all_ev_list, join_tree, output)


#%% Error evaluation
correct_bin_locations, actual_values = bn.evaluate_errors.get_correct_values(obs_dicts, output)
bin_ranges = bn.evaluate_errors.extract_bin_ranges(output, bin_edges)
distance_errors, norm_distance_errors, output_bin_means = bn.evaluate_errors.distribution_distance_error(correct_bin_locations,
                                                                                                        predicted_posteriors_list,
                                                                                                        actual_values, bin_ranges,
                                                                                                        plot=True, nbins = nbins)


#%% Plotting
# ax = bn.plot_overview.plot_results(obs_posteriors_dict, bin_edges, prior_xytrn, inputs, output)
