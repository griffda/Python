import bnmodel as bn

#%% INPUTS
csv_path = 'st20_trimmmed.csv'
test_bin_size = 0.2
inputs = ['fdene', 'fimp(14)', 'pseprmax', 'feffcd', 'aspect', 'boundu(2)', 'outlet_temp', 'beta', 'etanbi']
output = 'capcost'

structure = {
    'fdene':['capcost'],
    'fimp14':['capcost'],
    'pseprmax':['capcost'],  
    'feffcd':['capcost'],
    'aspect':['capcost'],   
    'boundu2':['capcost'],    
    'outlet_temp':['capcost'],  
    'beta':['capcost'], 
    'etanbi':['capcost'],  
    'capcost': []   
}

n_obs = 10
nbins = 30

#%% Prepare data
data = bn.utilities.prepare_csv(csv_path)


#%% Run the model
train_binned, test_binned, bin_edges, prior_xytrn = bn.discretisation.binning_data(data, test_bin_size, nbins = 7,
                                                                                   y_cols = [output])

join_tree = bn.join_tree_population.prob_dists(structure, train_binned)

obs_dicts = bn.generate_posteriors.generate_multiple_obs_dicts(test_binned, n_obs, output, data)

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
ax = bn.plot_overview.plot_results(obs_posteriors_dict, bin_edges, prior_xytrn, inputs, output, 3, 5)

# %%
