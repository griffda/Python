import bnmodel as bn

df_train_binned, df_test_binned, bin_edges_dict, prior_dict_xytrn = bn.discretisation.binning_data('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/outputv3.csv', 0.4, x_cols = ['force', 'mass'], y_cols = ['acceleration'])

structure = {
    'mass':[],
    'force': [],
    'acceleration': ['force', 'mass']
}

join_tree = bn.join_tree_population.prob_dists(structure, df_train_binned)

obs_dicts = bn.generate_posteriors.generate_multiple_obs_dicts(df_test_binned, 10, 'acceleration')

all_ev_list = bn.generate_posteriors.set_multiple_observations(df_test_binned, obs_dicts, 'acceleration')

obs_posteriors_dict, predicted_posteriors_list = bn.generate_posteriors.get_all_posteriors(all_ev_list, join_tree)

ax = bn.plot_overview.plot_results(obs_posteriors_dict, bin_edges_dict, prior_dict_xytrn)

