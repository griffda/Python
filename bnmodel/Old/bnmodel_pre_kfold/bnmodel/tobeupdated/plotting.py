from model_funcs import join_tree, get_obs_and_pred_posteriors, n_rows, n_cols, bin_edges_dict, prior_dict_xytrn, plot_posterior_probabilities

obs_posteriors, predictedTargetPosteriors = get_obs_and_pred_posteriors(join_tree, "acceleration") ###obs_posteriors comes first. This is to match the order of the unpacking in the function call.

plot_posterior_probabilities(n_rows, n_cols, bin_edges_dict, prior_dict_xytrn, obs_posteriors, plot=True) 
