import bnmodel as bn

#%% INPUTS
csv_path = 'outputv3.csv'

inputs = ['force', 'mass']
output = 'acceleration'

nbins = 5
histnbins = 30
numFolds = 2

#%% Prepare data
data = bn.utilities.prepare_csv(csv_path)

structure = bn.utilities.df2struct(data, inputs, output)

#%% Run the model
(obs_posteriors_dict, 
 bin_edges, 
 prior_xytrn, 
 norm_distance_errors, 
 prediction_accuracy, av_prediction_accuracy)  = bn.cross_validate.k_fold_cross_validation(structure, data, output, numFolds, histnbins, nbins=nbins)

  
#%% Plotting
# ax = bn.plot_overview.plot_results(obs_posteriors_dict, bin_edges, prior_xytrn, inputs, output, 3, 5)

ax_errors = bn.plotting.plot_errors(norm_distance_errors, histnbins, prediction_accuracy, av_prediction_accuracy, int(numFolds/2))

ax_overview = bn.plotting.plot_results(obs_posteriors_dict, bin_edges, prior_xytrn, inputs, output, 3, 5)