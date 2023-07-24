import bnmodel_copy as bn

#%% INPUTS
csv_path = 'st16_trimmed.csv'
inputs = ['fdene', 'fimp(14)', 'pseprmax', 'feffcd', 'aspect', 'boundu(2)', 'outlet_temp', 'beta', 'etanbi']
output = 'capcost'

nbins = 5
histnbins = 30
numFolds = 2

#%% Prepare data
data_obj = bn.BNdata(csv_path, inputs, output)
data = data_obj.data

# Create the structure
structure = data_obj.df2struct()

#%% Create the CrossValidator instance
cross_validator = bn.CrossValidator()

#%% Run the model and get the results
(obs_posteriors_dict, 
 bin_edges, 
 prior_xytrn, 
 norm_distance_errors, 
 prediction_accuracy, 
 av_prediction_accuracy) = cross_validator.k_fold_cross_validation(structure, data, output, numFolds, histnbins, nbins=nbins)

#%% Create the Plotter instance
plotter = bn.Plotter()

# Plot the errors
ax_errors = plotter.plot_errors(norm_distance_errors, histnbins, prediction_accuracy, av_prediction_accuracy, int(numFolds/2))

# Plot the overview
ax_overview = plotter.plot_results(obs_posteriors_dict, bin_edges, prior_xytrn, inputs, output, 3, 5)
