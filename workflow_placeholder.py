from bnmodel_copy.bayesian_network import BayesianNetwork
import bnmodel_copy as bn
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle

# 1. Load the inputs
# data
# discretisation method (equidistant, percentile...)
# train/test percentage
#%%
inputs = {'data': 'outputv3.csv', 
          'inputs': ['force', 'mass'],
          'output': ['acceleration'],
          'nbins': 5,
          'method': 'uniform',
          'disc_prior': 'equidistant', 
          'disc_target': 'percentile',
          'train_test_split': 0.2}


#%%
model = BayesianNetwork(inputs)

#%%
# 2. Train the model 
model.train()

# join_tree = bn.join_tree_population.prob_dists(model.struct, model.train_binned)

#%%
# 3. Cross-validate the model
model.validate()
obs_dicts = bn.generate_posteriors.generate_multiple_obs_dicts(model.test_binned, 
                                                               model.inputs['output'], 
                                                               model.inputs['data'])
all_ev_list = bn.generate_posteriors.gen_ev_list(model.test_binned,
                                                 obs_dicts,
                                                 model.inputs['output'])
obs_posteriors, predicted_posteriors = bn.generate_posteriors.get_all_posteriors(all_ev_list,
                                                                                 model.join_tree,
                                                                                 model.inputs['output'])
correct_bin_locations, actual_values = bn.evaluate_errors.get_correct_values(obs_dicts, inputs['output'])
bin_ranges = bn.evaluate_errors.extract_bin_ranges(inputs['output'], model.bin_edges)
norm_distance_errors, prediction_accuracy = bn.evaluate_errors.distribution_distance_error(correct_bin_locations,
                                                                                           predicted_posteriors,
                                                                                           actual_values,
                                                                                           bin_ranges)


# 4. Save the model

# 5. Run your own case



# %%
