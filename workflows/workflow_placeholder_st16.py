from bnmodel_copy.bayesian_network import BayesianNetwork

# 1. Load the inputs
# data
# discretisation method (equidistant, percentile...)
# train/test percentage
#%%
inputs = {'data': 'st16_trimmed.csv', 
          'inputs': ['fdene', 'fimp14', 'pseprmax', 'feffcd', 'aspect', 'boundu2', 'outlet_temp', 'beta', 'etanbi'],
          'output': ['capcost'],
          'nbins': 7,
          'nfolds': 3,
          'method': 'meta',
          'disc_prior': 'equidistant', 
          'disc_target': 'percentile',
          'train_test_split': 0.2,
          'evidence': [{'nod':'capcost', 'bin_index': '3', 'val': 1.0}],
          'save_join_tree_path': 'join_tree_st16.json',
          'save_bin_edges_path': 'bin_ranges_st16.json',
          'load_join_tree_path': 'join_tree_st16.json',
          'load_bin_edges_path': 'bin_ranges_st16.json',
          }


#%%
# 1. Load the inputs
model = BayesianNetwork(inputs)

#%%
# 2. Train the model 
model.train()

#%%
# 3. Cross-validate the model
model.validate()

#%%
# 4. Save the model
model.save('')

#%%
# 5. Run your own case
model.run_model()


# %%
