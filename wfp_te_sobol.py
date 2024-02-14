from bnmodel_copy.bayesian_network import BayesianNetwork

# 1. Load the inputs
# data
# discretisation method (equidistant, percentile...)
# train/test percentage
#%%
inputs = {'data': 'data/te_data_v3.csv', 
          'inputs': ['major_radius','aspect_ratio','effective_ion_charge','toroidal_field_on_plasma'],
          'output': ['capcost'],
          'nbins': [{'inputs':7, 'output':7}],
          'nbins_sensitivity_range': [3, 13],
          'kfoldnbins': 25,
          'nfolds': 2,    
          'method': 'meta',
          'error_type': 'D1',
          'discretisation': [{'inputs':'uniform', 'output':'percentile'}],
          'evidence_type': 'optimal_config',
          'inference_type': 'forward',
          'optimal_config_target(s)':'major_radius',
          'optimal_config_type': 'max_val',
          'evidence':[{'nod': 'major_radius', 'bin_index': '5', 'val': 1.0}, {'nod': 'aspect_ratio', 'bin_index': '7', 'val': 1.0}, {'nod': 'effective_ion_charge', 'bin_index': '4', 'val': 1.0}, {'nod': 'toroidal_field_on_plasma', 'bin_index': '1', 'val': 1.0}],
          'evidence_soft':{# Change this to a dictionary
                'capcost': [0.0, 0.0, 0.5, 0.5, 0.0, 0.0]},
          'train_test_split': 0.1,
          'save_join_tree_path': 'join_trees/join_tree_te_sobol_validation.json',
          'save_bin_edges_path': 'join_trees/bin_ranges_te_sobol_validation.json',
          'load_join_tree_path': 'join_trees/join_tree_te_sobol_validation.json',
          'load_bin_edges_path': 'join_trees/bin_ranges_te_sobol_validation.json',
          }
#%%
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
# 5. Perform sensitivity analysis
results = model.sensitivity_analysis()

#%%
# 6. Run your own case
model.run_model2() #this still needs work for soft evidence. 