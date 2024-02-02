from bnmodel_copy.bayesian_network import BayesianNetwork

# 1. Load the inputs
# data
# discretisation method (equidistant, percentile...)
# train/test percentage
#%%
inputs = {'data': 'synthetic_data_v3.csv', 
          'inputs': ['major_radius','aspect_ratio','effective_ion_charge','toroidal_field_on_plasma'],
          'output': ['capcost','Q_engineering','high_grade_wasteheat'],
          'nbins': [{'inputs':5, 'output':5}],
          'nbins_sensitivity_range': [3, 11],
          'kfoldnbins': 25,
          'nfolds': 10,    
          'method': 'meta',
          'error_type': 'D1',
          'discretisation': [{'inputs':'uniform', 'output':'percentile'}],
          'evidence':[{'nod':'Q_engineering', 'bin_index': '2', 'val': 1}],
          'evidence2':[{'Q_engineering': [0.0, 0.0, 1.0]}],
          'train_test_split': 0.1,
          'save_join_tree_path': 'join_trees/join_tree_synth_v3.json',
          'save_bin_edges_path': 'join_trees/bin_ranges_synth_v3.json',
          'load_join_tree_path': 'join_trees/join_tree_synth_v3.json',
          'load_bin_edges_path': 'join_trees/bin_ranges_synth_v3.json',
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
# 5. Run your own case
model.run_model()


# %%
