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
          'nbins_sensitivity_range': [3, 11],
          'kfoldnbins': 25,
          'nfolds': 10,    
          'method': 'meta',
          'error_type': 'D1',
          'discretisation': [{'inputs':'uniform', 'output':'percentile'}],
          'evidence_type': 'hard',
          'evidence':[{'nod':'capcost', 'bin_index': '5', 'val': 1}],
          'evidence_soft':{# Change this to a dictionary
                'capcost': [0.0, 0.0, 0.5, 0.5, 0.0, 0.0]},
          'train_test_split': 0.1,
          'save_join_tree_path': 'join_trees/join_tree_te_sobol_3outputs_v2.json',
          'save_bin_edges_path': 'join_trees/bin_ranges_te_sobol_3outputs_v2.json',
          'load_join_tree_path': 'join_trees/join_tree_te_sobol_3outputs_v2.json',
          'load_bin_edges_path': 'join_trees/bin_ranges_te_sobol_3outputs_v2.json',
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
model.run_model2() #this still needs work for soft evidence. 