from bnmodel_copy.bayesian_network import BayesianNetwork

# 1. Load the inputs
# data
# discretisation method (equidistant, percentile...)
# train/test percentage
#%%
inputs = {'data': 'outputv5.csv', 
          'inputs': ['force', 'mass'],
          'output': ['acceleration'],
          'nbins': [{'inputs':3, 'output':3}],
          'nbins_sensitivity_range': [3, 8],
          'kfoldnbins': 25,
          'nfolds': 10,    
          'method': 'kfold',
          'error_type': 'D1',
          'discretisation': [{'inputs':'uniform', 'output':'percentile'}],
          'evidence': [{'nod':'force', 'bin_index': '2', 'val': 1},
                       {'nod':'mass', 'bin_index': '2', 'val': 1}],
          'train_test_split': 0.1,
          'save_join_tree_path': 'join_tree_normal_fma_5k.json',
          'save_bin_edges_path': 'bin_ranges_normal_fma_5k.json',             
          'load_join_tree_path': 'join_tree_normal_fma_5k.json',
          'load_bin_edges_path': 'bin_ranges_normal_fma_5k.json',
          }


#%%
# 1. Load the inputsobs
model = BayesianNetwork(inputs)

#%%
# 2. Train the model 
model.train()

#%%
# 3. Cross-validate the model
model.validate()

#%%
# 4. Perform sensitivity analysis
# error might be occuring here because 
results = model.sensitivity_analysis()
print("Accuracies for different bin configurations: ", results)

#%%
# 4. Save the model
model.save('')

#%%
# 5. Run your own case
model.run_model()


# %%
