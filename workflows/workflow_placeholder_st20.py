from bnmodel_copy.bayesian_network import BayesianNetwork

# 1. Load the inputs
# data
# discretisation method (equidistant, percentile...)
# train/test percentage
#%%
inputs = {'data': 'data/st20_trimmed_v2.csv', 
          'inputs': ['aspect', 'outlet_temp', 'beta', 'etanbi'],
          'output': ['capcost'],
          'nbins': [{'inputs':5, 'output':5}],
          'nbins_sensitivity_range': [3, 11],
          'kfoldnbins': 25,
          'nfolds': 10,    
          'method': 'kfold',
          'error_type': 'D1',
          'discretisation': [{'inputs':'uniform', 'output':'percentile'}],
          'evidence':[{'nod':'fimp14', 'bin_index': '5', 'val': 1},
                      {'nod':'pseprmax', 'bin_index': '3', 'val': 1},
                      {'nod':'fdene', 'bin_index': '3', 'val': 1},
                      {'nod':'etanbi', 'bin_index': '4', 'val': 1},
                      {'nod':'feffcd', 'bin_index': '4', 'val': 1},
                      {'nod':'capcost', 'bin_index': '3', 'val': 1}],
          'train_test_split': 0.1,
          'save_join_tree_path': 'join_tree_st20.json',
          'save_bin_edges_path': 'bin_ranges_st20.json',
          'load_join_tree_path': 'join_tree_st20.json',
          'load_bin_edges_path': 'bin_ranges_st20.json',
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
