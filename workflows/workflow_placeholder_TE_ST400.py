from bnmodel_copy.bayesian_network import BayesianNetwork

# 1. Load the inputs
# data
# discretisation method (equidistant, percentile...)
# train/test percentage
#%%
inputs = {'data': 'st400-v4_snippet.csv', 
          'inputs': ['major_radius', 'aspect_ratio', 'effective_ion_charge', 'toroidal_field_on_plasma'],
          'output': ['CAPITAL_COST_2021_PRICES_[million_USD]','Q_engineering[#]','high_grade_heat[MWt]'],
          'nbins': [{'inputs':5, 'output':3}],
          'nbins_sensitivity_range': [3, 11],
          'kfoldnbins': 25,
          'nfolds': 10,    
          'method': 'meta',
          'error_type': 'D1',
          'discretisation': [{'inputs':'uniform', 'output':'percentile'}],
          'evidence':[{'nod':'CAPITAL_COST_2021_PRICES_million_USD', 'bin_index': '2', 'val': 1}],
          'evidence2':[{'CAPITAL_COST_2021_PRICES_million_USD': [0.0, 0.0, 1.0]}],
          'train_test_split': 0.1,
          'save_join_tree_path': 'join_tree_st400-v4_inputspercentile.json',
          'save_bin_edges_path': 'bin_ranges_st400-v4_inputspercentile.json',
          'load_join_tree_path': 'join_tree_st400-v4_inputspercentile.json',
          'load_bin_edges_path': 'bin_ranges_st400-v4_inputspercentile.json',
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
