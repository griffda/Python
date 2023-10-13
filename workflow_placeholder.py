from bnmodel_copy.bayesian_network import BayesianNetwork

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
          'train_test_split': 0.2,
          'evidence': [{'nod':'force', 'bin_index': '3', 'val': 1.0},
                       {'nod':'mass', 'bin_index': '5', 'val': 1.0}],
          'join_tree_path': 'simple-join-tree.json',
          'bin_edges_path': 'bin_ranges.json',
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
# 4. Save the model
model.save('')

#%%
# 5. Run your own case
model.run_model()


# %%
