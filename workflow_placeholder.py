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
          'evidence': [{'nod':'force', 'bin_index': 2, 'val': 1.0},
                       {'nod':'mass', 'bin_index': 2, 'val': 1.0}],
          'join_tree_path': 'join_tree_json.txt',
          'bin_edges_path': 'bin_edges_json.txt',
          }


#%%
model = BayesianNetwork(inputs)

#%%
# 2. Train the model 
model.train()

# join_tree = bn.join_tree_population.prob_dists(model.struct, model.train_binned)

#%%
# 3. Cross-validate the model
model.validate()

# 4. Save the model
#%%
model.save('model_json.txt')

# 5. Run your own case

model.run_model()



# %%
