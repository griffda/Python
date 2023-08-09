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
          'nfolds': 2,
          'method': 'kfold',
          'disc_prior': 'equidistant', 
          'disc_target': 'percentile'}


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

# 5. Run your own case



# %%
