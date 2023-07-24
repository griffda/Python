import bnmodel as bn


# 1. Load the inputs
# data
# discretisation method (equidistant, percentile...)
# train/test percentage

inputs = {'data': 'output3.csv', 
          'inputs': ['force', 'mass'],
          'output': 'acceleration',
          'nbins': 5,
          'method': 'uniform',
          'disc_prior': 'equidistant', 
          'disc_target': 'percentile',
          'train_test_split': 0.2}

model = bn.bayesian_network()


# 2. Train the model 
model.train()

# 3. Cross-validate the model
model.validate()

# 4. Save the model

# 5. Run your own case


