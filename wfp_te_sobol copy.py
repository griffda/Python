
from bnmodel_copy.bayesian_network import BayesianNetwork
# config(57)_3outputs_v2_
# 1. Load the inputs
# data
# discretisation method (equidistant, percentile...)
# train/test percentage
#%%
inputs = {'data': 'data/outputv8.csv',
          'inputs': ['force','mass'],
          'output': ['acceleration'],
          'inputs_plotting': {'force': '$F$ [N]', 'mass': '$m$ [kg]'},
          'output_plotting': {'acceleration':'$a$ [m s$^{2}$]'}, 
          'nbins': [{'inputs':8, 'output':8}],
          'nbins_sensitivity_range': [3, 11],
          'kfoldnbins': 25,
          'nfolds': 10,    
          'method': 'kfold',
          'error_type': 'D2',
          'discretisation': [{'inputs':'uniform', 'output':'percentile'}],
          'evidence_type': 'hard', # 'hard' or 'soft or 'optimal_config'
          #'analysis_type': [{'optimal_config': {'net_electrical_output': 'max', 'capcost': 'min', 'high_grade_wasteheat': 'max'}}],
          'analysis_type': [{'reverse_optimal'}],
          'inference_type': 'reverse',
          'evidence':[{'nod': 'force', 'bin_index': '7', 'val': 1.0}, {'nod': 'mass', 'bin_index': '1', 'val': 1.0}],
          'train_test_split': 0.1,
          'save_join_tree_path': 'join_trees/join_tree_fma_10k_uniform.json',
          'save_bin_edges_path': 'join_trees/bin_ranges_fma_10k_uniform.json',
          'load_join_tree_path': 'join_trees/join_tree_fma_10k_uniform.json',
          'load_bin_edges_path': 'join_trees/bin_ranges_fma_10k_uniform.json',
          'save_evidence_path': 'evidence/evidence_fma_10k_uniform.pkl',
          'load_evidence_path': 'evidence/evidence_fma_10k_uniform.pkl',
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
model.run_model2()
#%%
# 5. Perform sensitivity analysis
results = model.sensitivity_analysis()

#%%
# 6. Run your own case
model.run_model2()