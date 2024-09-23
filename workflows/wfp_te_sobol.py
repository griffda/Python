from bnmodel_copy.bayesian_network import BayesianNetwork
# config(57)_3outputs_v2_
# 1. Load the inputs
# data
# discretisation method (equidistant, percentile...)
# train/test percentage
#%%
inputs = {'data': 'data/te_data_march_v1.csv',
          'inputs': ['major_radius','aspect_ratio','effective_ion_charge','toroidal_field_on_plasma'],
          'output': ['capcost','high_grade_wasteheat','net_electrical_output'],
          'inputs_plotting': {'major_radius': '$R$ [m]', 'aspect_ratio': '$A$', 'effective_ion_charge': '$Z_{eff}$', 'toroidal_field_on_plasma': '$B_{T}$ [T]'},
          'output_plotting': {'high_grade_wasteheat':'$H$ [MWt]', 'capcost': '$C$ [million USD]', 'net_electrical_output': '$E$ [MWe]'}, 
          'nbins': [{'inputs':7, 'output':5}],
          'nbins_sensitivity_range': [3, 13],
          'kfoldnbins': 25,
          'nfolds': 10,    
          'method': 'meta',
          'error_type': 'D1',
          'discretisation': [{'inputs':'uniform', 'output':'percentile'}],
          'evidence_type': 'hard', # 'hard' or 'soft or 'optimal_config'
          #'analysis_type': [{'optimal_config': {'net_electrical_output': 'max', 'capcost': 'min', 'high_grade_wasteheat': 'max'}}],
          'analysis_type': [{'reverse_optimal'}],
          'inference_type': 'reverse',
          'evidence':[{'nod': 'capcost', 'bin_index': '2', 'val': 1.0}, {'nod': 'high_grade_wasteheat', 'bin_index': '5', 'val': 1.0}, {'nod': 'net_electrical_output', 'bin_index': '5', 'val': 1.0}],
          'train_test_split': 0.1,
          'save_join_tree_path': 'join_trees/join_tree_te_032024_data3outputs_v3.json',
          'save_bin_edges_path': 'bin_ranges/bin_ranges_te_032024_data3outputs_v3.json',
          'load_join_tree_path': 'join_trees/join_tree_te_032024_data3outputs_v3.json',
          'load_bin_edges_path': 'bin_ranges/bin_ranges_te_032024_data3outputs_v3.json',
          'save_evidence_path': 'evidence/evidence_te_032024_data3outputs_v3.pkl',
          'load_evidence_path': 'evidence/evidence_te_032024_data3outputs_v3.pkl',
          }
#%%
model = BayesianNetwork(inputs)
model.run_model2()
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
# 5. Perform sensitivity analysis
# results = model.sensitivity_analysis()

#%%
# 6. Run your own case
model.run_model2()

# hybrid  'evidence':[{'nod': 'aspect_ratio', 'bin_index': '3', 'val': 1.0}, {'nod': 'high_grade_wasteheat', 'bin_index': '7', 'val': 1.0}, {'nod': 'net_electrical_output', 'bin_index': '7', 'val': 1.0}, {'nod': 'major_radius', 'bin_index': '1', 'val': 1.0}],