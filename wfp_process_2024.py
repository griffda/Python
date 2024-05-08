
from bnmodel_copy.bayesian_network import BayesianNetwork
# config(57)_3outputs_v2_
# 1. Load the inputs
# data
# discretisation method (equidistant, percentile...)
# train/test percentage
#%%
inputs = {'data': 'data/st20_trimmed.csv', 
          'inputs': ['fdene', 'fimp14', 'pseprmax', 'feffcd', 'aspect', 'boundu2', 'outlet_temp', 'beta', 'etanbi'],
          'output': ['capcost'],
          'inputs_plotting': {'fdene': '$f_{GW}^{max}$', 'fimp14': '$F_{W}$', 'aspect': '$A$', 'pseprmax' : '$P_{sep}$/$R$ [MW/m]', 'feffcd' : '$f_{CD}$', 'boundu2' : '$B_{T}^{max}$ [T]', 'outlet_temp': '$\Delta T_{in}$ [ºC]', 'beta': '$\\beta$', 'etanbi': '$\eta_{NBI}$'},
          'output_plotting': {'capcost': '$C$ [million USD]'}, 
          'nbins': [{'inputs':8, 'output':5}],
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
          'evidence':[{'nod': 'fdene', 'bin_index': '5', 'val': 1.0}, {'nod': 'fimp14', 'bin_index': '5', 'val': 1.0}, {'nod': 'pseprmax', 'bin_index': '5', 'val': 1.0}, {'nod': 'feffcd', 'bin_index': '5', 'val': 1.0}, {'nod': 'capcost', 'bin_index': '2', 'val': 1.0}],
          'train_test_split': 0.1,
          'save_join_tree_path': 'join_trees/join_tree_st20.json',
          'save_bin_edges_path': 'bin_ranges/bin_ranges_st20.json',
          'load_join_tree_path': 'join_trees/join_tree_st20.json',
          'load_bin_edges_path': 'bin_ranges/bin_ranges_st20.json',
          'save_evidence_path': 'evidence/evidence_st20.pkl',
          'load_evidence_path': 'evidence/evidence_st20.pkl',
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
# 5. Perform sensitivity analysis
results = model.sensitivity_analysis()

#%%
# 6. Run your own case
model.run_model2()