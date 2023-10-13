fdene = [[0.851, 0.89], [0.89, 0.93], [0.93, 0.97], [0.97, 1.01], [1.01, 1.049]]
fimp14 = [[1.03e-05, 2.81e-05], [2.81e-05, 4.58e-05], [4.58e-05, 6.35e-05], [6.35e-05, 8.12e-05], [8.12e-05, 9.89e-05]]
pseprmax = [[20.073, 25.059], [25.059, 30.02], [30.02, 34.98], [34.98, 39.941], [39.941, 44.902]]
feffcd = [[0.801, 0.881], [0.881, 0.96], [0.96, 1.04], [1.04, 1.119], [1.119, 1.198]]
aspect = [[1.801, 1.84], [1.84, 1.88], [1.88, 1.92], [1.92, 1.96], [1.96, 1.999]]
boundu2 = [[3.50, 3.70], [3.70, 3.90], [3.90, 4.10], [4.10, 4.30], [4.30, 4.50]]
outlet_temp = [[793.87, 853.70], [853.70, 913.23], [913.23, 972.76], [972.76, 1032.30], [1032.30, 1091.83]],
beta = [[0.04, 0.07], [0.07, 0.10], [0.10, 0.14], [0.14, 0.17], [0.17, 0.20]],
etanbi = [[0.20, 0.28], [0.28, 0.36], [0.36, 0.44], [0.44, 0.52], [0.52, 0.60]],
capcost = [[4595.9, 4906.56], [4906.56, 5161.6], [5161.6, 5535.88], [5535.88, 7172.38], [7172.38, 24411.0]] 

For the 'optimistic' scenario:
ev_dict = {{'fdene', 'bin_index': '3','val': 1.0, 'actual_value': 1.00},
{'nod': 'fimp14','bin_index': '0','val': 1.0, 'actual_value': 1.00e-04},
{'nod': 'pseprmax','bin_index': '4','val': 1.0, 'actual_value': 42.0},
{'nod': 'feffcd','bin_index': '0','val': 1.0, 'actual_value': 0.80},
{'nod': 'aspect','bin_index': '4','val': 1.0, 'actual_value': 2.0},
{'nod': 'boundu2','bin_index': '4','val': 1.0, 'actual_value': 4.5},
{'nod': 'outlet_temp','bin_index': '0','val': 1.0, 'actual_value': 800.0},
{'nod': 'beta','bin_index': '3','val': 1.0, 'actual_value': 0.18},
{'nod': 'etanbi','bin_index': '3','val': 1.0, 'actual_value': 0.50},
{'nod': 'capcost','bin_index': '0','val': 1.0, 'actual_value': 4551.47}}


For the 'moderate' scenario:
ev_dict = {'fdene','bin_index': '1', 'val': 1.0,'actual_value': 0.90},
{'nod','fimp14','bin_index': '2','val': 1.0, 'actual_value': 5.00e-05},
{'nod','pseprmax','bin_index': '2','val': 1.0, 'actual_value': 33.0},
{'nod','feffcd','bin_index': '2','val': 1.0, 'actual_value': 1.0},
{'nod','aspect','bin_index': '4','val': 1.0, 'actual_value': 2.0},
{'nod','boundu)','bin_index': '2','val': 1.0, 'actual_value': 4.0},
{'nod','outlet_temp','bin_index': '0','val': 1.0, 'actual_vauel': 650.0},
{'nod','beta','bin_index': '2','val': 1.0, 'actual_value': 0.11},
{'nod','etanbi','bin_index': '2','val': 1.0, 'actual_value': 0.40},
{'nod','capcost','bin_index': '1','val': 1.0, 'actual_value': 5124.62}}

all_ev_list = [[{'nod': 'fdene', 'bin_index': '4', 'val': 1.0},
           {'nod': 'fimp14', 'bin_index': '1', 'val': 1.0},
           {'nod': 'pseprmax', 'bin_index': '5', 'val': 1.0},
           {'nod': 'feffcd', 'bin_index': '1', 'val': 1.0},
           {'nod': 'aspect', 'bin_index': '5', 'val': 1.0},
           {'nod': 'boundu2', 'bin_index': '5', 'val': 1.0},
           {'nod': 'outlet_temp', 'bin_index': '1', 'val': 1.0},
           {'nod': 'beta', 'bin_index': '4', 'val': 1.0},
           {'nod': 'etanbi', 'bin_index': '4', 'val': 1.0}], [{'nod': 'fdene', 'bin_index': '1', 'val': 1.0},
           {'nod': 'fimp14', 'bin_index': '2', 'val': 1.0},
           {'nod': 'pseprmax', 'bin_index': '3', 'val': 1.0},
           {'nod': 'feffcd', 'bin_index': '3', 'val': 1.0},
           {'nod': 'aspect', 'bin_index': '5', 'val': 1.0},
           {'nod': 'boundu)', 'bin_index': '3', 'val': 1.0},
           {'nod': 'outlet_temp', 'bin_index': '1', 'val': 1.0},
           {'nod': 'beta', 'bin_index': '3', 'val': 1.0},
           {'nod': 'etanbi', 'bin_index': '3', 'val': 1.0}], [{'nod': 'fdene', 'bin_index': '0', 'val': 1.0},
           {'nod': 'fimp14', 'bin_index': '2', 'val': 1.0},
           {'nod': 'pseprmax', 'bin_index': '1', 'val': 1.0},
           {'nod': 'feffcd', 'bin_index': '1', 'val': 1.0},
           {'nod': 'aspect', 'bin_index': '5', 'val': 1.0},
           {'nod': 'boundu2', 'bin_index': '1', 'val': 1.0},
           {'nod': 'outlet_temp', 'bin_index': '1', 'val': 1.0},
           {'nod': 'beta', 'bin_index': '2', 'val': 1.0},
           {'nod': 'etanbi', 'bin_index': '2', 'val': 1.0}]]



csv_path = "path/to/your/csv_file.csv"
inputs = ['input1', 'input2', ...]  # List of input column names
output = 'output_column'  # The output column name

# Create an instance of the BNdata class
data_obj = BNdata(csv_path, inputs, output)

# Access the DataFrame using the data attribute
data_df = data_obj.data

# Use the df2struct method to get the structure dictionary
structure_dict = data_obj.df2struct()

# Create an instance of the JoinTreeBuilder class
join_tree_builder = JoinTreeBuilder(structure, data)

# Call the prob_dists method to get the join tree
join_tree = join_tree_builder.prob_dists()

# Use the evidence method to prepare evidence for the join tree
ev = join_tree_builder.evidence('nod', 'bin_index', val, join_tree)

# Assuming you have defined the necessary variables and imported the required modules

# Create an instance of the ObservationGenerator class
obs_generator = ObservationGenerator(test_binned, output, data)

# Call the generate_obs_dict method to get a single observation
obs_dict = obs_generator.generate_obs_dict()

# Call the generate_multiple_obs_dicts method to get multiple observations
num_samples = 10
obs_dicts = obs_generator.generate_multiple_obs_dicts(num_samples)

# Call the gen_ev_list method to get evidence lists from observation dictionaries
all_ev_list = obs_generator.gen_ev_list(obs_dicts)

# Assuming you have already obtained the join_tree from prob_dists function
# Call the get_all_posteriors method to get posteriors for all observations
obs_posteriors, predicted_posteriors = obs_generator.get_all_posteriors(all_ev_list, join_tree)
