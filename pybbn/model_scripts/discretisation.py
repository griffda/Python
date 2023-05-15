import pandas as pd
from sklearn.model_selection import train_test_split


def binning_data(file_path, test_size=0.2, x_cols=None, y_cols=None): ###Steps 2a (store inoput values), and 2b (store response values) into csv columns.
    # Read the CSV file
    ###This loads csv into a dataframe to be manipulated.
    df = pd.read_csv(file_path)

    # Check if x_cols and y_cols are provided, otherwise use default column names
    if x_cols is None:
        x_cols = df.columns[:-1].tolist()  # Select all columns except the last one as x_cols
    if y_cols is None:
        y_cols = [df.columns[-1]]  # Select the last column as y_cols


    x_df = df[x_cols]
    y_df = df[y_cols]

    labels = [1,2,3,4,5]
    number_of_bins = 5

    # Define empty dictionaries
    bin_edges_dict = {}
    prior_dict_xytrn = {}
    
    ### Apply equidistant binning to the input variables
    for col in x_df.columns:
        col_bins = col + '_bins'
        x_df.loc[:, col_bins], bin_edges = pd.cut(x_df.loc[:, col], number_of_bins, labels=labels, retbins=True)
        bin_edges_dict[col_bins] = bin_edges
        prior = x_df.loc[:, col_bins].value_counts(normalize=True).sort_index()
        prior_dict_xytrn[col + '_priors'] = prior.to_dict()

    ### Apply percentile binning to the output variable
    for col in y_df.columns:
        col_bins = col + '_bins'
        y_df.loc[:, col_bins], bin_edges = pd.qcut(y_df.loc[:, col], number_of_bins, labels=labels, retbins=True)
        bin_edges_dict[col_bins] = bin_edges
        prior = y_df[col_bins].value_counts(normalize=True).sort_index()
        prior_dict_xytrn[col + '_priors'] = prior.to_dict()

    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=test_size, random_state=42)
    # print(y_test.head())
    
    # Combine the binned data into a single DataFrame for each set
    df_train_binned = pd.concat([x_train.drop(x_cols, axis=1), y_train.drop(y_cols, axis=1)], axis=1)
    df_test_binned = pd.concat([x_test.drop(x_cols, axis=1), y_test], axis=1)
    

    ###Pybbn only reads data types as strings, so this line converts the data in the csv from int64 to string
    df_train_binned = df_train_binned.astype(str)

    return df_train_binned, df_test_binned, bin_edges_dict, prior_dict_xytrn

df_train_binned, df_test_binned, bin_edges_dict, prior_dict_xytrn = binning_data('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/outputv3.csv', 
                                                                                 0.4, x_cols=['mass', 'force'], y_cols=['acceleration'])
print(df_train_binned.head())