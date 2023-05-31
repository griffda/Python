import bnmodel as bn
from bnmodel.discretisation import binning_data
from bnmodel.generate_posteriors import *
from bnmodel.join_tree_population import prob_dists
from sklearn.model_selection import KFold
import pandas as pd 


def k_fold_cross_validation(structure, data, output, numFolds=5, nbins=5):
    """
        Apply k-fold cross validation to split the data into training and testing sets:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
        new function to split the data into training and testing sets

        Parameters
        ----------
        x : pandas dataframe with the input data
        y : pandas dataframe with the output data
        numFolds : int number of folds
        Returns
        -------
        train_binned : pandas dataframe with the input and output data for training
        test_binned : pandas dataframe with the input and output data for testing
        
    """

    x, y, bin_edges, prior_xytrn = binning_data(data, nbins=nbins, y_cols=[output])
    

    fold_counter = 0

    kf = KFold(n_splits=numFolds, shuffle=True, random_state=42) # 5-fold cross validation
    kf.get_n_splits(x)
    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print("Fold number: ", fold_counter + 1)
    
        # Combine the binned data into a single DataFrame for each set
        train_binned = pd.concat([x_train, y_train], axis=1)
        test_binned = pd.concat([x_test, y_test], axis=1)

        n_obs = len(test_binned)
    
        ###Pybbn only reads data types as strings, so this line converts the data in the csv from int64 to string
        train_binned = train_binned.astype(str)

        join_tree = prob_dists(structure, train_binned)

        obs_dicts = bn.generate_posteriors.generate_multiple_obs_dicts(test_binned, output, data, n_obs)
    
        all_ev_list = bn.generate_posteriors.gen_ev_list(test_binned, obs_dicts, output)

        obs_posteriors_dict, predicted_posteriors_list = bn.generate_posteriors.get_all_posteriors(all_ev_list, join_tree, output)

        fold_counter += 1

    #return train_binned, test_binned, bin_edges, prior_xytrn, join_tree
    return obs_posteriors_dict, predicted_posteriors_list, obs_dicts, bin_edges, prior_xytrn