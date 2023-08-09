import bnmodel as bn
from bnmodel.discretisation import binning_data
from bnmodel.generate_posteriors import *
from bnmodel.join_tree_population import prob_dists
from sklearn.model_selection import KFold
import pandas as pd 


def k_fold_cross_validation(structure, data, output, numFolds, histnbins, nbins=5):
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
    norm_distance_errors = []
    prediction_accuracy = []
    av_prediction_accuracy = []
    for train_index, test_index in kf.split(x):
        x_train, x_test = x.iloc[train_index], x.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        print("Fold number: ", fold_counter + 1)
    
        # Combine the binned data into a single DataFrame for each set
        train_binned = pd.concat([x_train, y_train], axis=1)
        test_binned = pd.concat([x_test, y_test], axis=1)
    
        ###Pybbn only reads data types as strings, so this line converts the data in the csv from int64 to string
        train_binned = train_binned.astype(str)

        #generate the join tree and the probability distributions   
        join_tree = prob_dists(structure, train_binned)

        obs_dicts = bn.generate_posteriors.generate_multiple_obs_dicts(test_binned, output, data)
    
        all_ev_list = bn.generate_posteriors.gen_ev_list(test_binned, obs_dicts, output)

        obs_posteriors_dict, predicted_posteriors_list = bn.generate_posteriors.get_all_posteriors(all_ev_list, join_tree, output)

        correct_bin_locations, actual_values = bn.evaluate_errors.get_correct_values(obs_dicts, output)
        bin_ranges = bn.evaluate_errors.extract_bin_ranges(output, bin_edges)
        errors_aux, accuracy_aux = bn.evaluate_errors.distribution_distance_error(correct_bin_locations,
                                                                                                                predicted_posteriors_list,
                                                                                                                actual_values, 
                                                                                                                bin_ranges)
        
        norm_distance_errors.append(errors_aux)
        prediction_accuracy.append(accuracy_aux)

        #calculate average prediction accuracy for all folds
        av_prediction_accuracy = sum(prediction_accuracy) / len(prediction_accuracy)

        # ax = bn.evaluate_errors.plot_errors(norm_distance_errors, histnbins, numFolds, prediction_accuracy, plot=True)
        
        fold_counter += 1

    return obs_posteriors_dict, bin_edges, prior_xytrn, norm_distance_errors, prediction_accuracy, av_prediction_accuracy