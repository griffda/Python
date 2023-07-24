import bnmodel as bn
from bnmodel.discretisation import binning_data
from bnmodel.generate_posteriors import *
from bnmodel.join_tree_population import prob_dists
from sklearn.model_selection import KFold
import pandas as pd

class CrossValidator:
    def __init__(self, structure, output, numFolds, histnbins, nbins=5):
        self.structure = structure
        self.output = output
        self.numFolds = numFolds
        self.histnbins = histnbins
        self.nbins = nbins
        self.join_tree = None
        self.obs_posteriors_dict = None
        self.bin_edges = None
        self.prior_xytrn = None
        self.norm_distance_errors = None
        self.prediction_accuracy = None
        self.av_prediction_accuracy = None

    def k_fold_cross_validation(self, data):
        """
        Apply k-fold cross validation to split the data into training and testing sets.

        Parameters
        ----------
        data : pandas dataframe with the input and output data

        Returns
        -------
        None (sets the class attributes with results)
        """
        x, y, bin_edges, prior_xytrn = binning_data(data, nbins=self.nbins, y_cols=[self.output])

        fold_counter = 0

        kf = KFold(n_splits=self.numFolds, shuffle=True, random_state=42) # 5-fold cross-validation
        kf.get_n_splits(x)
        self.norm_distance_errors = []
        self.prediction_accuracy = []
        av_prediction_accuracy = []

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

            # Generate the join tree and the probability distributions
            self.join_tree = prob_dists(self.structure, train_binned)

            obs_dicts = bn.generate_posteriors.generate_multiple_obs_dicts(test_binned, self.output, data, n_obs)

            all_ev_list = bn.generate_posteriors.gen_ev_list(test_binned, obs_dicts, self.output)

            self.obs_posteriors_dict, predicted_posteriors_list = bn.generate_posteriors.get_all_posteriors(all_ev_list, self.join_tree, self.output)

            correct_bin_locations, actual_values = bn.evaluate_errors.get_correct_values(obs_dicts, self.output)
            bin_ranges = bn.evaluate_errors.extract_bin_ranges(self.output, bin_edges)
            errors_aux, accuracy_aux = bn.evaluate_errors.distribution_distance_error(correct_bin_locations,
                                                                                                                predicted_posteriors_list,
                                                                                                                actual_values,
                                                                                                                bin_ranges)

            self.norm_distance_errors.append(errors_aux)
            self.prediction_accuracy.append(accuracy_aux)

            # Calculate average prediction accuracy for all folds
            av_prediction_accuracy = sum(self.prediction_accuracy) / len(self.prediction_accuracy)

            # ax = bn.evaluate_errors.plot_errors(norm_distance_errors, histnbins, numFolds, prediction_accuracy, plot=True)

            fold_counter += 1

        self.av_prediction_accuracy = av_prediction_accuracy
