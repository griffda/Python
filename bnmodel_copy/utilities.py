import pandas as pd

# def df2struct(df, inputs, output):
#     structure = {}
#     for col in df.columns.tolist():
#         if col in inputs:
#             structure[col] = output
#         else:
#             structure[col] = []
#     for out_var in output:
#         structure[out_var] = []  # Set the output variable structure to an empty list
#     return structure

def df2struct(df, inputs, output):
    structure = {}
    for col in df.columns.tolist():
        if col in inputs:
            structure[col] = output
        else:
            structure[col] = []
    return structure


def remove_parenthesis(df):
    """
    Remove parenthesis from the column names of a dataframe
    """
    df.columns = df.columns.str.replace('(', '')
    df.columns = df.columns.str.replace(')', '')
    return df

def remove_square_brackets(df):
    """
    Remove square brackets from the column names of a dataframe
    """
    df.columns = df.columns.str.replace('[', '')
    df.columns = df.columns.str.replace(']', '')
    return df

def prepare_csv(csv_path, inputs, outputs):
    """
    Prepare the csv file for the model
    """
    data = pd.read_csv(csv_path)
    data = remove_parenthesis(data)
    data = remove_square_brackets(data)
    if 'run' in data.columns:
        data = data.drop('run', axis = 1)
    
    # Replace spaces in column names with underscores
    data.columns = data.columns.str.replace(' ', '_')
    
    # Only keep the columns specified as inputs and outputs
    data = data[inputs + outputs]
    
    return data

def find_best_evidence_for_bin(capcost_predictions, bin_index):
    """
    Find the evidence configuration with the highest probability for the specified bin.

    Parameters
    ----------
    capcost_predictions : dict
        Dictionary where the keys are the evidence configurations and the values are tuples containing the highest probability and the corresponding bin index.
    bin_index : int
        The bin index to find the best evidence configuration for.

    Returns
    -------
    best_evidence : list
        The best evidence configurations for the specified bin.
    """
    best_evidence = [evidence for evidence, (probability, bin_idx) in capcost_predictions.items() if bin_idx == bin_index and probability == max(probability for _, (probability, _) in capcost_predictions.items() if bin_idx == bin_index)]

    return best_evidence

