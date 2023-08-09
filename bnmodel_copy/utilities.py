import pandas as pd

def df2struct(df, inputs, output):
    structure = {}
    for col in df.columns.tolist():
        if col in inputs:
            structure[col] = output
        else:
            structure[col] = []
    for out_var in output:
        structure[out_var] = []  # Set the output variable structure to an empty list
    return structure


def remove_parenthesis(df):
    """
    Remove parenthesis from the column names of a dataframe
    """
    df.columns = df.columns.str.replace('(', '')
    df.columns = df.columns.str.replace(')', '')
    return df

def prepare_csv(csv_path):
    """
    Prepare the csv file for the model
    """
    data = pd.read_csv(csv_path)
    data = remove_parenthesis(data)
    if 'run' in data.columns:
        data = data.drop('run', axis = 1)
    return data

