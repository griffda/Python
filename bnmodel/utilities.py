import pandas as pd

def df2struct(df, inputs, outputs):
    """
    In process of being generalised
    """
    columns = df.columns.tolist()
    structure = {}
    for col in columns:
        if col != 'coe_bins':
            structure[col] = []
    structure['coe_bins'] = columns[:-1]
    return structure

def remove_parenthesis(df):
    """
    Remove parenthesis from the column names of a dataframe
    """
    df.columns = df.columns.str.replace('(', '')
    df.columns = df.columns.str.replace(')', '')
    return df