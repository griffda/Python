

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