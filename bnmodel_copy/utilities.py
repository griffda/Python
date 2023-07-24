import pandas as pd

class BNdata:
    def __init__(self, csv_path, inputs, output):
        self.csv_path = csv_path
        self.inputs = inputs
        self.output = output
        self.data = self.prepare_csv()  # Call the prepare_csv method to prepare the data

    def prepare_csv(self):
        """
        Prepare the csv file for the model
        """
        data = pd.read_csv(self.csv_path)
        data = self.remove_parenthesis(data)
        if 'run' in data.columns:
            data = data.drop('run', axis=1)
        return data

    @staticmethod
    def remove_parenthesis(df):
        """
        Remove parenthesis from the column names of a dataframe
        """
        df.columns = df.columns.str.replace('(', '')
        df.columns = df.columns.str.replace(')', '')
        return df

    def df2struct(self):
        structure = {}
        for col in self.data.columns.tolist():
            if col in self.inputs:
                structure[col] = [self.output]
            elif col == self.output:
                structure[col] = []
            else:
                structure[col] = [self.output]
        return structure


