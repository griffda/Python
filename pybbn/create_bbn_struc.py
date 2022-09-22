import pandas as pd
from pybbn.graph.factory import Factory

df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/pybbn/data-from-structure.csv')
structure = {
    'a': [],
    'b': ['a'],
    'c': ['b']
}

print(df.head(10))

bbn = Factory.from_data(structure, df)