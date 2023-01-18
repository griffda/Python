import pandas as pd
import numpy as np

df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/output2.csv',
                    index_col=False,
                    usecols=['m', 'theta','v0', 'vf', 'KE'],
                    encoding=('utf-8')
                    )

def binning_data(): 
    
    ##Create new data frame - call it binned and fill with the values and then use structure syntax below. 

    labels = [1,2,3,4]
    labels2 = [1,2,3,4,5,6]

    ###creates an oprn dictionary for us to fill with values. 
    ###dictionaries have two values: a key (in our case this will be the noes) and values (which will be the bins)
    bin_edges_dict = {}

    ###This is a more modular approach which uses qcut and a for loop to loop through each of the columns for the nodes
    ###providing bins, and calculating bin edges.
    ###cut command creates equispaced bins but frequency of samples is unequal in each bin - i.e., equidistant binning
    ###qcut command creates unequal size bins but frequency of samples is equal in each bin - i.e, percentile binning


    ###Equidistant binning for the inputs 
    for name in df.iloc[:,[0,1]]:
        name_bins = name + '_bins'
        df[name_bins], bin_edges = pd.cut(df[name], 4, labels=labels, retbins=True)
        bin_edges_dict[name_bins]=bin_edges
        


    ###Percentile binning for the outputs
    for name in df.iloc[:,[2,3,4]]:
        name_bins = name + '_bins'
        df[name_bins], bin_edges = pd.qcut(df[name], 4, labels=labels, retbins=True)
        bin_edges_dict[name_bins]=bin_edges


    df_binned = df.drop(['m', 'theta', 'v0', 'vf', 'KE'], axis=1)
    ###This line takes the first two columns of the new df to ensure the inputs get 
    #df_binned = df.iloc[:,[5,6]]
    print(bin_edges_dict)
    print(bin_edges)
    print(df_binned.head(10))

    df_binned.to_csv('binned_data.csv', index=False)
    return bin_edges, bin_edges_dict, df_binned 

binning_data()