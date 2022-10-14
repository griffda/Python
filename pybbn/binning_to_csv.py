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
    for name in df.columns:
        name_bins = name + '_bins'
        df[name_bins], bin_edges = pd.qcut(df[name], 4, labels=labels, retbins=True)
        bin_edges_dict[name_bins]=bin_edges

        
    # df['m_bins'], bin_edges = pd.qcut(df['m'], 4, labels=labels,retbins=True)
    # df['theta_bins'], bin_edges = pd.qcut(df['theta'], 4, labels=labels,retbins=True)
    # df['v0_bins'], bin_edges = pd.qcut(df['v0'], 4, labels=labels,retbins=True)
    # df['vf_bins'], bin_edges = pd.qcut(df['vf'], 6, labels=labels2,retbins=True)
    # df['KE_bins'], bin_edges = pd.qcut(df['KE'], 6, labels=labels2,retbins=True)

    df_binned = df.drop(['m', 'theta', 'v0', 'vf', 'KE'], axis=1)
    print(bin_edges_dict)
    print(bin_edges)
    print(df_binned.head(10))

    df_binned.to_csv('binned_data.csv', index=False)
    return bin_edges, bin_edges_dict, df_binned 

binning_data()